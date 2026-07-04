from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.grader import r1_zero_reward_fn
from src.rsft import (
    build_sampling_params,
    canonicalize_response_for_grader,
    load_generation_examples,
)
from src.sft import (
    resolve_dtype,
    resolve_project_path,
    set_seed,
    setup_logger,
    setup_wandb,
)


@dataclass
class DPOExample:
    index: int
    prompt: str
    chosen: str
    rejected: str
    question: str
    ground_truth: str
    chosen_reward_info: Dict[str, float]
    rejected_reward_info: Dict[str, float]
    metadata: Dict[str, Any]


class MathDPODataset(Dataset):
    def __init__(self, examples: List[DPOExample], tokenizer, max_length: int, add_eos_token: bool):
        self.features = []
        for ex in examples:
            chosen_feature = self._encode_pair(ex.prompt, ex.chosen, tokenizer, max_length, add_eos_token)
            rejected_feature = self._encode_pair(ex.prompt, ex.rejected, tokenizer, max_length, add_eos_token)
            if chosen_feature is None or rejected_feature is None:
                continue

            self.features.append(
                {
                    "index": ex.index,
                    "question": ex.question,
                    "ground_truth": ex.ground_truth,
                    "chosen": ex.chosen,
                    "rejected": ex.rejected,
                    "chosen_reward_info": ex.chosen_reward_info,
                    "rejected_reward_info": ex.rejected_reward_info,
                    "metadata": ex.metadata,
                    "chosen_input_ids": chosen_feature["input_ids"],
                    "chosen_response_mask": chosen_feature["response_mask"],
                    "rejected_input_ids": rejected_feature["input_ids"],
                    "rejected_response_mask": rejected_feature["response_mask"],
                }
            )

    @staticmethod
    def _encode_pair(prompt: str, response: str, tokenizer, max_length: int, add_eos_token: bool):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if add_eos_token:
            response_ids.append(tokenizer.eos_token_id)

        max_response_len = max_length - len(prompt_ids)
        if max_response_len <= 0:
            return None

        response_ids = response_ids[:max_response_len]
        if len(response_ids) == 0:
            return None

        return {
            "input_ids": prompt_ids + response_ids,
            "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


class DPODataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad(self, features: List[Dict[str, Any]], input_key: str, mask_key: str):
        max_len = max(len(f[input_key]) for f in features)
        max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        input_ids, attention_mask, response_mask = [], [], []
        for f in features:
            pad_len = max_len - len(f[input_key])
            input_ids.append(f[input_key] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(f[input_key]) + [0] * pad_len)
            response_mask.append(f[mask_key] + [0] * pad_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(response_mask, dtype=torch.bool),
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosen_input_ids, chosen_attention_mask, chosen_response_mask = self._pad(
            features,
            "chosen_input_ids",
            "chosen_response_mask",
        )
        rejected_input_ids, rejected_attention_mask, rejected_response_mask = self._pad(
            features,
            "rejected_input_ids",
            "rejected_response_mask",
        )

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_response_mask": chosen_response_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_response_mask": rejected_response_mask,
            "indices": [f["index"] for f in features],
            "questions": [f["question"] for f in features],
            "ground_truths": [f["ground_truth"] for f in features],
            "chosen": [f["chosen"] for f in features],
            "rejected": [f["rejected"] for f in features],
        }


def sequence_log_probs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:].float()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    sequence_logp = (token_log_probs * shift_mask).sum(dim=-1)
    response_tokens = shift_mask.sum(dim=-1)
    return sequence_logp, response_tokens


def compute_dpo_loss(policy_model, ref_model, batch: Dict[str, torch.Tensor], beta: float):
    chosen_logp, chosen_tokens = sequence_log_probs(
        policy_model,
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_response_mask"],
    )
    rejected_logp, rejected_tokens = sequence_log_probs(
        policy_model,
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_response_mask"],
    )

    with torch.no_grad():
        chosen_ref_logp, _ = sequence_log_probs(
            ref_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"],
        )
        rejected_ref_logp, _ = sequence_log_probs(
            ref_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"],
        )

    chosen_log_ratio = chosen_logp - chosen_ref_logp
    rejected_log_ratio = rejected_logp - rejected_ref_logp
    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        info = {
            "loss": float(loss.detach().cpu()),
            "reward_accuracy": float((logits > 0).float().mean().detach().cpu()),
            "reward_margin": float((chosen_log_ratio - rejected_log_ratio).mean().detach().cpu()),
            "chosen_logp": float(chosen_logp.mean().detach().cpu()),
            "rejected_logp": float(rejected_logp.mean().detach().cpu()),
            "chosen_ref_logp": float(chosen_ref_logp.mean().detach().cpu()),
            "rejected_ref_logp": float(rejected_ref_logp.mean().detach().cpu()),
            "chosen_tokens": float(chosen_tokens.mean().detach().cpu()),
            "rejected_tokens": float(rejected_tokens.mean().detach().cpu()),
        }
    return loss, info


def plot_dpo_training_curves(history: List[Dict[str, Any]], output_dir, cfg: Dict[str, Any]):
    if not cfg["plots"]["enabled"] or not history:
        return

    plt.style.use(cfg["plots"]["style"])
    steps = [row["global_step"] for row in history]
    panels = [
        ("loss", "Loss", "#31688e"),
        ("reward_accuracy", "Preference Accuracy", "#35b779"),
        ("reward_margin", "Reward Margin", "#f89540"),
        ("grad_norm", "Gradient Norm", "#cc4778"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("DPO Training Dynamics", fontsize=16, fontweight="bold")
    for ax, (key, title, color) in zip(axes.ravel(), panels):
        values = [row[key] for row in history]
        ax.plot(steps, values, color=color, linewidth=2.0)
        ax.scatter(steps[-1:], values[-1:], color=color, s=32, zorder=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Optimizer Step")
        ax.grid(True, alpha=0.28)

    fig.savefig(output_dir / "training_curves.png", dpi=int(cfg["plots"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def generate_dpo_data(cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    from vllm import LLM

    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_data_path = resolve_project_path(cfg["paths"]["dpo_data_path"])
    dpo_data_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg["dpo"]["skip_sampling_if_exists"] and dpo_data_path.exists():
        logger.info(f"DPO data already exists, skip sampling: {dpo_data_path}")
        return {"dpo_data_path": str(dpo_data_path), "skipped": True}

    examples = load_generation_examples(cfg, logger)
    prompts = [ex["prompt"] for ex in examples]

    generation_cfg = cfg["generation"]
    llm = LLM(
        model=str(resolve_project_path(cfg["paths"]["model_path"])),
        tensor_parallel_size=int(generation_cfg["tensor_parallel_size"]),
        dtype=generation_cfg["dtype"],
        gpu_memory_utilization=float(generation_cfg["gpu_memory_utilization"]),
        max_model_len=int(generation_cfg["max_model_len"]),
        trust_remote_code=bool(generation_cfg["trust_remote_code"]),
    )
    sampling_params = build_sampling_params(cfg)

    logger.info("Running DPO preference sampling")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - start_time

    grading_cfg = cfg["grading"]
    fast = bool(grading_cfg["fast"])
    normalize_tag_whitespace = bool(grading_cfg["normalize_tag_whitespace"])
    chosen_threshold = float(cfg["dpo"]["chosen_reward_threshold"])
    rejected_threshold = float(cfg["dpo"]["rejected_reward_threshold"])
    max_pairs_per_prompt = int(cfg["dpo"]["max_pairs_per_prompt"])

    total_candidates = 0
    chosen_candidates = 0
    rejected_candidates = 0
    pairs_written = 0
    prompts_with_pairs = 0

    with open(dpo_data_path, "w", encoding="utf-8") as fout:
        for ex, out in tqdm(zip(examples, outputs), total=len(examples), desc="Building DPO pairs"):
            chosen_pool = []
            rejected_pool = []

            for sample_id, candidate in enumerate(out.outputs):
                total_candidates += 1
                raw_response = candidate.text
                graded_response = (
                    canonicalize_response_for_grader(raw_response)
                    if normalize_tag_whitespace
                    else raw_response
                )
                reward_info = r1_zero_reward_fn(graded_response, ex["ground_truth"], fast=fast)
                reward = float(reward_info["reward"])

                record = {
                    "sample_id": sample_id,
                    "response": graded_response,
                    "reward_info": reward_info,
                }
                if reward >= chosen_threshold:
                    chosen_pool.append(record)
                    chosen_candidates += 1
                elif reward <= rejected_threshold:
                    rejected_pool.append(record)
                    rejected_candidates += 1

            if not chosen_pool or not rejected_pool:
                continue

            num_pairs = min(max_pairs_per_prompt, len(chosen_pool) * len(rejected_pool))
            for pair_id in range(num_pairs):
                chosen = chosen_pool[pair_id % len(chosen_pool)]
                rejected = rejected_pool[(pair_id // len(chosen_pool)) % len(rejected_pool)]
                row = {
                    "index": ex["index"],
                    "pair_id": pair_id,
                    "question": ex["question"],
                    "ground_truth": ex["ground_truth"],
                    "prompt": ex["prompt"],
                    "chosen": chosen["response"],
                    "rejected": rejected["response"],
                    "chosen_sample_id": chosen["sample_id"],
                    "rejected_sample_id": rejected["sample_id"],
                    "chosen_reward_info": chosen["reward_info"],
                    "rejected_reward_info": rejected["reward_info"],
                    "metadata": ex["metadata"],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                pairs_written += 1

            prompts_with_pairs += 1

    summary = {
        "dpo_data_path": str(dpo_data_path),
        "num_prompts": len(examples),
        "total_candidates": total_candidates,
        "chosen_candidates": chosen_candidates,
        "rejected_candidates": rejected_candidates,
        "pairs_written": pairs_written,
        "prompts_with_pairs": prompts_with_pairs,
        "pair_prompt_coverage": prompts_with_pairs / len(examples) if examples else 0.0,
        "generation_elapsed_seconds": elapsed,
    }
    with open(output_dir / "sampling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("DPO sampling summary:")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def sample_dpo(cfg: Dict[str, Any], config_path: str):
    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "sample.log")
    logger.info("Starting DPO sampling")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Output dir: {output_dir}")

    set_seed(int(cfg["task"]["seed"]))
    summary = generate_dpo_data(cfg, logger)

    logger.info("DPO sampling finished")
    return summary


def load_dpo_examples(cfg: Dict[str, Any], logger) -> List[DPOExample]:
    dpo_data_path = resolve_project_path(cfg["paths"]["dpo_data_path"])
    examples: List[DPOExample] = []

    with open(dpo_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                DPOExample(
                    index=int(row["index"]),
                    prompt=row["prompt"],
                    chosen=row["chosen"],
                    rejected=row["rejected"],
                    question=row["question"],
                    ground_truth=str(row["ground_truth"]),
                    chosen_reward_info=row["chosen_reward_info"],
                    rejected_reward_info=row["rejected_reward_info"],
                    metadata=row.get("metadata", {}),
                )
            )

    logger.info(f"Loaded DPO examples: {len(examples)} from {dpo_data_path}")
    if len(examples) == 0:
        raise ValueError("No DPO examples loaded. Run sampling first or check reward thresholds.")
    return examples


def train_dpo(cfg: Dict[str, Any], config_path: str):
    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    final_model_dir = resolve_project_path(cfg["paths"]["final_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "train.log")
    logger.info("Starting DPO")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Final model dir: {final_model_dir}")

    set_seed(int(cfg["task"]["seed"]))

    model_path = resolve_project_path(cfg["paths"]["model_path"])
    reference_model_path = resolve_project_path(cfg["paths"]["reference_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=cfg["model"]["trust_remote_code"])
    tokenizer.pad_token = tokenizer.eos_token

    examples = load_dpo_examples(cfg, logger)
    dataset = MathDPODataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=int(cfg["dataset"]["max_length"]),
        add_eos_token=cfg["dataset"]["add_eos_token"],
    )
    if len(dataset) == 0:
        raise ValueError("All DPO examples were filtered by max_length.")

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["train_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=cfg["training"]["pin_memory"],
        collate_fn=DPODataCollator(tokenizer),
    )

    model_kwargs = {
        "torch_dtype": resolve_dtype(cfg["model"]["torch_dtype"]),
        "trust_remote_code": cfg["model"]["trust_remote_code"],
    }
    if cfg["model"]["attn_implementation"]:
        model_kwargs["attn_implementation"] = cfg["model"]["attn_implementation"]

    policy_model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(str(reference_model_path), **model_kwargs)
    policy_model.config.use_cache = False
    ref_model.config.use_cache = False

    if cfg["model"]["gradient_checkpointing"]:
        policy_model.gradient_checkpointing_enable()

    device = torch.device("cuda")
    policy_model.to(device)
    ref_model.to(device)
    policy_model.train()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=tuple(float(x) for x in cfg["training"]["betas"]),
        eps=float(cfg["training"]["eps"]),
    )

    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
    num_epochs = int(cfg["training"]["num_train_epochs"])
    total_steps = math.ceil(len(dataloader) / grad_accum) * num_epochs
    warmup_steps = int(total_steps * float(cfg["training"]["warmup_ratio"]))
    scheduler = get_scheduler(
        name=cfg["training"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    beta = float(cfg["dpo"]["beta"])
    wandb_run = setup_wandb(cfg, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    history: List[Dict[str, Any]] = []
    global_step = 0
    start_time = time.time()

    logger.info(f"Training examples: {len(dataset)}")
    logger.info(f"Epochs: {num_epochs}, micro-batches per epoch: {len(dataloader)}")
    logger.info(f"Gradient accumulation steps: {grad_accum}, optimizer steps: {total_steps}")
    logger.info(f"DPO beta: {beta}")

    optimizer.zero_grad(set_to_none=True)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for epoch in range(num_epochs):
            running = {
                "loss": 0.0,
                "reward_accuracy": 0.0,
                "reward_margin": 0.0,
                "chosen_logp": 0.0,
                "rejected_logp": 0.0,
                "chosen_tokens": 0.0,
                "rejected_tokens": 0.0,
                "micro_steps": 0,
            }
            progress = tqdm(dataloader, desc=f"DPO epoch {epoch + 1}/{num_epochs}")

            for micro_step, batch in enumerate(progress, start=1):
                tensor_batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                loss, loss_info = compute_dpo_loss(policy_model, ref_model, tensor_batch, beta)
                (loss / grad_accum).backward()

                for key in running:
                    if key == "micro_steps":
                        continue
                    running[key] += loss_info[key]
                running["micro_steps"] += 1

                if micro_step % grad_accum != 0 and micro_step != len(dataloader):
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(),
                    float(cfg["training"]["max_grad_norm"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                denom = running["micro_steps"]
                record = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "loss": running["loss"] / denom,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "reward_accuracy": running["reward_accuracy"] / denom,
                    "reward_margin": running["reward_margin"] / denom,
                    "chosen_logp": running["chosen_logp"] / denom,
                    "rejected_logp": running["rejected_logp"] / denom,
                    "chosen_tokens": running["chosen_tokens"] / denom,
                    "rejected_tokens": running["rejected_tokens"] / denom,
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
                history.append(record)
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_file.flush()

                if wandb_run is not None:
                    wandb_run.log({f"train/{k}": v for k, v in record.items() if k != "global_step"}, step=global_step)

                if global_step % int(cfg["training"]["log_every"]) == 0:
                    logger.info(
                        "step=%d epoch=%d loss=%.6f lr=%.3e reward_acc=%.4f margin=%.4f grad_norm=%.4f",
                        global_step,
                        epoch + 1,
                        record["loss"],
                        record["learning_rate"],
                        record["reward_accuracy"],
                        record["reward_margin"],
                        record["grad_norm"],
                    )

                progress.set_postfix(
                    step=global_step,
                    loss=f"{record['loss']:.4f}",
                    acc=f"{record['reward_accuracy']:.3f}",
                    lr=f"{record['learning_rate']:.2e}",
                )
                running = {
                    "loss": 0.0,
                    "reward_accuracy": 0.0,
                    "reward_margin": 0.0,
                    "chosen_logp": 0.0,
                    "rejected_logp": 0.0,
                    "chosen_tokens": 0.0,
                    "rejected_tokens": 0.0,
                    "micro_steps": 0,
                }

    plot_dpo_training_curves(history, output_dir, cfg)
    policy_model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    summary = {
        "task": cfg["task"]["name"],
        "stage": cfg["task"]["stage"],
        "config_path": config_path,
        "model_path": str(model_path),
        "reference_model_path": str(reference_model_path),
        "dpo_data_path": str(resolve_project_path(cfg["paths"]["dpo_data_path"])),
        "beta": beta,
        "num_examples": len(dataset),
        "global_steps": global_step,
        "metrics_path": str(metrics_path),
        "plot_path": str(output_dir / "training_curves.png"),
        "final_model_dir": str(final_model_dir),
        "elapsed_seconds": time.time() - start_time,
        "last_metrics": history[-1] if history else None,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    logger.info("Saved final model: %s", final_model_dir)
    logger.info("DPO finished")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
