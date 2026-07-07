from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.grader import r1_zero_reward_fn
from src.rsft import canonicalize_response_for_grader, load_generation_examples
from src.sft import (
    resolve_dtype,
    resolve_project_path,
    set_seed,
    setup_logger,
    setup_wandb,
)


@dataclass
class GRPORollout:
    index: int
    group_id: int
    sample_id: int
    prompt: str
    response: str
    question: str
    ground_truth: str
    reward: float
    advantage: float
    reward_info: Dict[str, float]
    metadata: Dict[str, Any]


class MathGRPODataset(Dataset):
    def __init__(self, rollouts: List[GRPORollout], tokenizer, max_length: int, add_eos_token: bool):
        self.features: List[Dict[str, Any]] = []
        for rollout in rollouts:
            prompt_ids = tokenizer(rollout.prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(rollout.response, add_special_tokens=False)["input_ids"]
            if add_eos_token:
                response_ids.append(tokenizer.eos_token_id)

            max_response_len = max_length - len(prompt_ids)
            if max_response_len <= 0:
                continue

            response_ids = response_ids[:max_response_len]
            if len(response_ids) == 0:
                continue

            input_ids = prompt_ids + response_ids
            response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            self.features.append(
                {
                    "index": rollout.index,
                    "group_id": rollout.group_id,
                    "sample_id": rollout.sample_id,
                    "input_ids": input_ids,
                    "response_mask": response_mask,
                    "advantage": float(rollout.advantage),
                    "reward": float(rollout.reward),
                    "prompt": rollout.prompt,
                    "response": rollout.response,
                    "question": rollout.question,
                    "ground_truth": rollout.ground_truth,
                    "reward_info": rollout.reward_info,
                    "metadata": rollout.metadata,
                    "old_token_logps": None,
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


class GRPODataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        input_ids, attention_mask, response_mask = [], [], []
        old_token_logps = []
        has_old_logps = all(f.get("old_token_logps") is not None for f in features)

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * seq_len + [0] * pad_len)
            response_mask.append(f["response_mask"] + [0] * pad_len)

            if has_old_logps:
                old_values = f["old_token_logps"]
                old_pad_len = (max_len - 1) - len(old_values)
                old_token_logps.append(old_values + [0.0] * old_pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.bool),
            "advantages": torch.tensor([f["advantage"] for f in features], dtype=torch.float32),
            "rewards": torch.tensor([f["reward"] for f in features], dtype=torch.float32),
            "indices": [f["index"] for f in features],
            "group_ids": [f["group_id"] for f in features],
            "sample_ids": [f["sample_id"] for f in features],
            "questions": [f["question"] for f in features],
            "responses": [f["response"] for f in features],
            "ground_truths": [f["ground_truth"] for f in features],
        }
        if has_old_logps:
            batch["old_token_logps"] = torch.tensor(old_token_logps, dtype=torch.float32)
        return batch


def truncate_at_stop(text: str, stop_strings: Optional[List[str]], include_stop: bool) -> str:
    if not stop_strings:
        return text

    best_pos = None
    best_stop = None
    for stop in stop_strings:
        pos = text.find(stop)
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_pos = pos
            best_stop = stop

    if best_pos is None:
        return text

    end = best_pos + len(best_stop) if include_stop else best_pos
    return text[:end]


def sample_rollout_examples(examples: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
    if batch_size >= len(examples):
        return random.sample(examples, len(examples))
    return random.sample(examples, batch_size)


def generate_responses_with_policy(policy_model, tokenizer, prompts: List[str], cfg: Dict[str, Any]) -> List[str]:
    sampling_cfg = cfg["sampling"]
    grpo_cfg = cfg["grpo"]
    generation_batch_size = int(grpo_cfg.get("generation_batch_size", cfg["training"]["train_batch_size"]))

    was_training = policy_model.training
    policy_model.eval()

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    responses: List[str] = []
    device = next(policy_model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(prompts), generation_batch_size):
            batch_prompts = prompts[start : start + generation_batch_size]
            encoded = tokenizer(
                batch_prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]

            generated = policy_model.generate(
                **encoded,
                do_sample=float(sampling_cfg["temperature"]) > 0.0,
                temperature=max(float(sampling_cfg["temperature"]), 1.0e-6),
                top_p=float(sampling_cfg["top_p"]),
                max_new_tokens=int(sampling_cfg["max_tokens"]),
                min_new_tokens=int(sampling_cfg.get("min_tokens", 0)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            new_token_ids = generated[:, prompt_width:]
            decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
            for text in decoded:
                responses.append(
                    truncate_at_stop(
                        text,
                        sampling_cfg.get("stop"),
                        bool(sampling_cfg["include_stop_str_in_output"]),
                    )
                )

    tokenizer.padding_side = old_padding_side
    if was_training:
        policy_model.train()
    return responses


def generate_grpo_rollouts(policy_model, tokenizer, cfg: Dict[str, Any], logger, examples: List[Dict[str, Any]]):
    grpo_cfg = cfg["grpo"]
    group_size = int(grpo_cfg["group_size"])
    rollout_batch_size = int(grpo_cfg["rollout_batch_size"])
    normalize_advantage = bool(grpo_cfg["normalize_advantage"])
    advantage_eps = float(grpo_cfg["advantage_eps"])

    rollout_examples = sample_rollout_examples(examples, rollout_batch_size)
    repeated_prompts = []
    for ex in rollout_examples:
        repeated_prompts.extend([ex["prompt"]] * group_size)

    responses = generate_responses_with_policy(policy_model, tokenizer, repeated_prompts, cfg)
    grading_cfg = cfg["grading"]
    fast = bool(grading_cfg["fast"])
    normalize_tag_whitespace = bool(grading_cfg["normalize_tag_whitespace"])

    rollouts: List[GRPORollout] = []
    group_summaries = []
    response_cursor = 0

    for group_id, ex in enumerate(rollout_examples):
        group_records = []
        rewards = []

        for sample_id in range(group_size):
            raw_response = responses[response_cursor]
            response_cursor += 1
            graded_response = (
                canonicalize_response_for_grader(raw_response)
                if normalize_tag_whitespace
                else raw_response
            )
            reward_info = r1_zero_reward_fn(graded_response, ex["ground_truth"], fast=fast)
            reward = float(reward_info["reward"])
            rewards.append(reward)
            group_records.append(
                {
                    "sample_id": sample_id,
                    "response": graded_response,
                    "reward": reward,
                    "reward_info": reward_info,
                }
            )

        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards))
        if normalize_advantage:
            advantages = [(reward - reward_mean) / (reward_std + advantage_eps) for reward in rewards]
        else:
            advantages = [reward - reward_mean for reward in rewards]

        group_summaries.append(
            {
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "has_correct": any(reward == 1.0 for reward in rewards),
                "active": any(abs(advantage) > 0.0 for advantage in advantages),
            }
        )

        for record, advantage in zip(group_records, advantages):
            rollouts.append(
                GRPORollout(
                    index=int(ex["index"]),
                    group_id=group_id,
                    sample_id=int(record["sample_id"]),
                    prompt=ex["prompt"],
                    response=record["response"],
                    question=ex["question"],
                    ground_truth=str(ex["ground_truth"]),
                    reward=float(record["reward"]),
                    advantage=float(advantage),
                    reward_info=record["reward_info"],
                    metadata=ex["metadata"],
                )
            )

    summary = {
        "num_groups": len(group_summaries),
        "num_rollouts": len(rollouts),
        "reward_mean": float(np.mean([r.reward for r in rollouts])) if rollouts else 0.0,
        "reward_std": float(np.std([r.reward for r in rollouts])) if rollouts else 0.0,
        "advantage_mean": float(np.mean([r.advantage for r in rollouts])) if rollouts else 0.0,
        "advantage_abs_mean": float(np.mean([abs(r.advantage) for r in rollouts])) if rollouts else 0.0,
        "pass_at_group": float(np.mean([g["has_correct"] for g in group_summaries])) if group_summaries else 0.0,
        "active_group_rate": float(np.mean([g["active"] for g in group_summaries])) if group_summaries else 0.0,
    }
    logger.info("GRPO rollout summary: %s", json.dumps(summary, ensure_ascii=False))
    return rollouts, summary


def token_log_probs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:].float()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logps, shift_mask


def attach_old_token_logps(policy_model, dataset: MathGRPODataset, tokenizer, cfg: Dict[str, Any], device):
    collator = GRPODataCollator(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["train_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    was_training = policy_model.training
    policy_model.eval()

    cursor = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            response_mask = batch["response_mask"].to(device)
            old_logps, _ = token_log_probs(policy_model, input_ids, attention_mask, response_mask)
            old_logps = old_logps.detach().cpu()

            for row in range(input_ids.shape[0]):
                feature = dataset.features[cursor]
                seq_len = len(feature["input_ids"])
                feature["old_token_logps"] = old_logps[row, : seq_len - 1].tolist()
                cursor += 1

    if was_training:
        policy_model.train()


def compute_grpo_loss(policy_model, batch: Dict[str, torch.Tensor], clip_eps: float):
    new_logps, response_mask = token_log_probs(
        policy_model,
        batch["input_ids"],
        batch["attention_mask"],
        batch["response_mask"],
    )
    old_logps = batch["old_token_logps"]
    advantages = batch["advantages"].unsqueeze(1)

    log_ratio = new_logps - old_logps
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    objective = torch.minimum(unclipped, clipped)

    denom = response_mask.sum().clamp_min(1.0)
    loss = -((objective * response_mask).sum() / denom)

    with torch.no_grad():
        approx_kl = ((old_logps - new_logps) * response_mask).sum() / denom
        clip_frac = (((ratio - 1.0).abs() > clip_eps).float() * response_mask).sum() / denom
        response_tokens = response_mask.sum(dim=-1).float().mean()
        info = {
            "loss": float(loss.detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "clip_frac": float(clip_frac.detach().cpu()),
            "response_tokens": float(response_tokens.detach().cpu()),
            "ratio_mean": float(((ratio * response_mask).sum() / denom).detach().cpu()),
            "advantage_mean": float(batch["advantages"].mean().detach().cpu()),
            "advantage_abs_mean": float(batch["advantages"].abs().mean().detach().cpu()),
            "reward_mean": float(batch["rewards"].mean().detach().cpu()),
        }
    return loss, info


def plot_grpo_training_curves(history: List[Dict[str, Any]], output_dir: Path, cfg: Dict[str, Any]):
    if not cfg["plots"]["enabled"] or not history:
        return

    plt.style.use(cfg["plots"]["style"])
    steps = [row["global_step"] for row in history]
    panels = [
        ("loss", "Loss", "#31688e"),
        ("rollout_reward_mean", "Rollout Reward", "#35b779"),
        ("approx_kl", "Approx KL", "#f89540"),
        ("clip_frac", "Clip Fraction", "#cc4778"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("GRPO Training Dynamics", fontsize=16, fontweight="bold")
    for ax, (key, title, color) in zip(axes.ravel(), panels):
        values = [row[key] for row in history]
        ax.plot(steps, values, color=color, linewidth=2.0)
        ax.scatter(steps[-1:], values[-1:], color=color, s=32, zorder=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Optimizer Step")
        ax.grid(True, alpha=0.28)

    fig.savefig(output_dir / "training_curves.png", dpi=int(cfg["plots"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def train_grpo(cfg: Dict[str, Any], config_path: str):
    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    final_model_dir = resolve_project_path(cfg["paths"]["final_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "train.log")
    logger.info("Starting GRPO")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Final model dir: {final_model_dir}")

    set_seed(int(cfg["task"]["seed"]))

    model_path = resolve_project_path(cfg["paths"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=cfg["model"]["trust_remote_code"])
    tokenizer.pad_token = tokenizer.eos_token

    examples = load_generation_examples(cfg, logger)
    if len(examples) == 0:
        raise ValueError("No generation examples loaded for GRPO.")

    model_kwargs = {
        "torch_dtype": resolve_dtype(cfg["model"]["torch_dtype"]),
        "trust_remote_code": cfg["model"]["trust_remote_code"],
    }
    if cfg["model"]["attn_implementation"]:
        model_kwargs["attn_implementation"] = cfg["model"]["attn_implementation"]

    policy_model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    policy_model.config.use_cache = False
    if cfg["model"]["gradient_checkpointing"]:
        policy_model.gradient_checkpointing_enable()

    device = torch.device("cuda")
    policy_model.to(device)
    policy_model.train()

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=tuple(float(x) for x in cfg["training"]["betas"]),
        eps=float(cfg["training"]["eps"]),
    )

    n_grpo_steps = int(cfg["grpo"]["n_grpo_steps"])
    epochs_per_rollout_batch = int(cfg["grpo"]["epochs_per_rollout_batch"])
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
    rollout_batch_size = int(cfg["grpo"]["rollout_batch_size"])
    group_size = int(cfg["grpo"]["group_size"])
    train_batch_size = int(cfg["training"]["train_batch_size"])
    batches_per_rollout = math.ceil((rollout_batch_size * group_size) / train_batch_size)
    total_micro_steps = n_grpo_steps * epochs_per_rollout_batch * batches_per_rollout
    total_steps = math.ceil(total_micro_steps / grad_accum)
    warmup_steps = int(total_steps * float(cfg["training"]["warmup_ratio"]))
    scheduler = get_scheduler(
        name=cfg["training"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    wandb_run = setup_wandb(cfg, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    rollout_path = output_dir / "rollouts.jsonl"
    history: List[Dict[str, Any]] = []
    global_step = 0
    start_time = time.time()
    clip_eps = float(cfg["grpo"]["clip_eps"])

    logger.info(f"Training examples available: {len(examples)}")
    logger.info(f"GRPO rollout steps: {n_grpo_steps}")
    logger.info(f"Rollout batch size: {rollout_batch_size}, group size: {group_size}")
    logger.info(f"Epochs per rollout batch: {epochs_per_rollout_batch}")
    logger.info(f"Estimated optimizer steps: {total_steps}")

    collator = GRPODataCollator(tokenizer)
    optimizer.zero_grad(set_to_none=True)
    micro_since_step = 0

    with open(metrics_path, "w", encoding="utf-8") as metrics_file, open(
        rollout_path,
        "w",
        encoding="utf-8",
    ) as rollout_file:
        for grpo_step in range(1, n_grpo_steps + 1):
            rollouts, rollout_summary = generate_grpo_rollouts(policy_model, tokenizer, cfg, logger, examples)
            for rollout in rollouts:
                rollout_file.write(json.dumps(rollout.__dict__, ensure_ascii=False) + "\n")
            rollout_file.flush()

            dataset = MathGRPODataset(
                rollouts=rollouts,
                tokenizer=tokenizer,
                max_length=int(cfg["dataset"]["max_length"]),
                add_eos_token=cfg["dataset"]["add_eos_token"],
            )
            if len(dataset) == 0:
                logger.warning("GRPO step %d produced no trainable rollouts after tokenization.", grpo_step)
                continue

            attach_old_token_logps(policy_model, dataset, tokenizer, cfg, device)
            dataloader = DataLoader(
                dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=int(cfg["training"]["num_workers"]),
                pin_memory=cfg["training"]["pin_memory"],
                collate_fn=collator,
            )

            for rollout_epoch in range(1, epochs_per_rollout_batch + 1):
                running = {
                    "loss": 0.0,
                    "approx_kl": 0.0,
                    "clip_frac": 0.0,
                    "response_tokens": 0.0,
                    "ratio_mean": 0.0,
                    "reward_mean": 0.0,
                    "advantage_abs_mean": 0.0,
                    "micro_steps": 0,
                }
                progress = tqdm(
                    dataloader,
                    desc=f"GRPO rollout {grpo_step}/{n_grpo_steps} epoch {rollout_epoch}/{epochs_per_rollout_batch}",
                )

                for batch in progress:
                    tensor_batch = {
                        key: value.to(device)
                        for key, value in batch.items()
                        if isinstance(value, torch.Tensor)
                    }
                    loss, loss_info = compute_grpo_loss(policy_model, tensor_batch, clip_eps)
                    (loss / grad_accum).backward()
                    micro_since_step += 1

                    for key in running:
                        if key == "micro_steps":
                            continue
                        running[key] += loss_info[key]
                    running["micro_steps"] += 1

                    if micro_since_step % grad_accum != 0:
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
                        "grpo_step": grpo_step,
                        "rollout_epoch": rollout_epoch,
                        "global_step": global_step,
                        "loss": running["loss"] / denom,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "approx_kl": running["approx_kl"] / denom,
                        "clip_frac": running["clip_frac"] / denom,
                        "response_tokens": running["response_tokens"] / denom,
                        "ratio_mean": running["ratio_mean"] / denom,
                        "reward_mean": running["reward_mean"] / denom,
                        "advantage_abs_mean": running["advantage_abs_mean"] / denom,
                        "rollout_reward_mean": rollout_summary["reward_mean"],
                        "rollout_reward_std": rollout_summary["reward_std"],
                        "pass_at_group": rollout_summary["pass_at_group"],
                        "active_group_rate": rollout_summary["active_group_rate"],
                        "grad_norm": float(grad_norm.detach().cpu()),
                    }
                    history.append(record)
                    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_file.flush()

                    if wandb_run is not None:
                        wandb_run.log(
                            {f"train/{k}": v for k, v in record.items() if k != "global_step"},
                            step=global_step,
                        )

                    if global_step % int(cfg["training"]["log_every"]) == 0:
                        logger.info(
                            "step=%d grpo_step=%d loss=%.6f reward=%.4f pass@G=%.4f kl=%.5f clip=%.4f grad_norm=%.4f",
                            global_step,
                            grpo_step,
                            record["loss"],
                            record["rollout_reward_mean"],
                            record["pass_at_group"],
                            record["approx_kl"],
                            record["clip_frac"],
                            record["grad_norm"],
                        )

                    progress.set_postfix(
                        step=global_step,
                        loss=f"{record['loss']:.4f}",
                        reward=f"{record['rollout_reward_mean']:.3f}",
                        kl=f"{record['approx_kl']:.4f}",
                    )
                    running = {
                        "loss": 0.0,
                        "approx_kl": 0.0,
                        "clip_frac": 0.0,
                        "response_tokens": 0.0,
                        "ratio_mean": 0.0,
                        "reward_mean": 0.0,
                        "advantage_abs_mean": 0.0,
                        "micro_steps": 0,
                    }

        if micro_since_step % grad_accum != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                float(cfg["training"]["max_grad_norm"]),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            logger.info("Final partial optimizer step=%d grad_norm=%.4f", global_step, float(grad_norm.detach().cpu()))

    plot_grpo_training_curves(history, output_dir, cfg)
    policy_model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    summary = {
        "task": cfg["task"]["name"],
        "stage": cfg["task"]["stage"],
        "config_path": config_path,
        "model_path": str(model_path),
        "num_source_examples": len(examples),
        "n_grpo_steps": n_grpo_steps,
        "rollout_batch_size": rollout_batch_size,
        "group_size": group_size,
        "global_steps": global_step,
        "metrics_path": str(metrics_path),
        "rollout_path": str(rollout_path),
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
    logger.info("GRPO finished")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
