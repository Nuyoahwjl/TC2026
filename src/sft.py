from __future__ import annotations

import json
import logging
import math
import os
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
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def build_prompt(question: str, cfg: Dict[str, Any]) -> str:
    system_template = cfg["prompt"]["system_template"].strip()
    user_template = cfg["prompt"]["user_template"].format(question=question).strip()
    return f"{system_template}\n\n{user_template}"


def extract_boxed_answer(solution: str) -> Optional[str]:
    start = max(solution.rfind("\\boxed"), solution.rfind("\\fbox"))
    if start < 0:
        return None

    open_brace = solution.find("{", start)
    depth = 0
    for idx in range(open_brace, len(solution)):
        if solution[idx] == "{":
            depth += 1
        elif solution[idx] == "}":
            depth -= 1
            if depth == 0:
                return solution[open_brace + 1 : idx].strip()
    return None


def build_response(solution: str, final_answer: str) -> str:
    return f"{solution.strip()}</think> <answer>{final_answer.strip()}</answer>"


def load_env_file(path: Path):
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


@dataclass
class SFTExample:
    index: int
    prompt: str
    response: str
    question: str
    final_answer: str
    metadata: Dict[str, Any]


def load_sft_examples(cfg: Dict[str, Any], logger: logging.Logger) -> List[SFTExample]:
    dataset_cfg = cfg["dataset"]
    dataset = load_from_disk(str(resolve_project_path(cfg["paths"]["dataset_path"])))
    split = dataset[dataset_cfg["split"]]

    start = int(dataset_cfg["start_index"])
    limit = dataset_cfg["limit"]
    end = len(split) if limit is None else min(len(split), start + int(limit))

    examples: List[SFTExample] = []
    skipped_no_boxed = 0
    for idx in range(start, end):
        item = split[idx]
        question = item[dataset_cfg["question_field"]]
        solution = item[dataset_cfg["solution_field"]]
        final_answer = extract_boxed_answer(solution)

        if final_answer is None:
            skipped_no_boxed += 1
            if dataset_cfg["skip_without_boxed_answer"]:
                continue
            final_answer = solution.strip()

        examples.append(
            SFTExample(
                index=idx,
                prompt=build_prompt(question, cfg),
                response=build_response(solution, final_answer),
                question=question,
                final_answer=final_answer,
                metadata={field: item.get(field) for field in dataset_cfg["metadata_fields"]},
            )
        )

    logger.info(f"Loaded SFT examples: {len(examples)} from raw range [{start}, {end})")
    logger.info(f"Skipped examples without boxed answer: {skipped_no_boxed}")
    return examples


class MathSFTDataset(Dataset):
    def __init__(self, examples: List[SFTExample], tokenizer, max_length: int, add_eos_token: bool):
        self.features = []
        for ex in examples:
            prompt_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(ex.response, add_special_tokens=False)["input_ids"]
            if add_eos_token:
                response_ids.append(tokenizer.eos_token_id)

            max_response_len = max_length - len(prompt_ids)
            if max_response_len <= 0:
                continue
            response_ids = response_ids[:max_response_len]

            input_ids = prompt_ids + response_ids
            response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            self.features.append(
                {
                    "input_ids": input_ids,
                    "response_mask": response_mask,
                    "index": ex.index,
                    "question": ex.question,
                    "response": ex.response,
                    "final_answer": ex.final_answer,
                    "metadata": ex.metadata,
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


class SFTDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        input_ids, attention_mask, response_mask = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
            response_mask.append(f["response_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.bool),
            "indices": [f["index"] for f in features],
            "questions": [f["question"] for f in features],
            "responses": [f["response"] for f in features],
            "final_answers": [f["final_answer"] for f in features],
        }


def masked_response_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor):
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:].float()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_loss = -target_log_probs
    denom = shift_mask.sum().clamp_min(1.0)
    loss = (token_loss * shift_mask).sum() / denom

    with torch.no_grad():
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy = (entropy * shift_mask).sum() / denom

    return loss, {
        "response_tokens": float(denom.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }


def resolve_dtype(dtype_name: str):
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[dtype_name]


def setup_wandb(cfg: Dict[str, Any], output_dir: Path):
    wandb_cfg = cfg["wandb"]
    if not wandb_cfg["enabled"]:
        return None

    load_env_file(resolve_project_path(wandb_cfg["env_file"]))
    import wandb

    return wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ.get("WANDB_ENTITY"),
        name=wandb_cfg["run_name"],
        tags=wandb_cfg["tags"],
        config=cfg,
        dir=str(output_dir),
    )


def plot_training_curves(history: List[Dict[str, Any]], output_dir: Path, cfg: Dict[str, Any]):
    if not cfg["plots"]["enabled"]:
        return

    plt.style.use(cfg["plots"]["style"])
    steps = [row["global_step"] for row in history]
    panels = [
        ("loss", "Loss", "#31688e"),
        ("learning_rate", "Learning Rate", "#35b779"),
        ("entropy", "Response Entropy", "#f89540"),
        ("grad_norm", "Gradient Norm", "#cc4778"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("SFT Training Dynamics", fontsize=16, fontweight="bold")
    for ax, (key, title, color) in zip(axes.ravel(), panels):
        values = [row[key] for row in history]
        ax.plot(steps, values, color=color, linewidth=2.0)
        ax.scatter(steps[-1:], values[-1:], color=color, s=32, zorder=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Optimizer Step")
        ax.grid(True, alpha=0.28)

    fig.savefig(output_dir / "training_curves.png", dpi=int(cfg["plots"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def train_sft(cfg: Dict[str, Any], config_path: str):
    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    final_model_dir = resolve_project_path(cfg["paths"]["final_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "train.log")
    logger.info("Starting SFT training")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Final model dir: {final_model_dir}")

    set_seed(int(cfg["task"]["seed"]))

    model_path = resolve_project_path(cfg["paths"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=cfg["model"]["trust_remote_code"])
    tokenizer.pad_token = tokenizer.eos_token

    examples = load_sft_examples(cfg, logger)
    dataset = MathSFTDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=int(cfg["dataset"]["max_length"]),
        add_eos_token=cfg["dataset"]["add_eos_token"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["train_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=cfg["training"]["pin_memory"],
        collate_fn=SFTDataCollator(tokenizer),
    )

    model_kwargs = {
        "torch_dtype": resolve_dtype(cfg["model"]["torch_dtype"]),
        "trust_remote_code": cfg["model"]["trust_remote_code"],
    }
    if cfg["model"]["attn_implementation"]:
        model_kwargs["attn_implementation"] = cfg["model"]["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    model.config.use_cache = False
    if cfg["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    device = torch.device("cuda")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    wandb_run = setup_wandb(cfg, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    history: List[Dict[str, Any]] = []
    global_step = 0
    start_time = time.time()

    logger.info(f"Training examples: {len(dataset)}")
    logger.info(f"Epochs: {num_epochs}, micro-batches per epoch: {len(dataloader)}")
    logger.info(f"Gradient accumulation steps: {grad_accum}, optimizer steps: {total_steps}")

    optimizer.zero_grad(set_to_none=True)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for epoch in range(num_epochs):
            running = {"loss": 0.0, "entropy": 0.0, "response_tokens": 0.0, "micro_steps": 0}
            progress = tqdm(dataloader, desc=f"SFT epoch {epoch + 1}/{num_epochs}")

            for micro_step, batch in enumerate(progress, start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                response_mask = batch["response_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, loss_info = masked_response_cross_entropy(outputs.logits, input_ids, response_mask)
                (loss / grad_accum).backward()

                running["loss"] += float(loss.detach().cpu())
                running["entropy"] += loss_info["entropy"]
                running["response_tokens"] += loss_info["response_tokens"]
                running["micro_steps"] += 1

                if micro_step % grad_accum != 0 and micro_step != len(dataloader):
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
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
                    "entropy": running["entropy"] / denom,
                    "response_tokens": running["response_tokens"] / denom,
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
                history.append(record)
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_file.flush()

                if wandb_run is not None:
                    wandb_run.log({f"train/{k}": v for k, v in record.items() if k != "global_step"}, step=global_step)

                if global_step % int(cfg["training"]["log_every"]) == 0:
                    logger.info(
                        "step=%d epoch=%d loss=%.6f lr=%.3e entropy=%.4f response_tokens=%.1f grad_norm=%.4f",
                        global_step,
                        epoch + 1,
                        record["loss"],
                        record["learning_rate"],
                        record["entropy"],
                        record["response_tokens"],
                        record["grad_norm"],
                    )

                progress.set_postfix(step=global_step, loss=f"{record['loss']:.4f}", lr=f"{record['learning_rate']:.2e}")
                running = {"loss": 0.0, "entropy": 0.0, "response_tokens": 0.0, "micro_steps": 0}

    plot_training_curves(history, output_dir, cfg)
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    summary = {
        "task": cfg["task"]["name"],
        "stage": cfg["task"]["stage"],
        "config_path": config_path,
        "model_path": str(model_path),
        "dataset_path": str(resolve_project_path(cfg["paths"]["dataset_path"])),
        "split": cfg["dataset"]["split"],
        "num_examples": len(dataset),
        "global_steps": global_step,
        "metrics_path": str(metrics_path),
        "plot_path": str(output_dir / "training_curves.png"),
        "final_model_dir": str(final_model_dir),
        "elapsed_seconds": time.time() - start_time,
        "last_metrics": history[-1],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    logger.info("Saved final model: %s", final_model_dir)
    logger.info("SFT training finished")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
