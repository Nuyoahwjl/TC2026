from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.sft import (
    SFTDataCollator,
    masked_response_cross_entropy,
    resolve_dtype,
    resolve_project_path,
    set_seed,
    setup_logger,
    setup_wandb,
)


CHOICE_LETTERS = "ABCDE"


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))

        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)

        # Keep the wrapped layer exactly as the pretrained checkpoint loaded it.
        for param in self.base.parameters():
            param.requires_grad_(False)

        # The challenge statement initializes A as zero and B randomly. This
        # keeps the initial adapter update at zero while allowing gradients.
        nn.init.zeros_(self.lora_A.weight)
        nn.init.normal_(self.lora_B.weight, mean=0.0, std=0.02)

        self.lora_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        with torch.no_grad():
            delta = torch.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
            delta = delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
            self.base.weight.add_(delta)
        return self.base


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(False)


def _replace_lora_modules(module: nn.Module, target_modules: Tuple[str, ...], rank: int, alpha: float, dropout: float):
    replaced = []
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in target_modules:
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(child_name)
            continue
        child_replaced = _replace_lora_modules(child, target_modules, rank, alpha, dropout)
        replaced.extend(f"{child_name}.{name}" for name in child_replaced)
    return replaced


def inject_lora(model: nn.Module, target_modules: List[str], rank: int, alpha: float, dropout: float) -> List[str]:
    replaced = _replace_lora_modules(model, tuple(target_modules), rank, alpha, dropout)
    if not replaced:
        raise ValueError(f"No LoRA target modules matched: {target_modules}")
    return replaced


def merge_lora_weights(module: nn.Module):
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, child_name, child.merge())
        else:
            merge_lora_weights(child)


def trainable_parameter_summary(model: nn.Module) -> Dict[str, Any]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def save_lora_adapter(model: nn.Module, adapter_dir: Path, cfg: Dict[str, Any], replaced_modules: List[str]):
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapter_state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            adapter_state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()

    torch.save(adapter_state, adapter_dir / "adapter_model.bin")
    adapter_config = {
        "peft_type": "manual_lora",
        "target_modules": cfg["lora"]["target_modules"],
        "rank": int(cfg["lora"]["rank"]),
        "alpha": float(cfg["lora"]["alpha"]),
        "dropout": float(cfg["lora"]["dropout"]),
        "replaced_modules": replaced_modules,
    }
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, ensure_ascii=False, indent=2)


@dataclass
class ScienceQAExample:
    index: int
    prompt: str
    response: str
    question: str
    final_answer: str
    answer_text: str
    metadata: Dict[str, Any]


class ScienceQADataset(Dataset):
    def __init__(self, examples: List[ScienceQAExample], tokenizer, max_length: int, add_eos_token: bool):
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
            if not response_ids:
                continue

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
                    "answer_text": ex.answer_text,
                    "metadata": ex.metadata,
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


def format_choices(choices: List[str]) -> str:
    return "\n".join(f"{CHOICE_LETTERS[i]}. {choice}" for i, choice in enumerate(choices))


def build_scienceqa_prompt(item: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    prompt_cfg = cfg["prompt"]
    metadata_lines = []
    for field in cfg["dataset"].get("metadata_fields", []):
        value = item.get(field)
        if value is not None and str(value).strip():
            metadata_lines.append(f"{field.replace('_', ' ').title()}: {str(value).strip()}")

    hint = (item.get("hint") or "").strip()
    hint_block = f"\n\nHint:\n{hint}" if hint else ""
    query = prompt_cfg["user_template"].format(
        metadata="\n".join(metadata_lines),
        hint_block=hint_block,
        question=item["question"],
        choices=format_choices(item["choices"]),
    )
    return f"{prompt_cfg['system_template'].strip()}\n\n{query.strip()}"


def build_scienceqa_response(item: Dict[str, Any]) -> str:
    letter = item["answer_letter"].strip()
    answer_text = item["answer_text"].strip()
    solution = item["solution"].strip()
    conclusion = f"So the final answer is {letter}: {answer_text}."
    return f"{solution}\n\n{conclusion}</think> <answer>{letter}</answer>"


def load_scienceqa_examples(cfg: Dict[str, Any], logger) -> List[ScienceQAExample]:
    dataset_cfg = cfg["dataset"]
    dataset = load_from_disk(str(resolve_project_path(cfg["paths"]["dataset_path"])))
    split = dataset[dataset_cfg["split"]]

    start = int(dataset_cfg["start_index"])
    limit = dataset_cfg["limit"]
    end = len(split) if limit is None else min(len(split), start + int(limit))

    examples = []
    skipped = 0
    for idx in range(start, end):
        item = split[idx]
        choices = item.get("choices") or []
        answer = int(item["answer"])
        if answer < 0 or answer >= len(choices) or answer >= len(CHOICE_LETTERS):
            skipped += 1
            continue

        prompt = build_scienceqa_prompt(item, cfg)
        response = build_scienceqa_response(item)
        examples.append(
            ScienceQAExample(
                index=int(item.get("source_index", idx)),
                prompt=prompt,
                response=response,
                question=item["question"],
                final_answer=item["answer_letter"],
                answer_text=item["answer_text"],
                metadata={field: item.get(field) for field in dataset_cfg.get("metadata_fields", [])},
            )
        )

    logger.info(f"Loaded ScienceQA PEFT examples: {len(examples)} from raw range [{start}, {end})")
    logger.info(f"Skipped ScienceQA examples: {skipped}")
    if not examples:
        raise ValueError("No ScienceQA PEFT examples loaded.")
    return examples


def plot_peft_training_curves(history: List[Dict[str, Any]], output_dir: Path, cfg: Dict[str, Any]):
    if not cfg["plots"]["enabled"] or not history:
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
    fig.suptitle("PEFT LoRA Training Dynamics", fontsize=16, fontweight="bold")
    for ax, (key, title, color) in zip(axes.ravel(), panels):
        values = [row[key] for row in history]
        ax.plot(steps, values, color=color, linewidth=2.0)
        ax.scatter(steps[-1:], values[-1:], color=color, s=32, zorder=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Optimizer Step")
        ax.grid(True, alpha=0.28)

    fig.savefig(output_dir / "training_curves.png", dpi=int(cfg["plots"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def train_peft(cfg: Dict[str, Any], config_path: str):
    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    adapter_dir = resolve_project_path(cfg["paths"]["adapter_dir"])
    final_model_dir = resolve_project_path(cfg["paths"]["final_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "train.log")
    logger.info("Starting PEFT LoRA training")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Adapter dir: {adapter_dir}")
    logger.info(f"Final model dir: {final_model_dir}")

    set_seed(int(cfg["task"]["seed"]))

    model_path = resolve_project_path(cfg["paths"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=cfg["model"]["trust_remote_code"])
    tokenizer.pad_token = tokenizer.eos_token

    examples = load_scienceqa_examples(cfg, logger)
    dataset = ScienceQADataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=int(cfg["dataset"]["max_length"]),
        add_eos_token=cfg["dataset"]["add_eos_token"],
    )
    if len(dataset) == 0:
        raise ValueError("All ScienceQA PEFT examples were filtered by max_length.")

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
    freeze_model(model)
    replaced_modules = inject_lora(
        model,
        target_modules=list(cfg["lora"]["target_modules"]),
        rank=int(cfg["lora"]["rank"]),
        alpha=float(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
    )

    if cfg["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    device = torch.device("cuda")
    model.to(device)
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    param_summary = trainable_parameter_summary(model)
    logger.info("LoRA modules inserted: %d", len(replaced_modules))
    logger.info("Parameter summary: %s", json.dumps(param_summary, ensure_ascii=False))

    optimizer = torch.optim.AdamW(
        trainable_params,
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
            progress = tqdm(dataloader, desc=f"PEFT epoch {epoch + 1}/{num_epochs}")

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
                    trainable_params,
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

    plot_peft_training_curves(history, output_dir, cfg)
    save_lora_adapter(model, adapter_dir, cfg, replaced_modules)

    if bool(cfg["lora"]["merge_and_save"]):
        merge_lora_weights(model)
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
        "parameter_summary": param_summary,
        "num_lora_modules": len(replaced_modules),
        "metrics_path": str(metrics_path),
        "plot_path": str(output_dir / "training_curves.png"),
        "adapter_dir": str(adapter_dir),
        "final_model_dir": str(final_model_dir),
        "elapsed_seconds": time.time() - start_time,
        "last_metrics": history[-1] if history else None,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    logger.info("Saved LoRA adapter: %s", adapter_dir)
    logger.info("Saved final model: %s", final_model_dir)
    logger.info("PEFT LoRA training finished")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def extract_choice_answer(response: str) -> Optional[str]:
    answer_match = re.search(r"<answer>\s*([A-E])\s*</answer>", response, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Fallback for generations that stop before closing the answer tag.
    open_match = re.search(r"<answer>\s*([A-E])\b", response, flags=re.IGNORECASE)
    if open_match:
        return open_match.group(1).upper()

    candidates = re.findall(r"\b([A-E])\b", response.upper())
    return candidates[-1] if candidates else None
