from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, List, Optional

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.grader import extract_answer, r1_zero_reward_fn
from src.sft import (
    MathSFTDataset,
    SFTDataCollator,
    SFTExample,
    build_prompt,
    masked_response_cross_entropy,
    plot_training_curves,
    resolve_dtype,
    resolve_project_path,
    set_seed,
    setup_logger,
    setup_wandb,
)


def canonicalize_response_for_grader(response: str) -> str:
    return re.sub(r"</think>\s*<answer>", "</think> <answer>", response)


def extract_model_answer(response: str) -> Optional[str]:
    if "<answer>" not in response or "</answer>" not in response:
        return None
    return response.split("<answer>")[-1].split("</answer>")[0].strip()


def extract_ground_truth(solution: Optional[str]) -> Optional[str]:
    if solution is None:
        return None
    try:
        boxed = extract_answer(solution)
        if boxed is not None:
            return boxed
    except Exception:
        pass
    return solution


def load_generation_examples(cfg: Dict[str, Any], logger) -> List[Dict[str, Any]]:
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
        question = item.get(dataset_cfg["question_field"])
        solution = item.get(dataset_cfg["solution_field"])
        ground_truth = extract_ground_truth(solution)
        if not question or ground_truth is None or ground_truth == "":
            skipped += 1
            continue

        examples.append(
            {
                "index": idx,
                "question": question,
                "solution": solution,
                "ground_truth": ground_truth,
                "prompt": build_prompt(question, cfg),
                "metadata": {field: item.get(field) for field in dataset_cfg["metadata_fields"]},
            }
        )

    logger.info(f"Loaded generation examples: {len(examples)} from raw range [{start}, {end})")
    logger.info(f"Skipped generation examples: {skipped}")
    return examples


def build_sampling_params(cfg: Dict[str, Any]):
    from vllm import SamplingParams

    sampling_cfg = cfg["sampling"]
    kwargs = {
        "temperature": float(sampling_cfg["temperature"]),
        "top_p": float(sampling_cfg["top_p"]),
        "max_tokens": int(sampling_cfg["max_tokens"]),
        "stop": sampling_cfg["stop"],
        "include_stop_str_in_output": bool(sampling_cfg["include_stop_str_in_output"]),
        "n": int(sampling_cfg["n"]),
        "seed": int(sampling_cfg.get("seed", cfg["task"]["seed"])),
    }
    if sampling_cfg.get("min_tokens") is not None:
        kwargs["min_tokens"] = int(sampling_cfg["min_tokens"])
    return SamplingParams(**kwargs)


def generate_rsft_data(cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    from vllm import LLM

    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rsft_data_path = resolve_project_path(cfg["paths"]["rsft_data_path"])
    rsft_data_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg["rsft"]["skip_sampling_if_exists"] and rsft_data_path.exists():
        logger.info(f"RSFT data already exists, skip sampling: {rsft_data_path}")
        return {"rsft_data_path": str(rsft_data_path), "skipped": True}

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

    logger.info("Running RSFT rejection sampling")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - start_time

    grading_cfg = cfg["grading"]
    fast = bool(grading_cfg["fast"])
    normalize_tag_whitespace = bool(grading_cfg["normalize_tag_whitespace"])
    reward_threshold = float(cfg["rsft"]["reward_threshold"])
    max_accepted_per_prompt = int(cfg["rsft"]["max_accepted_per_prompt"])

    total_candidates = 0
    accepted = 0
    prompts_with_accept = 0

    with open(rsft_data_path, "w", encoding="utf-8") as fout:
        for ex, out in tqdm(zip(examples, outputs), total=len(examples), desc="Filtering RSFT samples"):
            accepted_for_prompt = 0
            for sample_id, candidate in enumerate(out.outputs):
                total_candidates += 1
                raw_response = candidate.text
                graded_response = (
                    canonicalize_response_for_grader(raw_response)
                    if normalize_tag_whitespace
                    else raw_response
                )
                reward_info = r1_zero_reward_fn(graded_response, ex["ground_truth"], fast=fast)
                if reward_info["reward"] < reward_threshold:
                    continue

                model_answer = extract_model_answer(graded_response)
                if model_answer is None:
                    continue

                record = {
                    "index": ex["index"],
                    "sample_id": sample_id,
                    "question": ex["question"],
                    "ground_truth": ex["ground_truth"],
                    "model_answer": model_answer,
                    "prompt": ex["prompt"],
                    "response": graded_response,
                    "reward_info": reward_info,
                    "metadata": ex["metadata"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                accepted += 1
                accepted_for_prompt += 1
                if accepted_for_prompt >= max_accepted_per_prompt:
                    break

            prompts_with_accept += int(accepted_for_prompt > 0)

    summary = {
        "rsft_data_path": str(rsft_data_path),
        "num_prompts": len(examples),
        "total_candidates": total_candidates,
        "accepted": accepted,
        "prompts_with_accept": prompts_with_accept,
        "acceptance_rate": accepted / total_candidates if total_candidates else 0.0,
        "prompt_coverage": prompts_with_accept / len(examples) if examples else 0.0,
        "generation_elapsed_seconds": elapsed,
    }
    with open(output_dir / "sampling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("RSFT sampling summary:")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def sample_rsft(cfg: Dict[str, Any], config_path: str):
    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "sample.log")
    logger.info("Starting RSFT sampling")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Output dir: {output_dir}")

    set_seed(int(cfg["task"]["seed"]))
    summary = generate_rsft_data(cfg, logger)

    logger.info("RSFT sampling finished")
    return summary


def load_rsft_examples(cfg: Dict[str, Any], logger) -> List[SFTExample]:
    rsft_data_path = resolve_project_path(cfg["paths"]["rsft_data_path"])
    examples: List[SFTExample] = []

    with open(rsft_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                SFTExample(
                    index=int(row["index"]),
                    prompt=row["prompt"],
                    response=row["response"],
                    question=row["question"],
                    final_answer=row["model_answer"],
                    metadata=row.get("metadata", {}),
                )
            )

    logger.info(f"Loaded RSFT examples: {len(examples)} from {rsft_data_path}")
    if len(examples) == 0:
        raise ValueError("No RSFT examples loaded. Run sampling first or check reward threshold.")
    return examples


def train_rsft(cfg: Dict[str, Any], config_path: str):
    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    final_model_dir = resolve_project_path(cfg["paths"]["final_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "train.log")
    logger.info("Starting RSFT")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Final model dir: {final_model_dir}")

    set_seed(int(cfg["task"]["seed"]))

    model_path = resolve_project_path(cfg["paths"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=cfg["model"]["trust_remote_code"])
    tokenizer.pad_token = tokenizer.eos_token

    examples = load_rsft_examples(cfg, logger)
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
    if len(dataset) == 0:
        raise ValueError("All RSFT examples were filtered by max_length.")

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
            progress = tqdm(dataloader, desc=f"RSFT epoch {epoch + 1}/{num_epochs}")

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
        "rsft_data_path": str(resolve_project_path(cfg["paths"]["rsft_data_path"])),
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
    logger.info("RSFT finished")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
