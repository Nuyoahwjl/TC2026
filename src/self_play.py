from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.grader import extract_answer, r1_zero_reward_fn
from src.rsft import canonicalize_response_for_grader, load_generation_examples
from src.sft import (
    MathSFTDataset,
    SFTDataCollator,
    SFTExample,
    build_prompt,
    masked_response_cross_entropy,
    resolve_dtype,
    resolve_project_path,
    set_seed,
    setup_logger,
    setup_wandb,
)


@dataclass
class SelfPlayProblem:
    problem_id: str
    step: int
    prompt_id: int
    problem: str
    ground_truth: str
    generation_prompt: str
    generation_response: str
    seed_index: Optional[int]
    seed_level: Optional[str]
    seed_type: Optional[str]
    metadata: Dict[str, Any]


def normalize_for_duplicate(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_problem_and_answer(gen_output: str) -> Tuple[Optional[str], Optional[str]]:
    problem_match = re.search(r"Problem\s*:\s*(.+?)(?=\n?\s*Answer\s*:|$)", gen_output, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"Answer\s*:\s*(.+?)(?:\n\s*(?:Problem|Generate)\s*:|$)", gen_output, re.DOTALL | re.IGNORECASE)
    if problem_match is None or answer_match is None:
        return None, None

    problem = problem_match.group(1).strip()
    answer = answer_match.group(1).strip()
    answer = re.sub(r"^<answer>\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"\s*</answer>$", "", answer, flags=re.IGNORECASE).strip()
    if "\\boxed" in answer:
        try:
            boxed = extract_answer(answer)
            if boxed is not None:
                answer = boxed.strip()
        except Exception:
            pass
    return problem, answer


def answer_filter_reason(answer: Optional[str], cfg: Dict[str, Any]) -> Optional[str]:
    if answer is None or answer.strip() == "":
        return "missing_answer"

    answer = answer.strip()
    quality_cfg = cfg["self_play"]["quality_filter"]
    if len(answer) > int(quality_cfg["max_answer_chars"]):
        return "answer_too_long"
    if len(answer) < int(quality_cfg["min_answer_chars"]):
        return "answer_too_short"
    if len(answer.split()) > int(quality_cfg["max_answer_words"]):
        return "answer_too_many_words"
    if "\n" in answer and len([line for line in answer.splitlines() if line.strip()]) > 1:
        return "answer_multiline"

    lowered = answer.lower()
    banned = [str(x).lower() for x in quality_cfg.get("answer_banned_substrings", [])]
    if any(piece in lowered for piece in banned):
        return "answer_contains_explanation"
    if "problem:" in lowered or "answer:" in lowered:
        return "answer_contains_labels"
    return None


def problem_filter_reason(
    problem: Optional[str],
    answer: Optional[str],
    cfg: Dict[str, Any],
    seen_problem_norms: set[str],
) -> Optional[str]:
    quality_cfg = cfg["self_play"]["quality_filter"]
    if problem is None or problem.strip() == "":
        return "missing_problem"

    problem = problem.strip()
    if len(problem) < int(quality_cfg["min_problem_chars"]):
        return "problem_too_short"
    if len(problem) > int(quality_cfg["max_problem_chars"]):
        return "problem_too_long"
    if "answer:" in problem.lower():
        return "problem_contains_answer_label"

    norm = normalize_for_duplicate(problem)
    if norm in seen_problem_norms:
        return "duplicate_problem"

    answer_reason = answer_filter_reason(answer, cfg)
    if answer_reason is not None:
        return answer_reason
    return None


def get_curriculum_stage(cfg: Dict[str, Any], step: int) -> Dict[str, Any]:
    base = dict(cfg["self_play"])
    for stage in cfg["self_play"].get("curriculum", []):
        until_step = stage.get("until_step")
        if until_step is None or step <= int(until_step):
            merged = dict(base)
            merged.update(stage)
            return merged
    return base


def filter_seed_examples(seed_examples: List[Dict[str, Any]], stage_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    seed_levels = stage_cfg.get("seed_levels")
    if not seed_levels:
        return seed_examples

    allowed = {str(level) for level in seed_levels}
    return [ex for ex in seed_examples if str(ex.get("metadata", {}).get("level")) in allowed]


def build_problem_generation_prompt(cfg: Dict[str, Any], seed_example: Optional[Dict[str, Any]]) -> str:
    if seed_example is None:
        return cfg["problem_generation"]["unseeded_template"].strip()

    template = cfg["problem_generation"]["seed_template"].strip()
    return template.format(
        seed_problem=seed_example["question"],
        seed_answer=seed_example["ground_truth"],
        seed_level=seed_example.get("metadata", {}).get("level", "unknown"),
        seed_type=seed_example.get("metadata", {}).get("type", "unknown"),
    )


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


def generate_texts_with_policy(
    policy_model,
    tokenizer,
    prompts: List[str],
    sampling_cfg: Dict[str, Any],
    generation_batch_size: int,
) -> List[str]:
    if len(prompts) == 0:
        return []

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

            temperature = float(sampling_cfg["temperature"])
            generated = policy_model.generate(
                **encoded,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1.0e-6),
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
                        sampling_cfg.get("stop") or [],
                        bool(sampling_cfg.get("include_stop_str_in_output", False)),
                    )
                )

    tokenizer.padding_side = old_padding_side
    if was_training:
        policy_model.train()
    return responses


def generate_self_play_problems(
    policy_model,
    tokenizer,
    cfg: Dict[str, Any],
    logger,
    seed_examples: List[Dict[str, Any]],
    step: int,
    seen_problem_norms: set[str],
) -> Tuple[List[SelfPlayProblem], Dict[str, Any], List[Dict[str, Any]]]:
    stage_cfg = get_curriculum_stage(cfg, step)
    target_count = int(stage_cfg["n_problems_per_step"])
    max_attempts = int(target_count * float(stage_cfg["max_generation_attempt_multiplier"]))
    generation_batch_size = int(stage_cfg.get("problem_generation_batch_size", cfg["training"]["train_batch_size"]))

    problem_sampling = dict(cfg["problem_sampling"])
    if stage_cfg.get("problem_temperature") is not None:
        problem_sampling["temperature"] = float(stage_cfg["problem_temperature"])
    if stage_cfg.get("problem_top_p") is not None:
        problem_sampling["top_p"] = float(stage_cfg["problem_top_p"])

    use_seed_problems = bool(stage_cfg.get("use_seed_problems", False))
    candidate_seed_examples = filter_seed_examples(seed_examples, stage_cfg) if use_seed_problems else []

    accepted: List[SelfPlayProblem] = []
    candidate_records: List[Dict[str, Any]] = []
    attempts = 0
    prompt_id = 0

    while len(accepted) < target_count and attempts < max_attempts:
        batch_size = min(generation_batch_size, max_attempts - attempts, target_count - len(accepted))
        gen_prompts = []
        prompt_metadata = []
        for _ in range(batch_size):
            seed_example = random.choice(candidate_seed_examples) if candidate_seed_examples else None
            gen_prompts.append(build_problem_generation_prompt(cfg, seed_example))
            prompt_metadata.append(seed_example)
        attempts += batch_size

        outputs = generate_texts_with_policy(policy_model, tokenizer, gen_prompts, problem_sampling, generation_batch_size)
        for generation_prompt, seed_example, raw_output in zip(gen_prompts, prompt_metadata, outputs):
            problem, answer = parse_problem_and_answer(raw_output)
            reason = problem_filter_reason(problem, answer, cfg, seen_problem_norms)
            accepted_flag = reason is None
            record = {
                "step": step,
                "prompt_id": prompt_id,
                "accepted": accepted_flag,
                "filter_reason": reason,
                "problem": problem,
                "ground_truth": answer,
                "generation_prompt": generation_prompt,
                "generation_response": raw_output,
                "seed_index": int(seed_example["index"]) if seed_example is not None else None,
                "seed_level": seed_example.get("metadata", {}).get("level") if seed_example is not None else None,
                "seed_type": seed_example.get("metadata", {}).get("type") if seed_example is not None else None,
            }
            candidate_records.append(record)

            if accepted_flag:
                assert problem is not None
                assert answer is not None
                seen_problem_norms.add(normalize_for_duplicate(problem))
                problem_id = f"selfplay-{step}-{prompt_id}"
                accepted.append(
                    SelfPlayProblem(
                        problem_id=problem_id,
                        step=step,
                        prompt_id=prompt_id,
                        problem=problem.strip(),
                        ground_truth=answer.strip(),
                        generation_prompt=generation_prompt,
                        generation_response=raw_output,
                        seed_index=int(seed_example["index"]) if seed_example is not None else None,
                        seed_level=str(seed_example.get("metadata", {}).get("level"))
                        if seed_example is not None
                        else None,
                        seed_type=str(seed_example.get("metadata", {}).get("type"))
                        if seed_example is not None
                        else None,
                        metadata={
                            "source": "self_play",
                            "seeded": seed_example is not None,
                            "curriculum_use_seed": use_seed_problems,
                            "curriculum_seed_levels": stage_cfg.get("seed_levels"),
                        },
                    )
                )
            prompt_id += 1

    filter_counts: Dict[str, int] = {}
    for record in candidate_records:
        reason = record["filter_reason"] or "accepted"
        filter_counts[reason] = filter_counts.get(reason, 0) + 1

    summary = {
        "step": step,
        "target_problems": target_count,
        "attempted_generations": attempts,
        "accepted_problems": len(accepted),
        "parse_or_quality_accept_rate": len(accepted) / attempts if attempts else 0.0,
        "use_seed_problems": use_seed_problems,
        "seed_levels": stage_cfg.get("seed_levels"),
        "filter_counts": filter_counts,
    }
    logger.info("Self-play problem generation summary: %s", json.dumps(summary, ensure_ascii=False))
    return accepted, summary, candidate_records


def generate_self_play_sft_examples(
    policy_model,
    tokenizer,
    cfg: Dict[str, Any],
    logger,
    problems: List[SelfPlayProblem],
) -> Tuple[List[SFTExample], Dict[str, Any], List[Dict[str, Any]]]:
    sft_cfg = cfg["self_play_sft"]
    samples_per_problem = int(sft_cfg["samples_per_problem"])
    max_accepted_per_problem = int(sft_cfg["max_accepted_per_problem"])
    reward_threshold = float(sft_cfg["reward_threshold"])
    generation_batch_size = int(sft_cfg.get("generation_batch_size", cfg["training"]["train_batch_size"]))

    solve_prompts = [build_prompt(problem.problem, cfg) for problem in problems]
    repeated_prompts = []
    for prompt in solve_prompts:
        repeated_prompts.extend([prompt] * samples_per_problem)

    responses = generate_texts_with_policy(policy_model, tokenizer, repeated_prompts, cfg["sampling"], generation_batch_size)

    grading_cfg = cfg["grading"]
    fast = bool(grading_cfg["fast"])
    normalize_tag_whitespace = bool(grading_cfg["normalize_tag_whitespace"])

    examples: List[SFTExample] = []
    group_records: List[Dict[str, Any]] = []
    cursor = 0
    total_samples = 0
    correct_samples = 0
    prompts_with_accept = 0

    for group_id, problem in enumerate(problems):
        accepted_for_problem = 0
        samples = []
        prompt = solve_prompts[group_id]

        for sample_id in range(samples_per_problem):
            raw_response = responses[cursor]
            cursor += 1
            total_samples += 1
            graded_response = canonicalize_response_for_grader(raw_response) if normalize_tag_whitespace else raw_response
            reward_info = r1_zero_reward_fn(graded_response, problem.ground_truth, fast=fast)
            reward = float(reward_info["reward"])
            correct_samples += int(reward >= reward_threshold)
            accepted = reward >= reward_threshold and accepted_for_problem < max_accepted_per_problem

            if accepted:
                examples.append(
                    SFTExample(
                        index=len(examples),
                        prompt=prompt,
                        response=graded_response,
                        question=problem.problem,
                        final_answer=problem.ground_truth,
                        metadata={
                            **problem.metadata,
                            "problem_id": problem.problem_id,
                            "seed_index": problem.seed_index,
                            "seed_level": problem.seed_level,
                            "seed_type": problem.seed_type,
                        },
                    )
                )
                accepted_for_problem += 1

            samples.append(
                {
                    "sample_id": sample_id,
                    "accepted": accepted,
                    "raw_response": raw_response,
                    "response": graded_response,
                    "reward": reward,
                    "reward_info": reward_info,
                }
            )

        prompts_with_accept += int(accepted_for_problem > 0)
        group_records.append(
            {
                "step": problem.step,
                "problem_id": problem.problem_id,
                "group_id": group_id,
                "problem": problem.problem,
                "ground_truth": problem.ground_truth,
                "prompt": prompt,
                "samples_per_problem": samples_per_problem,
                "accepted_count": accepted_for_problem,
                "has_accept": accepted_for_problem > 0,
                "samples": samples,
                "metadata": problem.metadata,
            }
        )

    summary = {
        "num_generated_problems": len(problems),
        "total_solution_samples": total_samples,
        "correct_solution_samples": correct_samples,
        "accepted_examples": len(examples),
        "prompts_with_accept": prompts_with_accept,
        "self_play_solve_accuracy": correct_samples / total_samples if total_samples else 0.0,
        "prompt_accept_rate": prompts_with_accept / len(problems) if problems else 0.0,
        "sample_accept_rate": len(examples) / total_samples if total_samples else 0.0,
    }
    logger.info("Self-play SFT sampling summary: %s", json.dumps(summary, ensure_ascii=False))
    return examples, summary, group_records


def plot_self_play_training_curves(history: List[Dict[str, Any]], output_dir: Path, cfg: Dict[str, Any]):
    if not cfg["plots"]["enabled"] or not history:
        return

    plt.style.use(cfg["plots"]["style"])
    steps = [row["global_step"] for row in history]
    panels = [
        ("loss", "Loss", "#31688e"),
        ("self_play_solve_accuracy", "Self-Play Solve Accuracy", "#35b779"),
        ("prompt_accept_rate", "Prompt Accept Rate", "#f89540"),
        ("entropy", "Response Entropy", "#cc4778"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Self-Play SFT Training Dynamics", fontsize=16, fontweight="bold")
    for ax, (key, title, color) in zip(axes.ravel(), panels):
        values = [row.get(key, 0.0) for row in history]
        ax.plot(steps, values, color=color, linewidth=2.0)
        ax.scatter(steps[-1:], values[-1:], color=color, s=32, zorder=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Optimizer Step")
        ax.grid(True, alpha=0.28)

    fig.savefig(output_dir / "training_curves.png", dpi=int(cfg["plots"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def train_self_play(cfg: Dict[str, Any], config_path: str):
    output_dir = resolve_project_path(cfg["paths"]["output_dir"])
    final_model_dir = resolve_project_path(cfg["paths"]["final_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "train.log")
    logger.info("Starting Self-Play SFT")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Final model dir: {final_model_dir}")

    set_seed(int(cfg["task"]["seed"]))

    model_path = resolve_project_path(cfg["paths"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=cfg["model"]["trust_remote_code"])
    tokenizer.pad_token = tokenizer.eos_token

    seed_examples: List[Dict[str, Any]] = []
    if bool(cfg["self_play"].get("load_seed_examples", True)):
        seed_examples = load_generation_examples(cfg, logger)

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

    n_steps = int(cfg["self_play"]["n_steps"])
    epochs_per_step = int(cfg["self_play_sft"]["epochs_per_step"])
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
    train_batch_size = int(cfg["training"]["train_batch_size"])
    expected_examples = int(cfg["self_play"]["n_problems_per_step"]) * int(cfg["self_play_sft"]["max_accepted_per_problem"])
    batches_per_step = max(math.ceil(max(expected_examples, 1) / train_batch_size), 1)
    total_micro_steps = n_steps * epochs_per_step * batches_per_step
    total_steps = max(math.ceil(total_micro_steps / grad_accum), 1)
    warmup_steps = int(total_steps * float(cfg["training"]["warmup_ratio"]))
    scheduler = get_scheduler(
        name=cfg["training"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    wandb_run = setup_wandb(cfg, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    candidate_path = output_dir / "problem_candidates.jsonl"
    accepted_problem_path = output_dir / "generated_problems.jsonl"
    solution_path = output_dir / "self_play_solutions.jsonl"
    sft_example_path = output_dir / "accepted_sft_examples.jsonl"

    history: List[Dict[str, Any]] = []
    global_step = 0
    micro_since_step = 0
    start_time = time.time()
    seen_problem_norms: set[str] = set()

    cumulative = {
        "problem_candidates": 0,
        "accepted_problems": 0,
        "solution_samples": 0,
        "accepted_sft_examples": 0,
        "skipped_steps": 0,
    }

    logger.info(f"Seed examples available: {len(seed_examples)}")
    logger.info(f"Self-play SFT steps: {n_steps}")
    logger.info(f"Expected accepted examples per step: {expected_examples}")
    logger.info(f"Estimated optimizer steps: {total_steps}")

    optimizer.zero_grad(set_to_none=True)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file, open(
        candidate_path,
        "w",
        encoding="utf-8",
    ) as candidate_file, open(accepted_problem_path, "w", encoding="utf-8") as accepted_file, open(
        solution_path,
        "w",
        encoding="utf-8",
    ) as solution_file, open(sft_example_path, "w", encoding="utf-8") as sft_file:
        for sp_step in range(1, n_steps + 1):
            problems, generation_summary, candidate_records = generate_self_play_problems(
                model,
                tokenizer,
                cfg,
                logger,
                seed_examples,
                sp_step,
                seen_problem_norms,
            )
            for record in candidate_records:
                candidate_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            candidate_file.flush()
            for problem in problems:
                accepted_file.write(json.dumps(asdict(problem), ensure_ascii=False) + "\n")
            accepted_file.flush()

            cumulative["problem_candidates"] += len(candidate_records)
            cumulative["accepted_problems"] += len(problems)
            if len(problems) == 0:
                cumulative["skipped_steps"] += 1
                logger.warning("Self-play step %d produced no accepted problems.", sp_step)
                continue

            examples, sampling_summary, solution_records = generate_self_play_sft_examples(
                model,
                tokenizer,
                cfg,
                logger,
                problems,
            )
            for record in solution_records:
                solution_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            solution_file.flush()
            for example in examples:
                sft_file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            sft_file.flush()

            cumulative["solution_samples"] += int(sampling_summary["total_solution_samples"])
            cumulative["accepted_sft_examples"] += len(examples)

            dataset = MathSFTDataset(
                examples=examples,
                tokenizer=tokenizer,
                max_length=int(cfg["dataset"]["max_length"]),
                add_eos_token=cfg["dataset"]["add_eos_token"],
            )
            if len(dataset) == 0:
                cumulative["skipped_steps"] += 1
                logger.warning("Self-play step %d produced no trainable accepted examples.", sp_step)
                continue

            dataloader = DataLoader(
                dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=int(cfg["training"]["num_workers"]),
                pin_memory=cfg["training"]["pin_memory"],
                collate_fn=SFTDataCollator(tokenizer),
            )

            for epoch in range(1, epochs_per_step + 1):
                running = {"loss": 0.0, "entropy": 0.0, "response_tokens": 0.0, "micro_steps": 0}
                progress = tqdm(dataloader, desc=f"Self-play {sp_step}/{n_steps} epoch {epoch}/{epochs_per_step}")

                for batch in progress:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    response_mask = batch["response_mask"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss, loss_info = masked_response_cross_entropy(outputs.logits, input_ids, response_mask)
                    (loss / grad_accum).backward()
                    micro_since_step += 1

                    running["loss"] += float(loss.detach().cpu())
                    running["entropy"] += loss_info["entropy"]
                    running["response_tokens"] += loss_info["response_tokens"]
                    running["micro_steps"] += 1

                    if micro_since_step % grad_accum != 0:
                        continue

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["max_grad_norm"]))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    denom = max(running["micro_steps"], 1)
                    record = {
                        "self_play_step": sp_step,
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": running["loss"] / denom,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "entropy": running["entropy"] / denom,
                        "response_tokens": running["response_tokens"] / denom,
                        "grad_norm": float(grad_norm.detach().cpu()),
                        "generated_problem_accept_rate": generation_summary["parse_or_quality_accept_rate"],
                        "accepted_problems": generation_summary["accepted_problems"],
                        "accepted_sft_examples": sampling_summary["accepted_examples"],
                        "self_play_solve_accuracy": sampling_summary["self_play_solve_accuracy"],
                        "prompt_accept_rate": sampling_summary["prompt_accept_rate"],
                        "sample_accept_rate": sampling_summary["sample_accept_rate"],
                    }
                    history.append(record)
                    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_file.flush()

                    if wandb_run is not None:
                        wandb_run.log(
                            {f"train/{key}": value for key, value in record.items() if key != "global_step"},
                            step=global_step,
                        )

                    if global_step % int(cfg["training"]["log_every"]) == 0:
                        logger.info(
                            "step=%d self_play_step=%d loss=%.6f self_acc=%.4f accepted=%d grad_norm=%.4f",
                            global_step,
                            sp_step,
                            record["loss"],
                            record["self_play_solve_accuracy"],
                            record["accepted_sft_examples"],
                            record["grad_norm"],
                        )

                    progress.set_postfix(
                        step=global_step,
                        loss=f"{record['loss']:.4f}",
                        self_acc=f"{record['self_play_solve_accuracy']:.3f}",
                        accepted=record["accepted_sft_examples"],
                    )
                    running = {"loss": 0.0, "entropy": 0.0, "response_tokens": 0.0, "micro_steps": 0}

        if micro_since_step % grad_accum != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            logger.info("Final partial optimizer step=%d grad_norm=%.4f", global_step, float(grad_norm.detach().cpu()))

    plot_self_play_training_curves(history, output_dir, cfg)
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    summary = {
        "task": cfg["task"]["name"],
        "stage": cfg["task"]["stage"],
        "config_path": config_path,
        "model_path": str(model_path),
        "num_seed_examples": len(seed_examples),
        "n_self_play_steps": n_steps,
        "global_steps": global_step,
        "cumulative": cumulative,
        "metrics_path": str(metrics_path),
        "problem_candidates_path": str(candidate_path),
        "generated_problems_path": str(accepted_problem_path),
        "self_play_solutions_path": str(solution_path),
        "accepted_sft_examples_path": str(sft_example_path),
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
    logger.info("Self-Play SFT finished")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
