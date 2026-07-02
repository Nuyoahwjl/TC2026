import argparse
import json
import logging
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.grader import extract_answer, r1_zero_reward_fn


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def resolve_output_dir(cfg: Dict[str, Any]) -> Tuple[Path, Path, str, str]:
    stage = cfg["task"]["stage"]
    model_tag = cfg["paths"]["model_tag"]
    eval_root = Path(cfg["paths"]["output_dir"])
    output_dir = eval_root / model_tag
    return eval_root, output_dir, stage, model_tag


def build_prompt(question: str, cfg: Dict[str, Any]) -> str:
    prompt_cfg = cfg["prompt"]
    system_template = prompt_cfg["system_template"].strip()
    user_template = prompt_cfg["user_template"]
    instruction = user_template.format(question=question).strip()
    prompt = f"{system_template}\n\n{instruction}"
    return prompt


def canonicalize_response_for_grader(response: str) -> str:
    """
    The uploaded grader checks the exact substring '</think> <answer>'.
    Some models output '</think>\\n<answer>' instead, so this normalizes
    whitespace only between these two tags.
    """
    return re.sub(r"</think>\s*<answer>", "</think> <answer>", response)


def extract_math_ground_truth(solution: Optional[str]) -> Optional[str]:
    """
    MATH solutions usually contain a final \\boxed{...}.
    Use the grader's boxed-answer extractor first.
    """
    if solution is None:
        return None
    try:
        boxed = extract_answer(solution)
        if boxed is not None:
            return boxed
    except Exception:
        pass
    return solution


def extract_ground_truth(item: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Any]:
    dataset_cfg = cfg["dataset"]
    solution_field = dataset_cfg["solution_field"]
    solution = item.get(solution_field) if solution_field else None

    return extract_math_ground_truth(solution)



def load_eval_examples(cfg: Dict[str, Any], logger) -> List[Dict[str, Any]]:
    dataset_path = Path(cfg["paths"]["dataset_path"])

    dataset_cfg = cfg["dataset"]

    split = dataset_cfg["split"]
    limit = dataset_cfg["limit"]
    start_index = int(dataset_cfg["start_index"])

    question_field = dataset_cfg["question_field"]
    solution_field = dataset_cfg["solution_field"]
    metadata_fields = dataset_cfg["metadata_fields"]

    ds = load_from_disk(str(dataset_path))

    if hasattr(ds, "keys") and split in ds:
        split_ds = ds[split]
    elif hasattr(ds, "keys") and split not in ds:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")
    else:
        split_ds = ds

    if limit is None or int(limit) <= 0:
        end_index = len(split_ds)
    else:
        end_index = min(len(split_ds), start_index + int(limit))

    examples = []
    skipped = 0

    for i in range(start_index, end_index):
        item = split_ds[i]

        question = item.get(question_field)
        solution = item.get(solution_field) if solution_field else None
        gt = extract_ground_truth(item, cfg)

        if not question or gt is None or gt == "":
            skipped += 1
            continue

        record = {
            "index": i,
            "question": question,
            "solution": solution,
            "ground_truth": gt,
        }

        for field in metadata_fields:
            record[field] = item.get(field)

        examples.append(record)

    logger.info(f"Loaded dataset from: {dataset_path}")
    logger.info(f"Split: {split}, raw range: [{start_index}, {end_index})")
    logger.info(f"Valid examples: {len(examples)}, skipped: {skipped}")

    if len(examples) == 0:
        raise ValueError("No valid examples loaded. Please check dataset config.")

    return examples


def make_group_stats():
    return defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "format_correct": 0,
            "answer_correct": 0,
        }
    )


def update_group_stats(
    group_stats,
    key: Optional[Any],
    reward_info: Dict[str, float],
):
    key = str(key) if key is not None and key != "" else "unknown"

    stat = group_stats[key]

    stat["total"] += 1
    stat["correct"] += int(reward_info["reward"] == 1.0)
    stat["format_correct"] += int(reward_info["format_reward"] == 1.0)
    stat["answer_correct"] += int(reward_info["answer_reward"] == 1.0)


def finalize_group_stats(group_stats) -> Dict[str, Dict[str, Any]]:
    result = {}

    for key, stat in sorted(group_stats.items(), key=lambda x: x[0]):
        total = stat["total"]

        result[key] = {
            "total": total,
            "correct": stat["correct"],
            "accuracy": stat["correct"] / total if total else 0.0,
            "format_accuracy": stat["format_correct"] / total if total else 0.0,
            "answer_accuracy": stat["answer_correct"] / total if total else 0.0,
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval/baseline.yaml",
        help="Path to evaluation config yaml.",
    )
    args = parser.parse_args()

    project_root = get_project_root()

    cfg = load_yaml(args.config)

    seed = int(cfg["task"]["seed"])
    set_seed(seed)

    eval_root, output_dir, stage, model_tag = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "eval.log"
    logger = setup_logger(log_file)

    logger.info("Starting evaluation")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config path: {args.config}")
    logger.info(f"Eval root: {eval_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Stage: {stage}")
    logger.info(f"Model tag: {model_tag}")

    examples = load_eval_examples(cfg, logger)
    prompts = [build_prompt(ex["question"], cfg) for ex in examples]

    logger.info(f"Loading model from: {cfg['paths']['model_path']}")

    generation_cfg = cfg["generation"]

    llm = LLM(
        model=cfg["paths"]["model_path"],
        tensor_parallel_size=int(generation_cfg["tensor_parallel_size"]),
        dtype=generation_cfg["dtype"],
        gpu_memory_utilization=float(generation_cfg["gpu_memory_utilization"]),
        max_model_len=int(generation_cfg["max_model_len"]),
        trust_remote_code=bool(generation_cfg["trust_remote_code"]),
    )

    sampling_cfg = cfg["sampling"]

    sampling_params = SamplingParams(
        temperature=float(sampling_cfg["temperature"]),
        top_p=float(sampling_cfg["top_p"]),
        max_tokens=int(sampling_cfg["max_tokens"]),
        stop=sampling_cfg["stop"],
        include_stop_str_in_output=bool(sampling_cfg["include_stop_str_in_output"]),
        seed = int(sampling_cfg.get("seed", seed)),
    )

    logger.info("Running vLLM generation")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time
    logger.info(f"Generation finished. elapsed={elapsed:.2f}s")

    result_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    metrics_by_level_path = output_dir / "metrics_by_level.json"
    metrics_by_type_path = output_dir / "metrics_by_type.json"

    total = 0
    correct = 0
    format_correct = 0
    answer_correct = 0

    total_response_chars = 0
    total_response_tokens = 0
    response_token_count_available = 0

    level_stats = make_group_stats()
    type_stats = make_group_stats()

    grading_cfg = cfg["grading"]
    fast = bool(grading_cfg["fast"])

    logging_cfg = cfg["logging"]
    save_prompt = bool(logging_cfg["save_prompt"])
    save_raw_response = bool(logging_cfg["save_raw_response"])
    save_solution = bool(logging_cfg["save_solution"])
    log_every = int(logging_cfg["log_every"])

    logger.info(f"Grading fast mode: {fast}")
    logger.info("Main metric source: raw_response")

    normalize_tag_whitespace = bool(cfg["grading"]["normalize_tag_whitespace"])

    with open(result_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in tqdm(
            zip(examples, prompts, outputs),
            total=len(examples),
            desc="Grading",
        ):
            raw_response = out.outputs[0].text

            if normalize_tag_whitespace:
                graded_response = canonicalize_response_for_grader(raw_response)
            else:
                graded_response = raw_response

            reward_info = r1_zero_reward_fn(
                graded_response,
                ex["ground_truth"],
                fast=fast,
            )

            is_correct = reward_info["reward"] == 1.0
            is_format_correct = reward_info["format_reward"] == 1.0
            is_answer_correct = reward_info["answer_reward"] == 1.0

            total += 1
            correct += int(is_correct)
            format_correct += int(is_format_correct)
            answer_correct += int(is_answer_correct)

            response_chars = len(raw_response)
            total_response_chars += response_chars

            try:
                response_token_count = len(out.outputs[0].token_ids)
            except Exception:
                response_token_count = None

            if response_token_count is not None:
                total_response_tokens += response_token_count
                response_token_count_available += 1

            update_group_stats(
                level_stats,
                ex.get("level"),
                reward_info,
            )

            update_group_stats(
                type_stats,
                ex.get("type"),
                reward_info,
            )

            running_acc = correct / total
            running_format_acc = format_correct / total
            running_answer_acc = answer_correct / total

            record = {
                "index": ex["index"],
                "level": ex.get("level"),
                "type": ex.get("type"),
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],

                "reward_info": reward_info,
                "correct": is_correct,

                "response_chars": response_chars,
                "response_tokens": response_token_count,
            }

            if save_solution:
                record["solution"] = ex.get("solution")

            if save_prompt:
                record["prompt"] = prompt

            if save_raw_response:
                record["raw_response"] = raw_response

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if log_every > 0 and total % log_every == 0:
                logger.info(
                    f"Progress {total}/{len(examples)} | "
                    f"acc={running_acc:.4f} | "
                    f"format_acc={running_format_acc:.4f} | "
                    f"answer_acc={running_answer_acc:.4f}"
                )

    accuracy = correct / total if total else 0.0
    format_accuracy = format_correct / total if total else 0.0
    answer_accuracy = answer_correct / total if total else 0.0

    avg_response_chars = total_response_chars / total if total else 0.0

    if response_token_count_available > 0:
        avg_response_tokens = total_response_tokens / response_token_count_available
    else:
        avg_response_tokens = None

    metrics_by_level = finalize_group_stats(level_stats)
    metrics_by_type = finalize_group_stats(type_stats)

    with open(metrics_by_level_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_level, f, ensure_ascii=False, indent=2)

    with open(metrics_by_type_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_type, f, ensure_ascii=False, indent=2)

    summary = {
        "task": cfg["task"]["name"],
        "stage": stage,
        "model_tag": model_tag,
        "model_path": cfg["paths"]["model_path"],

        "dataset_path": cfg["paths"]["dataset_path"],
        "split": cfg["dataset"]["split"],
        "start_index": int(cfg["dataset"]["start_index"]),
        "limit": cfg["dataset"]["limit"],
        "num_examples": total,

        "grader": cfg["grading"]["name"],
        "fast_grading": fast,
        "metric_source": "raw_response",

        "correct": correct,
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "answer_accuracy": answer_accuracy,

        "avg_response_chars": avg_response_chars,
        "avg_response_tokens": avg_response_tokens,
        "generation_elapsed_seconds": elapsed,

        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "metrics_by_level_path": str(metrics_by_level_path),
        "metrics_by_type_path": str(metrics_by_type_path),
        "log_path": str(log_file),

        "metrics_by_level": metrics_by_level,
        "metrics_by_type": metrics_by_type,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Final summary:")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))

    logger.info(f"Predictions saved to: {result_path}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Metrics by level saved to: {metrics_by_level_path}")
    logger.info(f"Metrics by type saved to: {metrics_by_type_path}")


if __name__ == "__main__":
    main()