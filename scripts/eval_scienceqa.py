import argparse
import gc
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.peft import build_scienceqa_prompt, extract_choice_answer


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_scienceqa")
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


def resolve_project_path(path: str) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def resolve_output_dir(cfg: Dict[str, Any]) -> Tuple[Path, Path, str, str]:
    stage = cfg["task"]["stage"]
    model_tag = cfg["paths"]["model_tag"]
    eval_root = resolve_project_path(cfg["paths"]["output_dir"])
    output_dir = eval_root / model_tag
    return eval_root, output_dir, stage, model_tag


def load_eval_examples(cfg: Dict[str, Any], logger) -> List[Dict[str, Any]]:
    dataset_cfg = cfg["dataset"]
    dataset = load_from_disk(str(resolve_project_path(cfg["paths"]["dataset_path"])))
    split_ds = dataset[dataset_cfg["split"]]

    start = int(dataset_cfg["start_index"])
    limit = dataset_cfg["limit"]
    end = len(split_ds) if limit is None else min(len(split_ds), start + int(limit))

    examples = []
    skipped = 0
    for idx in range(start, end):
        item = split_ds[idx]
        answer = str(item.get("answer_letter") or "").strip().upper()
        if answer not in "ABCDE":
            skipped += 1
            continue
        examples.append(
            {
                "index": int(item.get("source_index", idx)),
                "prompt": build_scienceqa_prompt(item, cfg),
                "question": item["question"],
                "choices": item["choices"],
                "ground_truth": answer,
                "answer_text": item.get("answer_text"),
                "metadata": {field: item.get(field) for field in dataset_cfg.get("metadata_fields", [])},
            }
        )

    logger.info(f"Loaded ScienceQA eval examples: {len(examples)} from raw range [{start}, {end})")
    logger.info(f"Skipped ScienceQA eval examples: {skipped}")
    if not examples:
        raise ValueError("No ScienceQA eval examples loaded.")
    return examples


def make_group_stats():
    return defaultdict(lambda: {"total": 0, "correct": 0, "format_correct": 0})


def update_group_stats(group_stats, key: Optional[Any], correct: bool, format_correct: bool):
    key = str(key) if key is not None and key != "" else "unknown"
    stat = group_stats[key]
    stat["total"] += 1
    stat["correct"] += int(correct)
    stat["format_correct"] += int(format_correct)


def finalize_group_stats(group_stats) -> Dict[str, Dict[str, Any]]:
    result = {}
    for key, stat in sorted(group_stats.items(), key=lambda x: x[0]):
        total = stat["total"]
        result[key] = {
            "total": total,
            "correct": stat["correct"],
            "accuracy": stat["correct"] / total if total else 0.0,
            "format_accuracy": stat["format_correct"] / total if total else 0.0,
        }
    return result


def build_sampling_params(cfg: Dict[str, Any], seed: int) -> SamplingParams:
    sampling_cfg = cfg["sampling"]
    return SamplingParams(
        temperature=float(sampling_cfg["temperature"]),
        top_p=float(sampling_cfg["top_p"]),
        max_tokens=int(sampling_cfg["max_tokens"]),
        stop=sampling_cfg["stop"],
        include_stop_str_in_output=bool(sampling_cfg["include_stop_str_in_output"]),
        seed=int(sampling_cfg.get("seed", seed)),
    )


def resolve_eval_models(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    if "eval_models" in cfg:
        return [
            {
                "name": str(model_cfg["name"]),
                "path": str(model_cfg["path"]),
            }
            for model_cfg in cfg["eval_models"]
        ]

    return [
        {
            "name": str(cfg["paths"]["model_tag"]),
            "path": str(cfg["paths"]["model_path"]),
        }
    ]


def evaluate_one_model(
    model_name: str,
    model_path: str,
    cfg: Dict[str, Any],
    examples: List[Dict[str, Any]],
    output_dir: Path,
    logger,
    sampling_params: SamplingParams,
) -> Dict[str, Any]:
    prompts = [ex["prompt"] for ex in examples]
    generation_cfg = cfg["generation"]
    resolved_model_path = resolve_project_path(model_path)
    logger.info("Evaluating model '%s' from: %s", model_name, resolved_model_path)

    llm = LLM(
        model=str(resolved_model_path),
        tensor_parallel_size=int(generation_cfg["tensor_parallel_size"]),
        dtype=generation_cfg["dtype"],
        gpu_memory_utilization=float(generation_cfg["gpu_memory_utilization"]),
        max_model_len=int(generation_cfg["max_model_len"]),
        trust_remote_code=bool(generation_cfg["trust_remote_code"]),
    )

    logger.info("Running vLLM generation for model '%s'", model_name)
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time
    logger.info("Generation finished for model '%s'. elapsed=%.2fs", model_name, elapsed)

    result_path = output_dir / f"predictions_{model_name}.jsonl"
    summary_path = output_dir / f"summary_{model_name}.json"
    metrics_by_topic_path = output_dir / f"metrics_by_topic_{model_name}.json"
    metrics_by_category_path = output_dir / f"metrics_by_category_{model_name}.json"

    total = 0
    correct = 0
    format_correct = 0
    topic_stats = make_group_stats()
    category_stats = make_group_stats()

    logging_cfg = cfg["logging"]
    save_prompt = bool(logging_cfg["save_prompt"])
    save_raw_response = bool(logging_cfg["save_raw_response"])
    log_every = int(logging_cfg["log_every"])

    with open(result_path, "w", encoding="utf-8") as fout:
        for ex, prompt, out in tqdm(zip(examples, prompts, outputs), total=len(examples), desc="Grading"):
            raw_response = out.outputs[0].text
            pred = extract_choice_answer(raw_response)
            is_format_correct = pred is not None
            is_correct = pred == ex["ground_truth"]

            total += 1
            correct += int(is_correct)
            format_correct += int(is_format_correct)

            update_group_stats(topic_stats, ex["metadata"].get("topic"), is_correct, is_format_correct)
            update_group_stats(category_stats, ex["metadata"].get("category"), is_correct, is_format_correct)

            record = {
                "model_name": model_name,
                "index": ex["index"],
                "topic": ex["metadata"].get("topic"),
                "category": ex["metadata"].get("category"),
                "skill": ex["metadata"].get("skill"),
                "question": ex["question"],
                "choices": ex["choices"],
                "ground_truth": ex["ground_truth"],
                "answer_text": ex["answer_text"],
                "prediction": pred,
                "correct": is_correct,
                "format_correct": is_format_correct,
            }
            if save_prompt:
                record["prompt"] = prompt
            if save_raw_response:
                record["raw_response"] = raw_response

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if log_every > 0 and total % log_every == 0:
                logger.info(
                    f"[{model_name}] Progress {total}/{len(examples)} | "
                    f"acc={correct / total:.4f} | "
                    f"format_acc={format_correct / total:.4f}"
                )

    metrics_by_topic = finalize_group_stats(topic_stats)
    metrics_by_category = finalize_group_stats(category_stats)

    with open(metrics_by_topic_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_topic, f, ensure_ascii=False, indent=2)
    with open(metrics_by_category_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_category, f, ensure_ascii=False, indent=2)

    summary = {
        "task": cfg["task"]["name"],
        "stage": cfg["task"]["stage"],
        "model_name": model_name,
        "model_path": model_path,
        "dataset_path": cfg["paths"]["dataset_path"],
        "split": cfg["dataset"]["split"],
        "start_index": int(cfg["dataset"]["start_index"]),
        "limit": cfg["dataset"]["limit"],
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "format_correct": format_correct,
        "format_accuracy": format_correct / total if total else 0.0,
        "generation_elapsed_seconds": elapsed,
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "metrics_by_topic_path": str(metrics_by_topic_path),
        "metrics_by_category_path": str(metrics_by_category_path),
        "log_path": str(output_dir / "eval.log"),
        "metrics_by_topic": metrics_by_topic,
        "metrics_by_category": metrics_by_category,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Final summary for model '%s':", model_name)
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def build_comparison_summary(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_name = {summary["model_name"]: summary for summary in summaries}
    comparison = {
        "models": {
            name: {
                "model_path": summary["model_path"],
                "num_examples": summary["num_examples"],
                "accuracy": summary["accuracy"],
                "format_accuracy": summary["format_accuracy"],
                "correct": summary["correct"],
                "format_correct": summary["format_correct"],
            }
            for name, summary in by_name.items()
        }
    }

    if "self_play" in by_name and "peft_lora" in by_name:
        base = by_name["self_play"]
        peft = by_name["peft_lora"]
        comparison["delta"] = {
            "accuracy": peft["accuracy"] - base["accuracy"],
            "format_accuracy": peft["format_accuracy"] - base["format_accuracy"],
            "correct": peft["correct"] - base["correct"],
        }
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval/peft.yaml",
        help="Path to ScienceQA PEFT evaluation config yaml.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["task"]["seed"])
    set_seed(seed)

    eval_root, output_dir, stage, model_tag = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "eval.log")

    logger.info("Starting ScienceQA PEFT comparison evaluation")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Config path: {args.config}")
    logger.info(f"Eval root: {eval_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Stage: {stage}")
    logger.info(f"Model tag: {model_tag}")

    examples = load_eval_examples(cfg, logger)
    sampling_params = build_sampling_params(cfg, seed)
    eval_models = resolve_eval_models(cfg)
    logger.info("Eval models: %s", json.dumps(eval_models, ensure_ascii=False))

    summaries = []
    for model_cfg in eval_models:
        summaries.append(
            evaluate_one_model(
                model_name=model_cfg["name"],
                model_path=model_cfg["path"],
                cfg=cfg,
                examples=examples,
                output_dir=output_dir,
                logger=logger,
                sampling_params=sampling_params,
            )
        )

    comparison = build_comparison_summary(summaries)
    comparison.update(
        {
            "task": cfg["task"]["name"],
            "stage": stage,
            "dataset_path": cfg["paths"]["dataset_path"],
            "split": cfg["dataset"]["split"],
            "num_examples": len(examples),
            "output_dir": str(output_dir),
            "model_summaries": summaries,
        }
    )
    comparison_path = output_dir / "summary.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    logger.info("Comparison summary:")
    logger.info(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
