#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT}"

if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="$PROJECT_ROOT/$LOG_DIR"
fi

usage() {
  cat <<'USAGE'
Usage:
  bash shell/run_job.sh <job>

Environment:
  CUDA_VISIBLE_DEVICES  GPU id(s), default: 0
  LOG_DIR               Log/pid directory, default: project root

Jobs:
  eval-baseline
  train-sft
  eval-sft
  sample-rsft
  train-rsft
  eval-rsft
  sample-dpo
  train-dpo
  eval-dpo
  train-grpo
  eval-grpo
  train-self-play
  eval-self-play
  train-peft
  eval-peft
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 64
fi

job="$1"
python_script=""
config_file=""
log_name=""
tokenizers_parallelism=""

case "$job" in
  eval-baseline)
    python_script="scripts/eval.py"
    config_file="configs/eval/baseline.yaml"
    log_name="eval_baseline.log"
    ;;
  train-sft)
    python_script="scripts/train_sft.py"
    config_file="configs/train/sft.yaml"
    log_name="train_sft.log"
    tokenizers_parallelism="false"
    ;;
  eval-sft)
    python_script="scripts/eval.py"
    config_file="configs/eval/sft.yaml"
    log_name="eval_sft.log"
    ;;
  sample-rsft)
    python_script="scripts/sample_rsft.py"
    config_file="configs/train/rsft.yaml"
    log_name="sample_rsft.log"
    ;;
  train-rsft)
    python_script="scripts/train_rsft.py"
    config_file="configs/train/rsft.yaml"
    log_name="train_rsft.log"
    tokenizers_parallelism="false"
    ;;
  eval-rsft)
    python_script="scripts/eval.py"
    config_file="configs/eval/rsft.yaml"
    log_name="eval_rsft.log"
    ;;
  sample-dpo)
    python_script="scripts/sample_dpo.py"
    config_file="configs/train/dpo.yaml"
    log_name="sample_dpo.log"
    ;;
  train-dpo)
    python_script="scripts/train_dpo.py"
    config_file="configs/train/dpo.yaml"
    log_name="train_dpo.log"
    tokenizers_parallelism="false"
    ;;
  eval-dpo)
    python_script="scripts/eval.py"
    config_file="configs/eval/dpo.yaml"
    log_name="eval_dpo.log"
    ;;
  train-grpo)
    python_script="scripts/train_grpo.py"
    config_file="configs/train/grpo.yaml"
    log_name="train_grpo.log"
    tokenizers_parallelism="false"
    ;;
  eval-grpo)
    python_script="scripts/eval.py"
    config_file="configs/eval/grpo.yaml"
    log_name="eval_grpo.log"
    ;;
  train-self-play)
    python_script="scripts/train_self_play.py"
    config_file="configs/train/self_play.yaml"
    log_name="train_self_play.log"
    tokenizers_parallelism="false"
    ;;
  eval-self-play)
    python_script="scripts/eval.py"
    config_file="configs/eval/self_play.yaml"
    log_name="eval_self_play.log"
    ;;
  train-peft)
    python_script="scripts/train_peft.py"
    config_file="configs/train/peft.yaml"
    log_name="train_peft.log"
    tokenizers_parallelism="false"
    ;;
  eval-peft)
    python_script="scripts/eval_scienceqa.py"
    config_file="configs/eval/peft.yaml"
    log_name="eval_peft.log"
    ;;
  *)
    echo "Unknown job: $job" >&2
    echo >&2
    usage >&2
    exit 64
    ;;
esac

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
else
  export PYTHONPATH="$PROJECT_ROOT"
fi

if [[ -n "$tokenizers_parallelism" ]]; then
  export TOKENIZERS_PARALLELISM="$tokenizers_parallelism"
fi

log_file="$LOG_DIR/$log_name"
pid_file="$LOG_DIR/${job}.pid"

echo "Starting job: $job"
echo "Project root: $PROJECT_ROOT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Log file: $log_file"

nohup python -u "$python_script" --config "$config_file" > "$log_file" 2>&1 &
pid="$!"
echo "$pid" > "$pid_file"

echo "PID: $pid"
echo "PID file: $pid_file"
