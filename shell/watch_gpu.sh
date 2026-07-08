#!/usr/bin/env bash
set -euo pipefail

interval="${1:-1}"

watch -n "$interval" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
