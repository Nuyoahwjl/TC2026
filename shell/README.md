# Shell helpers

Linux shell scripts for project-wide training, sampling, evaluation, and GPU monitoring.

Run from the project root:

```bash
bash shell/watch_gpu.sh
bash shell/run_job.sh eval-baseline
bash shell/run_job.sh train-sft
```

Useful environment overrides:

```bash
CUDA_VISIBLE_DEVICES=1 bash shell/run_job.sh eval-sft
LOG_DIR=logs CUDA_VISIBLE_DEVICES=0 bash shell/run_job.sh train-grpo
```

Supported jobs:

- `eval-baseline`
- `train-sft`
- `eval-sft`
- `sample-rsft`
- `train-rsft`
- `eval-rsft`
- `sample-dpo`
- `train-dpo`
- `eval-dpo`
- `train-grpo`
- `eval-grpo`
- `train-self-play`
- `eval-self-play`
- `train-peft`
- `eval-peft`

Each job runs with `nohup` in the background and writes a `.pid` file next to its log.
