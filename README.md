# TC2026

This repository contains my implementation for the [Technical Challenge 2026](./docs/Technical%20Challenge%202026.pdf) post-training task.

The project focuses on building a simple post-training pipeline for `Qwen2.5-Math-1.5B`, including:

- Baseline Evaluation
- Supervised Fine-Tuning (SFT)
- Rejection Sampling Fine-Tuning / Expert Iteration (RSFT / EI)
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization / Reinforcement Learning with Verified Rewards (GRPO / RLVR)
- Self-Play Reinforcement Learning (Self-Play RL)
- Parameter-Efficient Fine-Tuning with Manual Low-Rank Adaptation (PEFT / LoRA)

Install dependencies and prepare local model/data files with:

```bash
pip install -r requirements.txt
python scripts/download.py
```

The experiments use [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) for math tasks and a filtered subset of [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) for PEFT / LoRA.

Training and evaluation were run on A800 PCIe and RTX PRO 6000 Blackwell GPUs.

For models and results in the `outputs/` directory, click [here](./docs/link.md).
