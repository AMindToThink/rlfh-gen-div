# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code for the paper "Understanding the Effects of RLHF on LLM Generalisation and Diversity" ([arxiv:2310.06452](https://arxiv.org/abs/2310.06452)). Implements a multi-stage RLHF pipeline: SFT → Reward Model → RL (PPO) → Evaluation/Diversity analysis.

## Setup & Dependencies

- **Package system**: `setup.py` (setuptools), package name `rlvsil`
- **Dependencies**: `pip install -r requirements.txt` in a virtualenv (Python 3.10)
- **Venv**: `.venv/` with Python 3.10. No uv/pyproject.toml — use pip directly for this repo.
- **GPU**: Training scripts require CUDA. Evaluation/diversity scripts can run on CPU.
- **DeepSpeed**: Optional, needed for multi-GPU SFT/RM training (`pip install deepspeed`)

## Running Scripts

All scripts are designed to run **from inside the `rlvsil/` folder**:

```bash
cd rlvsil/
python train_and_eval.py ...          # SFT training
python train_summarisation_reward_model.py ...  # Reward model
accelerate launch experiment_accel.py ...       # RL (PPO) training
python sample_best_of_N.py ...        # Best-of-N sampling
python calculate_diversity.py ...     # Diversity metrics
```

No formal test suite exists. Pre-commit hooks (`.pre-commit-config.yaml` in `rlvsil/`) check for trailing whitespace, large files, and leftover `pdb` imports.

## Architecture

### Pipeline Flow

SFT model → Reward model (trained on preference pairs) → PPO RL training (policy + reward + reference models) → Evaluation (best-of-N sampling, diversity metrics)

### Key Modules (`rlvsil/`)

| Module | Role |
|--------|------|
| `lab/` | Training orchestration: `Experiment` class, `Args` dataclass, SFT/RM trainers |
| `algos/ppo.py` | `AutoRegressivePPOTrainer` — PPO with adaptive KL penalty, GAE, clipped objectives |
| `models/` | Model loading (LLaMA, OPT, GPT-2), device parallelism, value heads |
| `dataset/` | Summarization data loading, feedback dataloaders, batch collation |
| `evaluation/` | `RewardFunction` hierarchy (BLEU, ROUGE, sentiment, summarization RM), factory via `make_reward_function()` |
| `diversity/` | Diversity metrics (Distinct N-grams, EAD, SentBERT similarity, NLI-based) with `DiversityMetric` base class |
| `core/` | RL utilities: stats tracking, V-trace, tensor ops |

### Config Systems

- **RL training** (`experiment_accel.py`): Hydra config (`conf/config_accel.yaml`, ~170 params)
- **SFT/RM training**: HuggingFace `TrainingArguments` + custom `Args` dataclass (`lab/args.py`)
- **Diversity CLI** (`calculate_diversity.py`): Click

### External Integrations

- **Weights & Biases**: Deeply integrated — all experiments log to wandb. Evaluation scripts scrape results from wandb runs.
- **HuggingFace Accelerate**: Multi-GPU distributed training for RL
- **DeepSpeed**: Optional parallel training for SFT/RM

## License

CC-BY-NC (main code), MIT (diversity-eval), Apache-2.0 (TRL portions).
