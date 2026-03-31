# Align a Small LLM with GRPO for Strict JSON Generation

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Information

| Field | Value |
|---|---|
| **Project ID** | 23 |
| **Track** | Align a Small LLM with GRPO for Strict Code or JSON Generation |
| **Module** | Reinforcement Learning |

## Overview

This project applies **Group Relative Policy Optimization (GRPO)** to fine-tune
**TinyLlama-1.1B** so that it generates **syntactically valid, schema-conformant JSON**.
Instead of a neural reward model, four rule-based reward components score each
completion (format, validity, schema, reasoning), providing a dense additive signal.
The model is trained with **4-bit quantization** and **LoRA** via
[Unsloth](https://github.com/unslothai/unsloth) + vLLM for fast rollouts on a
single T4 GPU (Google Colab).

> For theoretical details, ablations, and results see **[REPORT.md](docs/REPORT.md)**.

## Repository Structure

```
src/
  datasets/          Synthetic dataset generation and prompt formatting
  models/            Model/tokenizer loading (Unsloth & HuggingFace backends)
  training/          GRPO and SFT training loops, reward functions, callbacks
  evaluation/        Baseline and post-training evaluation
  utils/             Config loader and visualization helpers
experiments/
  configs/           YAML configs for baseline, GRPO, and SFT
  logs/              Training logs and wandb offline runs
    <run_name>/
      figures/       Figures generated during/after evaluation
  checkpoints/       Saved LoRA adapters and trainer state
    <run_name>/
      figures/       Comparison figures for that experiment
notes/               Colab notebooks (full pipeline + reference implementations)
docs/                Final report and cluster guide
data/                Synthetic dataset (generated, not committed)
```

## Setup

**Prerequisites**: Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GiuseppeBellamacina/grpo-strict-generation.git
cd grpo-strict-generation

pip install uv          # if not already installed
uv sync                 # core dependencies
uv sync --extra dev     # + ruff, pytest, black
uv sync --extra fast    # + unsloth, vllm (GPU required)
```

On **Google Colab** the notebook installs Unsloth and dependencies automatically
— see [`notebooks/03_full_pipeline.ipynb`](notebooks/03_full_pipeline.ipynb).

## Usage

### 1. Generate the Synthetic Dataset

```bash
uv run python -m src.datasets.synthetic_dataset \
    --output data/synthetic \
    --num_samples 5000 \
    --test_ratio 0.2
```

### 2. Baseline Evaluation

Evaluate the off-the-shelf model (no fine-tuning):

```bash
uv run python -m src.evaluation.baseline_eval \
    --config experiments/configs/baseline.yaml
```

### 3. Training

The unified entry point auto-detects the training mode from the config
(`grpo:` section → GRPO, `sft:` section → SFT) and ensures Unsloth is
imported before torch/transformers when `use_unsloth: true`.

**GRPO Training**:

```bash
uv run python -m src.training --config experiments/configs/grpo.yaml
```

**SFT Training** (for comparison):

```bash
uv run python -m src.training --config experiments/configs/sft.yaml
```

**Resume from checkpoint**:

```bash
uv run python -m src.training --config experiments/configs/grpo.yaml --resume
```

**Evaluate checkpoints only** (no training):

```bash
uv run python -m src.training \
    --config experiments/configs/grpo.yaml \
    --eval-only experiments/checkpoints/grpo
```

### 4. Post-Training Evaluation

```bash
uv run python -m src.evaluation.evaluate --config experiments/configs/grpo.yaml
```

### 5. Sync with Cluster (Windows)

Use `sync_cluster.ps1` in the repo root to transfer files to/from the DMI UniCT GPU cluster:

```powershell
# Upload project files
.\sync_cluster.ps1 -Action upload

# Download all results after training
.\sync_cluster.ps1 -Action download

# Selectively download
.\sync_cluster.ps1 -Action download-logs          # experiments/logs/ (includes figures)
.\sync_cluster.ps1 -Action download-checkpoints   # experiments/checkpoints/
.\sync_cluster.ps1 -Action download-wandb         # wandb/ offline runs
```

See [docs/CLUSTER.md](docs/CLUSTER.md) for full cluster setup instructions.

### 5. Multi-GPU Training

Per il training su più GPU, imposta `num_gpus` nel file di configurazione YAML:

```yaml
model:
  num_gpus: 2   # usa 2 GPU
```

Quando `num_gpus > 1`, il sistema **disabilita automaticamente** `use_unsloth` e
`fast_inference` (incompatibili con DDP/multi-GPU) e stampa un avviso a terminale.

Lancia il training con **Accelerate**:

```bash
accelerate launch --num_processes 2 -m src.training \
    --config experiments/configs/grpo.yaml
```

Oppure con **torchrun**:

```bash
torchrun --nproc_per_node 2 -m src.training \
    --config experiments/configs/grpo.yaml
```

> **Nota**: Tutti i print, i log su W&B e il salvataggio dei file sono già protetti
> con `is_main_process()` — solo il processo rank 0 esegue queste operazioni.

## Reward Function

All four components are **additive** and their weights sum to 1.0.
When `thinking: false`, the reasoning weight is redistributed to the others.

| Component | Weight | Description |
|---|---|---|
| **Format** | 0.20 | Presence of a ` ```json ... ``` ` code block (0.5 for generic ` ``` `) |
| **Validity** | 0.35 | JSON parseable by `json.loads` (graded score) |
| **Schema** | 0.35 | Structural conformance to prompt constraints (keys, types, counts) |
| **Reasoning** | 0.10 | `<think>…</think>` block with real content (0 when `thinking: false`) |

## Configs

| Config | Purpose | Key Parameters |
|---|---|---|
| [`baseline.yaml`](experiments/configs/baseline.yaml) | Off-the-shelf model evaluation | `temperature: 0.7`, `max_new_tokens: 512`, `num_gpus: 1` |
| [`grpo.yaml`](experiments/configs/grpo.yaml) | GRPO fine-tuning | `num_generations: 4`, `max_completion_length: 768`, `beta: 0.04`, `num_gpus: 1` |
| [`sft.yaml`](experiments/configs/sft.yaml) | Supervised fine-tuning | `epochs: 3`, `batch_size: 4`, `lr: 2e-5`, `num_gpus: 1` |

## License

[MIT](LICENSE)
