# Align a Small LLM with GRPO for Strict JSON Generation

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![References](https://img.shields.io/badge/References-REFERENCES.md-green)](docs/REFERENCES.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project applies **Group Relative Policy Optimization (GRPO)** to fine-tune five small LLMs (135M–2B parameters) so that they generate **syntactically valid, schema-conformant JSON**. Instead of a neural reward model, up to **seven rule-based reward components** score each completion, providing a dense, deterministic additive signal.

We extensively evaluated **four distinct training strategies** across 2 500 training steps (using 4-bit NF4 quantization and LoRA r=16 on a single NVIDIA L40S GPU):
1. **No-Think Standard:** Direct JSON generation on a mixed-difficulty dataset.
2. **No-Think Curriculum:** Direct generation with progressive difficulty scaling (3 stages).
3. **Think Standard:** Intermediate `<think>` reasoning steps required on a mixed dataset.
4. **Think Curriculum:** Intermediate `<think>` steps with progressive difficulty scaling.

> 📖 **For comprehensive theoretical details, heatmaps, error evolution, and cross-modality analyses, read the full [REPORT.md](docs/REPORT.md).**

### Models

| Model | Parameters | Architecture |
|---|---|---|
| SmolLM2-135M-Instruct | 135M | LLaMA-like |
| SmolLM2-360M-Instruct | 360M | LLaMA-like |
| Qwen2.5-0.5B-Instruct | 0.5B | Qwen2.5 |
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | LLaMA 2 |
| Gemma-2-2B-it | 2B | Gemma 2 |

### Key Results (Peak Performance)

After 2 500 GRPO training steps, all five models converged to a tight **87–99% Pass@1 band**. The addition of reasoning tokens and Curriculum Learning proved exceptionally transformative for smaller, capacity-constrained models:

| Model | Baseline | Peak Post-GRPO | Best Configuration | Max Absolute Gain |
|:---|:---:|:---:|:---|:---:|
| SmolLM2-135M | ~31% | **90.00%** | Think / Standard | **+58.67 pp** |
| SmolLM2-360M | ~78% | **96.67%** | No-Think / Standard | **+19.00 pp** |
| Qwen2.5-0.5B | ~92% | **98.00%** | Think / Curriculum | **+6.33 pp** |
| TinyLlama-1.1B | ~79% | **99.33%** | No-Think / Curriculum | **+20.00 pp** |
| Gemma-2-2B | ~97% | **97.67%** | Think & No-Think / Curriculum | **+1.33 pp** |

*Note: Baselines varied slightly between Think/No-Think system prompts. See the [full report](docs/REPORT.md) for detailed stage-by-stage breakdowns.*

## Repository Structure

[**Complete Repository Structure**](docs/REPO_STRUCTURE.md)

```text
├── 📁 cluster/                     # Slurm scripts and cluster management
├── 📁 data/                        # Synthetic dataset generation outputs
├── 📁 docs/                        # Documentation and Papers
│   ├── 📝 REPORT.md                # <--- FULL RESULTS AND ANALYSIS HERE
│   └── ...
├── 📁 experiments/
│   ├── 📁 configs/                 # YAML configurations for training & eval
│   └── 📁 logs/grpo/
│       ├── 📁 nothink/
│       │   ├── 📁 standard/        # Evaluation artifacts & figures (No-Think)
│       │   └── 📁 curriculum/
│       └── 📁 think/
│           ├── 📁 standard/        # Evaluation artifacts & figures (Think)
│           └── 📁 curriculum/
├── 📁 notebooks/                   # Jupyter notebooks for fast prototyping
├── 📁 src/
│   ├── 📁 datasets/                # Synthetic data generation and loaders
│   ├── 📁 evaluation/              # Pass@1, schema validation, figure generation
│   ├── 📁 models/                  # LoRA and quantization utilities
│   ├── 📁 training/                # GRPO Trainer and Curriculum logic
│   │   └── 🐍 rewards.py           # 7-component rule-based reward system
│   └── 📁 utils/
└── 📄 pyproject.toml               # uv dependencies
```

## Reward Function

The framework utilizes a purely rule-based approach, avoiding the overhead of a neural reward model. The total reward is an additive combination of up to seven components. When reasoning (`thinking: false`) is disabled, its weight is automatically redistributed to preserve component ratios.

| Component | Purpose |
|---|---|
| **Format** | Checks for a proper ` ```json ... ``` ` markdown code fence. |
| **Validity** | Graduated score based on JSON parseability (partial credit for late-string errors). |
| **Schema** | Structural conformance to exact constraints (keys, types, counts, nesting depth). |
| **Reasoning** | Evaluates `<think>…</think>` blocks for minimum character count and originality. |
| **Truncation** | Penalizes generations interrupted mid-token (e.g., unclosed braces/brackets). |
| **Repetition** | Penalizes degenerate loops (token looping, repeated lines, duplicate code blocks). |
| **Strictness** | Penalizes "chatty" text outside the requested JSON or Think blocks. |

## Setup

**Prerequisites**: Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GiuseppeBellamacina/grpo-strict-generation.git
cd grpo-strict-generation

pip install uv          # if not already installed
uv sync                 # core dependencies
uv sync --extra dev     # + ruff, pytest, black
```

**Cluster setup**: see [docs/QUICK_SETUP.md](docs/QUICK_SETUP.md) for step-by-step instructions or [docs/CLUSTER.md](docs/CLUSTER.md) for the full Slurm guide.

## Usage

### 1. Generate the Synthetic Dataset (optional)

The training and evaluation pipelines generate the dataset automatically from the YAML config. You only need this to inspect it independently:

```bash
uv run python -m src.datasets.synthetic_dataset \
    --output data/synthetic \
    --num_samples 5000 \
    --test_ratio 0.2
```

### 2. Baseline Evaluation

Evaluate off-the-shelf models without any fine-tuning:

```bash
uv run python -m src.evaluation --config experiments/configs/baseline.yaml
```

### 3. GRPO Training

Each model has its own config. To enable reasoning or curriculum learning, adjust the `thinking` and `curriculum` flags in the respective YAML files.

```bash
# Single model training
uv run python -m src.training --config experiments/configs/grpo_smollm2_135m.yaml

# Resume from checkpoint
uv run python -m src.training --config experiments/configs/grpo_smollm2_135m.yaml --resume
```

On the cluster, use the multi-model chain pipeline:
```bash
run-all                    # train + eval all models sequentially
run-all --models=1,2,3     # specific models only
monitor                    # live dashboard (compact)
```

### 4. Post-Training Evaluation

```bash
# Evaluate final checkpoint vs baseline
uv run python -m src.evaluation --config experiments/configs/grpo_smollm2_135m.yaml --compare

# Evaluate all curriculum stages + baseline (generates the full analysis suite)
uv run python -m src.evaluation --config experiments/configs/grpo_smollm2_135m.yaml --curriculum
```

## License

[MIT](LICENSE)
