# Align a Small LLM with GRPO for Strict JSON Generation
- **Group ID**: G23
- **Project ID**: 23

---

## 1. Introduction and Objective

Large Language Models can produce fluent natural language but frequently struggle with **structured output** — syntactically valid, schema-conformant JSON is critical for tool-use, API integrations, and agent pipelines, yet even instruction-tuned models inject conversational filler, omit closing brackets, or violate type constraints.

This project investigates whether **Group Relative Policy Optimization (GRPO)** [9], a reinforcement learning technique that forgoes a neural reward model in favour of group-level advantage normalisation, can reliably teach small LLMs (135 M – 2 B parameters) to produce strict JSON output.

Recent work has shown that rule-based rewards can effectively guide LLMs toward strict schema adherence [1] and that lightweight RL frameworks can enforce structured output without large-scale RLHF infrastructure [2]. Reward-driven RL has also proved effective for tool-use tasks [4], while curriculum-style GRPO training has been applied to code generation [3]. Building on these insights, our main hypothesis is that a combination of (i) five rule-based reward components providing dense, interpretable signal, (ii) a 3-stage curriculum that gradually increases task difficulty, and (iii) parameter-efficient 4-bit LoRA fine-tuning is sufficient to bring sub-2 B models from near-zero JSON compliance to high pass rates — closing the gap to much larger proprietary models for this narrow but practically important task.

## 2. Contribution and Added Value

We built a **multi-model GRPO fine-tuning pipeline for strict JSON generation** that goes beyond simply running the TRL `GRPOTrainer`:

1. **Five rule-based reward components** (format, validity, schema, truncation, reasoning) with additive composition and graduated partial credit — inspired by the reward architectures of [1] and [2], and designed to avoid the GRPO "zero-advantage collapse" that occurs when all completions in a group receive the same score.

2. **3-stage curriculum learning** that shifts the difficulty distribution across 2 500 training steps, enabling models to first master the code-fence format before tackling complex nested schemas — extending the curriculum ideas explored in [3] to structured-output tasks.

3. **Systematic comparison across five model families** (SmolLM2 135 M/360 M, Qwen 2.5-0.5 B, TinyLlama 1.1 B, Gemma-2 2 B), all under identical quantisation, LoRA, and curriculum settings — isolating the effect of model capacity.

4. **End-to-end reproducible infrastructure**: per-model configs, automated SLURM chain pipeline with live monitoring, stratified evaluation with difficulty breakdown, and a parametric synthetic dataset generator producing unlimited training samples.

## 3. Data Used

### Source

The dataset is **fully synthetic**, generated programmatically by `src/datasets/synthetic_dataset.py` using a library of 24 parameterised prompt templates (8 per difficulty tier, defined in `src/datasets/templates.py`). No external data is downloaded or scraped.

### Statistics

| Split | Samples | Purpose |
|---|---|---|
| Train | 1 500 per curriculum stage × 3 stages | GRPO policy gradient updates |
| Eval | 999 (333 simple / 333 medium / 333 hard) | Balanced assessment |
| Baseline test | 999 (same balanced set) | Pre-training comparison |

Each sample consists of a **system prompt**, a **user instruction**, and a **difficulty label** (simple · medium · hard). No ground-truth JSON is provided — the reward functions evaluate completions directly.

### Template taxonomy

| Difficulty | Templates | Typical task |
|---|---|---|
| **Simple** (8) | Flat key-value objects, typed arrays, entity cards, key-value mappings | "Generate a JSON object with 3 keys: name (string), age (integer), active (boolean)" |
| **Medium** (8) | Nested objects, config files, form schemas, multi-section documents | "Generate a user profile with 6 fields including a nested address and a tags array" |
| **Hard** (8) | JSON Schema draft-07, paginated API responses, deeply nested hierarchies, OpenAPI specs, workflow definitions | "Generate a JSON Schema for a REST API error with required fields, type constraints, and array validation" |

### Preprocessing

Each template's `params()` method draws random parameters (key names, counts $N \in [2, 7]$, entity types) from curated domain lists, ensuring high diversity. The difficulty distribution varies per curriculum stage (see §4.3). The random seed (42) and per-stage metadata are cached to guarantee reproducibility.

## 4. Methodology and Architecture

### 4.1 Overview

**Base architecture.** All five models are decoder-only transformer LLMs loaded from HuggingFace Hub, quantised to **4-bit NF4** with `bfloat16` compute dtype using `BitsAndBytesConfig`. A **LoRA** adapter ($r{=}16$, $\alpha{=}32$, dropout $= 0$) is attached to the seven linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), keeping the base weights frozen and training only ~0.5–2% of total parameters.

**Training algorithm.** We use the TRL `GRPOTrainer`, which implements GRPO as follows: for each prompt $x$, the current policy $\pi_\theta$ generates $G{=}8$ completions $\{y_1, \ldots, y_G\}$; each completion is scored by the rule-based reward (§4.2); rewards are normalised within the group to obtain advantages $\hat{A}_i = (r_i - \mu_G) / \sigma_G$; the policy is updated to increase the probability of above-average completions while penalising below-average ones, regularised by a KL divergence term ($\beta{=}0.04$) against the reference policy.

**Training hyperparameters** (shared across all models):

| Parameter | Value |
|---|---|
| Effective batch size | $1 \times 8$ (per-device × gradient accumulation) |
| Learning rate | $5 \times 10^{-6}$ (cosine schedule, 10% warmup) |
| Optimizer | Paged AdamW 8-bit |
| Weight decay | 0.1 |
| Max grad norm | 0.1 |
| Max prompt length | 512 tokens |
| Max completion length | 1 024 tokens |
| Max sequence length | 2 048 tokens |
| Generations per prompt ($G$) | 8 |
| KL penalty ($\beta$) | 0.04 |
| Checkpoint save interval | Every 60 steps (keep 3) |

**Hardware.** All training runs on a single NVIDIA L40S GPU (48 GB, 22 528 MB SLURM shard) via the DMI UniCT cluster, managed by Apptainer containers and SLURM.

### 4.2 Reward Functions

GRPO replaces the traditional reward model with rule-based reward functions that provide a dense, interpretable signal [1, 2, 4]. Each completion is scored independently by five components; GRPOTrainer computes the weighted sum $r = \sum_{i} w_i \cdot r_i$ and normalises across the generation group to obtain per-sample advantages.

#### Component summary

| Component | Range | Think Weight | No-Think Weight | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| `format_reward` | $[0, 1]$ | 0.10 | 0.20 | Checks for a proper ` ```json ... ``` ` code fence (1.0), a generic ` ``` ... ``` ` block (0.5), or no fence (0.0). |
| `validity_reward` | $[0, 1]$ | 0.15 | 0.25 | Graduated score based on JSON parseability. Valid JSON → 1.0. If invalid, gives partial credit (0.70 down to 0.20) proportional to how far into the string the first parse error occurs. |
| `schema_reward` | $[0, 1]$ | 0.20 | 0.25 | Average of structural checks for constraints (exact/minimum counts, required keys, types, depth, top-level type) extracted via registry metadata or regex. Returns 1.0 if JSON is valid but no constraints can be extracted. |
| `reasoning_reward` | $[-0.5, 1]$ | 0.25 | 0.0 | Evaluates the `<think>...</think>` block. Penalizes missing blocks or copying the placeholder (-0.5). Scales linearly from 10 to 80 characters of useful content towards 1.0. Disabled in *No-Think* mode. |
| `truncation_reward` | $[-1, 1]$ | 0.10 | 0.10 | Detects if the generation was cut off mid-way (e.g., hitting the token limit). Returns 1.0 if structurally complete, 0.0 (neutral) if no JSON is detected, and **-1.0** for unclosed braces/brackets, unterminated strings, or trailing commas. |
| `repetition_reward` | $[-1, 1]$ | 0.10 | 0.10 | Penalizes degenerate repetitive outputs (token loops, repeated lines, or duplicate code blocks). Returns 1.0 for normal outputs, 0.0 for moderate repetition, and **-1.0** for severe loops. |
| `strictness_reward` | $[-1, 1]$ | 0.10 | 0.10 | Penalizes extra text outside the JSON block (preambles, extra explanations). Returns 1.0 for clean outputs (only JSON), dropping gradually to **-1.0** if the extra text exceeds 50% of the total length. |

#### Design rationale

- **Additive, not gating.** Each component contributes independently to the total reward. This avoids the "zero advantage" problem: if all completions in a GRPO group scored exactly 0 (e.g., because of a hard gate on format), the advantage standard deviation would be 0 and the policy gradient would vanish — the model would never learn to produce code blocks.

- **Weight redistribution.** When `thinking: false`, the `reasoning_reward` weight is redistributed proportionally across all remaining active components, preserving their relative ratios (e.g., format 0.25 : validity 0.30 : schema 0.30 : truncation 0.15 stays in the same ratio).

### 4.3 Curriculum Learning

Training is divided into three progressive stages totalling 2 500 optimisation steps. Each stage draws fresh training samples (1 500 per stage) from the same template library but with different difficulty distributions and generation temperatures.

| Stage | Name | Steps | Simple | Medium | Hard | Temperature |
|---|---|---|---|---|---|---|
| 1 | `format_basics` | 800 | 35% | 55% | 10% | 0.8 |
| 2 | `progressive` | 800 | 15% | 35% | 50% | 0.7 |
| 3 | `full_difficulty` | 900 | 10% | 25% | 65% | 0.6 |

**Stage 1 — Format basics.** The model sees mostly simple and medium prompts with high temperature, learning to reliably wrap output in ` ```json ``` ` fences and produce parseable JSON. The high simple-prompt ratio reduces frustration early on.

**Stage 2 — Progressive.** Hard prompts become the majority. The model has learned the output format and now focuses on structural correctness (nested objects, arrays, required keys).

**Stage 3 — Full difficulty.** Two-thirds of prompts are hard (JSON Schema, OpenAPI, workflows). Temperature is lowered to 0.6, tightening the generation distribution and improving consistency on complex structures.

**Implementation details.** The same model object is reused across stages (weights updated in-place), but a fresh optimizer and LR schedule are created for each stage. All stages log to a single Weights & Biases run. Per-stage datasets are cached to disk with metadata and regenerated only if the configuration changes.

## 5. Results and Discussion

In this section, we present the evaluation results for both the *Think* and *No-Think* modalities, comparing the post-training performance against the baseline and analysing the impact of curriculum learning and reward components.

### 5.2 No-Think Modality

In this section, we analyze the results obtained by training the models to directly generate the required JSON output, without the use of intermediate reasoning tokens.

#### 5.2.1 Standard Training (2500 steps)

**Table 1**: Baseline Pass@1 — off-the-shelf models on the balanced evaluation set (300 samples: 100 simple / 100 medium / 100 hard).

| Model | Overall | Simple | Medium | Hard |
|:---|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.3900 | 0.5800 | 0.3900 | 0.2000 |
| SmolLM2-360M-Instruct | 0.7767 | 0.8500 | 0.8600 | 0.6200 |
| Qwen2.5-0.5B-Instruct | 0.9267 | 0.9500 | 0.9500 | 0.8800 |
| TinyLlama-1.1B-Chat-v1.0 | 0.7933 | 0.8900 | 0.7600 | 0.7300 |
| Gemma-2-2B-it | 0.9833 | 1.0000 | 1.0000 | 0.9500 |

The baseline evaluation reveals a steep capability gradient largely corresponding to model size and architecture. **Gemma-2-2B** acts as a near-perfect ceiling right out of the box, achieving 98.33% overall and 100% on both simple and medium prompts. **Qwen2.5-0.5B** punches significantly above its weight class, delivering an excellent 92.67% overall and outperforming the much larger **TinyLlama-1.1B** (79.33%). 

At the bottom end of the spectrum, the extreme small-scale **SmolLM2-135M** struggles to follow structured formatting instructions reliably, recording a mere 39.00% overall. Across all models, the `hard` difficulty tier consistently shows the lowest pass rates — dropping to 20.00% for the 135M model and 62.00% for the 360M model — confirming that deeply nested schemas, exact key constraints, and missing reasoning capabilities are the primary failure points for smaller instruction-tuned models prior to targeted RLHF.

**Table 2**: Post-training Pass@1 — Standard No-Think Training (2500 continuous steps on the mixed dataset).

| Model | Overall | Simple | Medium | Hard | Δ Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.8767 | 0.8700 | 0.8900 | 0.8700 | **+0.4867** |
| SmolLM2-360M-Instruct | 0.9667 | 0.9700 | 0.9700 | 0.9600 | **+0.1900** |
| Qwen2.5-0.5B-Instruct | 0.9733 | 0.9900 | 0.9500 | 0.9800 | **+0.0467** |
| TinyLlama-1.1B-Chat-v1.0 | 0.9767 | 1.0000 | 0.9600 | 0.9700 | **+0.1833** |
| Gemma-2-2B-it | 0.9667 | 1.0000 | 0.9900 | 0.9100 | **-0.0167** |

The standard training approach without curriculum yields impressive gains for the smaller and less capable models. **SmolLM2-135M** exhibits a massive leap (+48.67 pp), transforming from an unreliable baseline into a consistent generator across all difficulty levels. **SmolLM2-360M** and **TinyLlama-1.1B** also show excellent adaptability, reaching near-perfect overall scores (~97%).

However, standard training reveals a crucial limitation for already-strong models. **Gemma-2-2B**, which started with a near-perfect baseline (98.33%), actually suffers a slight regression (-1.67 pp overall), driven primarily by a noticeable drop in hard prompt performance (95% → 91%). This suggests that aggressively training a highly capable model on a mixed-difficulty distribution without careful pacing (curriculum) can lead to overfitting or catastrophic forgetting of complex edge cases.

---

**Per-model analysis**

##### SmolLM2-135M-Instruct

> **Eval directory**: [`experiments/logs/grpo/nothink/standard/smollm2-135m/eval_20260409_152243/`](../experiments/logs/grpo/nothink/standard/smollm2-135m/eval_20260409_152243/)

This model sees the most dramatic transformation. The hard-prompt pass rate skyrockets from a mere 39% to 87%, matching its performance on simple prompts. The standard training completely eliminates the initial capability gap across difficulty tiers, achieving a perfectly balanced, reliable output.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/nothink/standard/smollm2-135m/eval_20260409_152243/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/nothink/standard/smollm2-135m/eval_20260409_152243/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/standard/smollm2-135m/eval_20260409_152243/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/standard/smollm2-135m/eval_20260409_152243/figures/reward_breakdown.png) |

---

##### SmolLM2-360M-Instruct

> **Eval directory**: [`experiments/logs/grpo/nothink/standard/smollm2-360m/eval_20260409_182519/`](../experiments/logs/grpo/nothink/standard/smollm2-360m/eval_20260409_182519/)

Benefiting from its larger parameter count relative to its 135M sibling, the 360M model pushes its solid 77.67% baseline to an impressive 96.67%. The most significant growth occurs in the hard tier, jumping from 62% to 96%, proving standard GRPO is highly effective for this architecture size.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/nothink/standard/smollm2-360m/eval_20260409_182519/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/nothink/standard/smollm2-360m/eval_20260409_182519/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/standard/smollm2-360m/eval_20260409_182519/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/standard/smollm2-360m/eval_20260409_182519/figures/reward_breakdown.png) |

---

##### Qwen2.5-0.5B-Instruct

> **Eval directory**: [`experiments/logs/grpo/nothink/standard/qwen25-05b/eval_20260409_205251/`](../experiments/logs/grpo/nothink/standard/qwen25-05b/eval_20260409_205251/)

Qwen2.5-0.5B takes an already outstanding baseline (92.67%) and polishes it near perfection (97.33%). Remarkably, its performance on hard prompts jumps 10 points (88% → 98%), making it one of the most reliable models in the study for complex schemas without reasoning tokens.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/nothink/standard/qwen25-05b/eval_20260409_205251/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/nothink/standard/qwen25-05b/eval_20260409_205251/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/standard/qwen25-05b/eval_20260409_205251/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/standard/qwen25-05b/eval_20260409_205251/figures/reward_breakdown.png) |

---

##### TinyLlama-1.1B-Chat-v1.0

> **Eval directory**: [`experiments/logs/grpo/nothink/standard/tinyllama-11b/eval_20260410_023046/`](../experiments/logs/grpo/nothink/standard/tinyllama-11b/eval_20260410_023046/)

TinyLlama shows exceptional responsiveness to the standard RLHF signal. It achieves a flawless 100% on simple prompts and raises its hard-prompt pass rate from 73% to 97%. The 18.33 pp overall increase demonstrates that the base model has ample latent capacity for structured data generation that standard GRPO successfully unlocks.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/nothink/standard/tinyllama-11b/eval_20260410_023046/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/nothink/standard/tinyllama-11b/eval_20260410_023046/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/standard/tinyllama-11b/eval_20260410_023046/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/standard/tinyllama-11b/eval_20260410_023046/figures/reward_breakdown.png) |

---

##### Gemma-2-2B-it

> **Eval directory**: [`experiments/logs/grpo/nothink/standard/gemma2-2b/eval_20260410_151742/`](../experiments/logs/grpo/nothink/standard/gemma2-2b/eval_20260410_151742/)

As the largest model with the highest baseline (98.33%), Gemma-2-2B presents the only case of overall regression (-1.67 pp). While it maintains a perfect 100% on simple prompts, its hard-prompt performance drops from 95% to 91%. This indicates that throwing a chaotic mix of difficulty levels at an already highly-tuned model for 2500 steps might degrade its nuanced understanding of edge cases.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/nothink/standard/gemma2-2b/eval_20260410_151742/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/nothink/standard/gemma2-2b/eval_20260410_151742/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/standard/gemma2-2b/eval_20260410_151742/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/standard/gemma2-2b/eval_20260410_151742/figures/reward_breakdown.png) |

#### 5.2.2 Curriculum Learning (800-800-900 steps)

**Table 3**: Baseline Pass@1 — off-the-shelf models on the balanced evaluation set (300 samples: 100 simple / 100 medium / 100 hard).

| Model | Overall | Simple | Medium | Hard |
|:---|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.3300 | 0.4900 | 0.3200 | 0.1800 |
| SmolLM2-360M-Instruct | 0.7367 | 0.8200 | 0.8400 | 0.5500 |
| Qwen2.5-0.5B-Instruct | 0.8933 | 0.9800 | 0.9100 | 0.7900 |
| TinyLlama-1.1B-Chat-v1.0 | 0.7933 | 0.8900 | 0.8300 | 0.6600 |
| Gemma-2-2B-it | 0.9700 | 1.0000 | 0.9800 | 0.9300 |

The baseline evaluation reveals a steep capability gradient largely corresponding to model size and architecture. **Gemma-2-2B** acts as a near-perfect ceiling right out of the box, achieving 97.00% overall and 100% on simple prompts. **Qwen2.5-0.5B** punches significantly above its weight class, delivering an excellent 89.33% overall and outperforming the much larger **TinyLlama-1.1B** (79.33%). 

At the bottom end of the spectrum, the extreme small-scale **SmolLM2-135M** struggles to follow structured formatting instructions reliably, recording a mere 33.00% overall. Across all models, the `hard` difficulty tier consistently shows the lowest pass rates — dropping to 18.00% for the 135M model and 55.00% for the 360M model — confirming that deeply nested schemas, exact key constraints, and missing reasoning capabilities are the primary failure points for smaller instruction-tuned models prior to targeted RLHF.

**Table 4**: Post-training Pass@1 — after 2500 GRPO steps with 3-stage curriculum (No-Think).

| Model | Overall | Simple | Medium | Hard | Δ Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.8900 | 0.9000 | 0.8600 | 0.9100 | **+0.5600** |
| SmolLM2-360M-Instruct | 0.9533 | 0.9300 | 0.9500 | 0.9800 | **+0.2167** |
| Qwen2.5-0.5B-Instruct | 0.9733 | 1.0000 | 0.9500 | 0.9700 | **+0.0800** |
| TinyLlama-1.1B-Chat-v1.0 | 0.9933 | 1.0000 | 0.9900 | 0.9900 | **+0.2000** |
| Gemma-2-2B-it | 0.9767 | 1.0000 | 0.9900 | 0.9400 | **+0.0067** |

**Table 5**: Pass@1 at the end of each curriculum stage.

| Model | Baseline | Stage 1 | Stage 2 | Stage 3 |
|:---|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.3300 | 0.6267 | 0.8467 | 0.8900 |
| SmolLM2-360M-Instruct | 0.7367 | 0.7967 | 0.9133 | 0.9533 |
| Qwen2.5-0.5B-Instruct | 0.8933 | 0.9567 | 0.9700 | 0.9733 |
| TinyLlama-1.1B-Chat-v1.0 | 0.7933 | 0.9433 | 0.9833 | 0.9933 |
| Gemma-2-2B-it | 0.9700 | 0.9833 | 0.9767 | 0.9767 |

The implementation of a 3-stage curriculum learning strategy proves highly effective, outperforming the standard training approach across almost all metrics. The staged introduction of complexity allows models to solidify basic JSON formatting before tackling deep nesting and exact schema constraints. 

**TinyLlama-1.1B** is the standout performer here, achieving a near-perfect 99.33% overall pass rate (compared to 97.67% in standard training). **SmolLM2-135M** also hits a higher peak (89.00%), with an incredible 91% pass rate on hard prompts. Notably, the curriculum approach protects the highly tuned **Gemma-2-2B** from the catastrophic forgetting seen in standard training, yielding a slight positive delta (+0.0067) instead of a regression. Table 5 confirms the validity of the curriculum design: performance increases monotonically for almost all models as they progress from Stage 1 (basics) through Stage 3 (full difficulty).

---

**Per-model analysis**

##### SmolLM2-135M-Instruct

> **Eval directory**: [`experiments/logs/grpo/nothink/curriculum/smollm2-135m/eval_20260410_060558/`](../experiments/logs/grpo/nothink/curriculum/smollm2-135m/eval_20260410_060558/)

The curriculum approach unlocks the highest potential for the smallest model in our study. Gaining an astounding 56 percentage points, the 135M model climbs steadily through each stage. Crucially, its hard-prompt performance jumps from 18% to 91%, actually outperforming its own medium-prompt score and proving that gradual exposure to complexity works exceptionally well for capacity-constrained models.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/nothink/curriculum/smollm2-135m/eval_20260410_060558/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/nothink/curriculum/smollm2-135m/eval_20260410_060558/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/nothink/curriculum/smollm2-135m/eval_20260410_060558/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/curriculum/smollm2-135m/eval_20260410_060558/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/curriculum/smollm2-135m/eval_20260410_060558/figures/reward_breakdown.png) |

---

##### SmolLM2-360M-Instruct

> **Eval directory**: [`experiments/logs/grpo/nothink/curriculum/smollm2-360m/eval_20260410_094244/`](../experiments/logs/grpo/nothink/curriculum/smollm2-360m/eval_20260410_094244/)

The 360M model exhibits a textbook curriculum learning curve. After a modest gain in Stage 1 (+6 pp), it accelerates in Stage 2 (+11.6 pp) when medium schemas are introduced, and finishes strong in Stage 3. It achieves an outstanding 98% pass rate on hard prompts, completely reversing its baseline weakness (55%).

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/nothink/curriculum/smollm2-360m/eval_20260410_094244/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/nothink/curriculum/smollm2-360m/eval_20260410_094244/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/nothink/curriculum/smollm2-360m/eval_20260410_094244/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/curriculum/smollm2-360m/eval_20260410_094244/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/curriculum/smollm2-360m/eval_20260410_094244/figures/reward_breakdown.png) |

---

##### Qwen2.5-0.5B-Instruct

> **Eval directory**: [`experiments/logs/grpo/nothink/curriculum/qwen25-05b/eval_20260410_174001/`](../experiments/logs/grpo/nothink/curriculum/qwen25-05b/eval_20260410_174001/)

Qwen2.5-0.5B absorbs the Stage 1 formatting rules rapidly (jumping from 89.33% to 95.67%) and then uses Stages 2 and 3 to refine its handling of complex structures. Finishing at 97.33%, it demonstrates that a structured curriculum efficiently patches the few structural blind spots remaining in its highly capable base model.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/nothink/curriculum/qwen25-05b/eval_20260410_174001/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/nothink/curriculum/qwen25-05b/eval_20260410_174001/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/nothink/curriculum/qwen25-05b/eval_20260410_174001/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/curriculum/qwen25-05b/eval_20260410_174001/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/curriculum/qwen25-05b/eval_20260410_174001/figures/reward_breakdown.png) |

---

##### TinyLlama-1.1B-Chat-v1.0

> **Eval directory**: [`experiments/logs/grpo/nothink/curriculum/tinyllama-11b/eval_20260410_202131/`](../experiments/logs/grpo/nothink/curriculum/tinyllama-11b/eval_20260410_202131/)

This is arguably the most successful application of curriculum learning in the study. TinyLlama-1.1B scales majestically across the stages: 79.33% → 94.33% → 98.33% → 99.33%. By isolating formatting basics in Stage 1 before introducing schema complexity, the model achieves a near-flawless state, practically eliminating invalid JSON across all difficulty levels.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/nothink/curriculum/tinyllama-11b/eval_20260410_202131/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/nothink/curriculum/tinyllama-11b/eval_20260410_202131/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/nothink/curriculum/tinyllama-11b/eval_20260410_202131/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/curriculum/tinyllama-11b/eval_20260410_202131/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/curriculum/tinyllama-11b/eval_20260410_202131/figures/reward_breakdown.png) |

---

##### Gemma-2-2B-it

> **Eval directory**: [`experiments/logs/grpo/nothink/curriculum/gemma2-2b/eval_20260411_010201/`](../experiments/logs/grpo/nothink/curriculum/gemma2-2b/eval_20260411_010201/)

Unlike the standard training approach which caused a minor regression, the curriculum strategy preserves Gemma-2-2B's excellent baseline capabilities while slightly polishing them. It peaks at 98.33% in Stage 1 and stabilizes at 97.67% by Stage 3. The pacing ensures the model doesn't over-index on complex edge cases at the expense of general reliability.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/nothink/curriculum/gemma2-2b/eval_20260411_010201/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/nothink/curriculum/gemma2-2b/eval_20260411_010201/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/nothink/curriculum/gemma2-2b/eval_20260411_010201/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/nothink/curriculum/gemma2-2b/eval_20260411_010201/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/nothink/curriculum/gemma2-2b/eval_20260411_010201/figures/reward_breakdown.png) |

### 5.3 Think Modality

In this section, we evaluate the impact of intermediate reasoning: models are prompted to plan the structure inside `<think>...</think>` tags before generating the final JSON. The reward functions were recalibrated to explicitly reward the presence, length, and consistency of the reasoning block alongside the formatting constraints.

**Table 6**: Baseline Pass@1 (Think Modality) — off-the-shelf models evaluated with reasoning instructions on the balanced evaluation set.

| Model | Overall | Simple | Medium | Hard |
|:---|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.3133 | 0.4400 | 0.3300 | 0.1700 |
| SmolLM2-360M-Instruct | 0.6933 | 0.7900 | 0.8000 | 0.4900 |
| Qwen2.5-0.5B-Instruct | 0.9033 | 0.9800 | 0.9200 | 0.8100 |
| TinyLlama-1.1B-Chat-v1.0 | 0.8567 | 0.9000 | 0.8700 | 0.8000 |
| Gemma-2-2B-it | 0.9667 | 0.9900 | 0.9800 | 0.9300 |

*Baseline observations:* Introducing the `<think>` requirement alters the baseline landscape. While **TinyLlama-1.1B** naturally benefits from the instruction (jumping from 0.7933 in No-Think to 0.8567 here), the **SmolLM2** models actually perform slightly worse (e.g., 360M drops from 0.7367 to 0.6933), likely because the added prompt complexity disrupts their fragile context window before RLHF aligns them.

#### 5.3.1 Standard Training (2500 steps)

**Table 7**: Post-training Pass@1 — Standard Think Training (2500 continuous steps on the mixed dataset).

| Model | Overall | Simple | Medium | Hard | Δ Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.9000 | 0.9600 | 0.8700 | 0.8700 | **+0.5867** |
| SmolLM2-360M-Instruct | 0.9567 | 0.9600 | 0.9700 | 0.9400 | **+0.2633** |
| Qwen2.5-0.5B-Instruct | 0.9767 | 1.0000 | 0.9800 | 0.9500 | **+0.0733** |
| TinyLlama-1.1B-Chat-v1.0 | 0.9867 | 0.9900 | 0.9800 | 0.9900 | **+0.1300** |
| Gemma-2-2B-it | 0.9600 | 0.9900 | 0.9900 | 0.9000 | **-0.0067** |

Introducing the intermediate `<think>` step under standard training conditions yields remarkable benefits, particularly for smaller models. **SmolLM2-135M** records the largest absolute improvement in the entire study (+58.67 pp), reaching an impressive 90.00% overall score. By allowing the model to "plan" its JSON structure before committing to it, the model overcomes its inherent capacity bottlenecks. **TinyLlama-1.1B** also shines, hitting 98.67% overall and a staggering 99% on hard prompts.

Interestingly, the **Gemma-2-2B** model again experiences a slight regression (-0.0067) primarily driven by a drop in hard prompt performance (93% → 90%). Just as observed in the No-Think modality, throwing a highly capable model into a mixed-difficulty standard training run without curriculum pacing can slightly disrupt its pre-existing instruction-following equilibrium, even when reasoning tokens are utilized.

---

**Per-model analysis**

##### SmolLM2-135M-Instruct

> **Eval directory**: [`experiments/logs/grpo/think/standard/smollm2-135m/eval_20260411_192610/`](../experiments/logs/grpo/think/standard/smollm2-135m/eval_20260411_192610/)

The addition of reasoning tokens acts as a massive multiplier for this tiny architecture. It leaps from a 31.33% baseline to a highly reliable 90.00%. Hard prompts see a dramatic turnaround, skyrocketing from 17% to 87%. The `<think>` tags effectively give the 135M model the scratchpad it needs to track nested keys and bracket closures.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/think/standard/smollm2-135m/eval_20260411_192610/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/think/standard/smollm2-135m/eval_20260411_192610/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/standard/smollm2-135m/eval_20260411_192610/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/standard/smollm2-135m/eval_20260411_192610/figures/reward_breakdown.png) |

---

##### SmolLM2-360M-Instruct

> **Eval directory**: [`experiments/logs/grpo/think/standard/smollm2-360m/eval_20260411_223435/`](../experiments/logs/grpo/think/standard/smollm2-360m/eval_20260411_223435/)

The 360M model fully leverages the reasoning rewards, gaining 26.33 percentage points to reach 95.67%. Its performance on hard prompts nearly doubles (49% → 94%). The think modality proves highly synergistic with this architecture, allowing it to perform on par with much larger off-the-shelf models.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/think/standard/smollm2-360m/eval_20260411_223435/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/think/standard/smollm2-360m/eval_20260411_223435/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/standard/smollm2-360m/eval_20260411_223435/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/standard/smollm2-360m/eval_20260411_223435/figures/reward_breakdown.png) |

---

##### Qwen2.5-0.5B-Instruct

> **Eval directory**: [`experiments/logs/grpo/think/standard/qwen25-05b/eval_20260412_010300/`](../experiments/logs/grpo/think/standard/qwen25-05b/eval_20260412_010300/)

Starting from a strong baseline of 90.33%, Qwen2.5-0.5B uses the `<think>` modality to achieve absolute perfection on simple prompts (100%) and to push its hard prompt capabilities from 81% to 95%. It finishes at an excellent 97.67% overall.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/think/standard/qwen25-05b/eval_20260412_010300/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/think/standard/qwen25-05b/eval_20260412_010300/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/standard/qwen25-05b/eval_20260412_010300/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/standard/qwen25-05b/eval_20260412_010300/figures/reward_breakdown.png) |

---

##### TinyLlama-1.1B-Chat-v1.0

> **Eval directory**: [`experiments/logs/grpo/think/standard/tinyllama-11b/eval_20260412_034025/`](../experiments/logs/grpo/think/standard/tinyllama-11b/eval_20260412_034025/)

TinyLlama exhibits phenomenal responsiveness to the Think standard training. It reaches 98.67% overall, with an incredible 99% pass rate on hard schemas. For this model, the reasoning tokens seem to completely resolve the structural amnesia that previously caused failures in deep nesting.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/think/standard/tinyllama-11b/eval_20260412_034025/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/think/standard/tinyllama-11b/eval_20260412_034025/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/standard/tinyllama-11b/eval_20260412_034025/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/standard/tinyllama-11b/eval_20260412_034025/figures/reward_breakdown.png) |

---

##### Gemma-2-2B-it

> **Eval directory**: [`experiments/logs/grpo/think/standard/gemma2-2b/eval_20260412_090902/`](../experiments/logs/grpo/think/standard/gemma2-2b/eval_20260412_090902/)

As seen in the No-Think standard run, Gemma-2-2B is the only model to slightly regress (-0.67 pp). While it easily masters simple and medium schemas (99% on both), its hard prompt accuracy drops slightly from 93% to 90%. The mixed-difficulty distribution of standard training without curriculum gating proves suboptimal for preserving edge-case performance in an already highly tuned 2B model.

| Baseline vs GRPO | Error Evolution |
| :---: | :---: |
| ![Baseline vs GRPO Comparison](../experiments/logs/grpo/think/standard/gemma2-2b/eval_20260412_090902/figures/baseline_vs_grpo_comparison.png) | ![Error Evolution](../experiments/logs/grpo/think/standard/gemma2-2b/eval_20260412_090902/figures/errors_grpo_checkpoint-2500.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/standard/gemma2-2b/eval_20260412_090902/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/standard/gemma2-2b/eval_20260412_090902/figures/reward_breakdown.png) |

#### 5.3.2 Curriculum Learning (800-800-900 steps)

**Table 8**: Baseline Pass@1 (Think Curriculum Modality) — off-the-shelf models evaluated with reasoning instructions on the balanced evaluation set.

| Model | Overall | Simple | Medium | Hard |
|:---|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.3167 | 0.5100 | 0.2500 | 0.1900 |
| SmolLM2-360M-Instruct | 0.7133 | 0.7600 | 0.7900 | 0.5900 |
| Qwen2.5-0.5B-Instruct | 0.9167 | 0.9900 | 0.8900 | 0.8700 |
| TinyLlama-1.1B-Chat-v1.0 | 0.8367 | 0.9600 | 0.8400 | 0.7100 |
| Gemma-2-2B-it | 0.9633 | 1.0000 | 0.9600 | 0.9300 |

*(Note: Baseline values show slight variance due to generation sampling under the `<think>` system prompt constraints, but maintain the same overall capability hierarchy).*

**Table 9**: Post-training Pass@1 — after 2500 GRPO steps with 3-stage curriculum (Think Modality).

| Model | Overall | Simple | Medium | Hard | Δ Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.8933 | 0.9400 | 0.8800 | 0.8600 | **+0.5767** |
| SmolLM2-360M-Instruct | 0.9300 | 0.9200 | 0.9700 | 0.9000 | **+0.2167** |
| Qwen2.5-0.5B-Instruct | 0.9800 | 1.0000 | 0.9700 | 0.9700 | **+0.0633** |
| TinyLlama-1.1B-Chat-v1.0 | 0.9767 | 1.0000 | 0.9600 | 0.9700 | **+0.1400** |
| Gemma-2-2B-it | 0.9767 | 0.9900 | 0.9900 | 0.9500 | **+0.0133** |

**Table 10**: Pass@1 at the end of each curriculum stage (Think Modality).

| Model | Baseline | Stage 1 | Stage 2 | Stage 3 |
|:---|:---:|:---:|:---:|:---:|
| SmolLM2-135M-Instruct | 0.3167 | 0.5567 | 0.8000 | 0.8933 |
| SmolLM2-360M-Instruct | 0.7133 | 0.7833 | 0.9033 | 0.9300 |
| Qwen2.5-0.5B-Instruct | 0.9167 | 0.9467 | 0.9333 | 0.9800 |
| TinyLlama-1.1B-Chat-v1.0 | 0.8367 | 0.9433 | 0.9700 | 0.9767 |
| Gemma-2-2B-it | 0.9633 | 0.8300 | 0.6600 | 0.9767 |

Combining curriculum learning with explicit reasoning requirements produces the highest peaks for capable models (Qwen reaches 98% overall), but it reveals a fascinating training dynamic in Table 10. 

Smaller models (SmolLM2, TinyLlama) follow a steady, monotonically increasing curve, benefiting clearly from the gradual scaling of complexity. However, the highly-aligned **Gemma-2-2B** exhibits a severe "U-shaped" learning curve. It collapses during Stage 1 (83.00%) and Stage 2 (66.00%) before fully recovering and surpassing its baseline in Stage 3 (97.67%). This suggests that forcing a highly capable model to generate long reasoning chains for overly simple schemas (Stages 1 and 2) may temporarily confuse its alignment, but once full complexity is restored (Stage 3), the reasoning pathway solidifies into a net positive.

---

**Per-model analysis**

##### SmolLM2-135M-Instruct

> **Eval directory**: [`experiments/logs/grpo/think/curriculum/smollm2-135m/eval_20260413_034728/`](../experiments/logs/grpo/think/curriculum/smollm2-135m/eval_20260413_034728/)

This model climbs steadily and strongly (+57.67 pp). While the final score (89.33%) is roughly equivalent to its No-Think Curriculum counterpart, the progression across stages is smoother. The hard-prompt accuracy vaults from a near-broken 19% to a highly robust 86%.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/think/curriculum/smollm2-135m/eval_20260413_034728/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/think/curriculum/smollm2-135m/eval_20260413_034728/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/think/curriculum/smollm2-135m/eval_20260413_034728/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/curriculum/smollm2-135m/eval_20260413_034728/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/curriculum/smollm2-135m/eval_20260413_034728/figures/reward_breakdown.png) |

---

##### SmolLM2-360M-Instruct

> **Eval directory**: [`experiments/logs/grpo/think/curriculum/smollm2-360m/eval_20260412_154205/`](../experiments/logs/grpo/think/curriculum/smollm2-360m/eval_20260412_154205/)

The 360M model responds beautifully to the Think Curriculum, advancing from 71.33% to 93.00%. The Stage 2 introduction of medium schemas serves as a massive accelerant for this model (jumping from 78.33% to 90.33%), preparing it well for the hard schemas in Stage 3.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/think/curriculum/smollm2-360m/eval_20260412_154205/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/think/curriculum/smollm2-360m/eval_20260412_154205/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/think/curriculum/smollm2-360m/eval_20260412_154205/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/curriculum/smollm2-360m/eval_20260412_154205/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/curriculum/smollm2-360m/eval_20260412_154205/figures/reward_breakdown.png) |

---

##### Qwen2.5-0.5B-Instruct

> **Eval directory**: [`experiments/logs/grpo/think/curriculum/qwen25-05b/eval_20260412_184536/`](../experiments/logs/grpo/think/curriculum/qwen25-05b/eval_20260412_184536/)

Qwen achieves the absolute highest score in the entire study (98.00% overall, with 97% on hard schemas) using this configuration. There is a very slight dip in Stage 2 (93.33%), showing a mild version of the "U-shaped" curve seen in Gemma, but it quickly realigns during Stage 3 to achieve near-flawless structured output generation.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/think/curriculum/qwen25-05b/eval_20260412_184536/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/think/curriculum/qwen25-05b/eval_20260412_184536/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/think/curriculum/qwen25-05b/eval_20260412_184536/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/curriculum/qwen25-05b/eval_20260412_184536/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/curriculum/qwen25-05b/eval_20260412_184536/figures/reward_breakdown.png) |

---

##### TinyLlama-1.1B-Chat-v1.0

> **Eval directory**: [`experiments/logs/grpo/think/curriculum/tinyllama-11b/eval_20260412_213801/`](../experiments/logs/grpo/think/curriculum/tinyllama-11b/eval_20260412_213801/)

TinyLlama shows immense stability. With the Think Curriculum, it rapidly climbs from 83.67% to an exceptional 97.67%. Unlike the standard training run where hard schemas can be overwhelming initially, the curriculum paces the model perfectly, allowing it to score 97% on the hardest evaluations by the end of training.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/think/curriculum/tinyllama-11b/eval_20260412_213801/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/think/curriculum/tinyllama-11b/eval_20260412_213801/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/think/curriculum/tinyllama-11b/eval_20260412_213801/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/curriculum/tinyllama-11b/eval_20260412_213801/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/curriculum/tinyllama-11b/eval_20260412_213801/figures/reward_breakdown.png) |

---

##### Gemma-2-2B-it

> **Eval directory**: [`experiments/logs/grpo/think/curriculum/gemma2-2b/eval_20260413_030321/`](../experiments/logs/grpo/think/curriculum/gemma2-2b/eval_20260413_030321/)

Gemma-2-2B exhibits the most unusual training dynamic of the group. Its performance severely drops in Stage 1 and hits a low of 66.00% in Stage 2, before rebounding aggressively in Stage 3 to 97.67%. This indicates that for highly tuned, larger models, overly simple curriculum stages coupled with forced reasoning tokens might degrade performance temporarily. However, the final state proves that once matched with appropriately hard schemas, the model recalibrates and achieves an outstanding +0.0133 improvement over its already near-perfect baseline.

**Curriculum Progression**
![Curriculum Progression](../experiments/logs/grpo/think/curriculum/gemma2-2b/eval_20260413_030321/figures/curriculum_progression.png)

| Difficulty Heatmap | Error Evolution |
| :---: | :---: |
| ![Heatmap](../experiments/logs/grpo/think/curriculum/gemma2-2b/eval_20260413_030321/figures/stage_difficulty_heatmap.png) | ![Error Evolution](../experiments/logs/grpo/think/curriculum/gemma2-2b/eval_20260413_030321/figures/error_evolution.png) |
| **Rescued vs Regressed** | **Reward Breakdown** |
| ![Rescued](../experiments/logs/grpo/think/curriculum/gemma2-2b/eval_20260413_030321/figures/rescued_vs_regressed.png) | ![Reward Breakdown](../experiments/logs/grpo/think/curriculum/gemma2-2b/eval_20260413_030321/figures/reward_breakdown.png) |

### 5.4 Cross-Modality and Strategy Discussion

* **Think vs. No-Think:** Does reasoning actually help on harder prompts? The data shows a nuanced picture. For smaller architectures (like SmolLM2-135M and SmolLM2-360M), the `<think>` tags act as a crucial structural scratchpad, allowing them to track nested keys and brackets. This is evident in the Standard training, where SmolLM2-135M peaked at 90.00% in Think modality vs 87.67% in No-Think. However, for already highly-capable models, the reasoning overhead doesn't always guarantee a higher peak (e.g., TinyLlama achieved its absolute best of 99.33% in No-Think Curriculum). Ultimately, while `<think>` tokens raise the floor for small models, they come at the cost of higher token consumption and slower inference times.
* **Standard vs. Curriculum Learning:** The curriculum offers a highly tangible advantage, primarily in *training stability and retention*. Standard training on a mixed-difficulty dataset caused the highly-aligned Gemma-2-2B to regress (-1.67 pp in No-Think, -0.67 pp in Think) due to catastrophic forgetting of complex edge cases. Curriculum learning protected Gemma, yielding positive deltas instead. Furthermore, curriculum learning produced the absolute highest scores in the study (e.g., TinyLlama at 99.33% in No-Think, Qwen at 98.00% in Think). The "U-shaped" anomaly observed in Gemma during Think Curriculum suggests that curriculum pacing must be carefully matched to the model's base capacity to avoid temporary alignment confusion.
* **Does model size correlate with improvement?** The relationship follows a distinct **inverted-U pattern** bounded by an absolute performance ceiling. The smallest model (135M) benefits the most, realizing massive absolute gains of +48 to +58 percentage points. Mid-range models (360M, 1.1B) show substantial +13 to +26 pp improvements. Conversely, the highly capable 0.5B and 2B models see marginal gains (+0 to +8 pp) simply because their baselines are already in the 90–98% range, leaving almost no room for upward movement.
* **Which difficulty tier benefits most?** The `hard` tier consistently drives the overall improvement across all models and modalities. Baselines for hard prompts were extremely brittle (e.g., 17–20% for SmolLM2-135M). Post-GRPO, hard prompt pass rates routinely reached 86–99%. Simple prompts, by contrast, saturated very early (often hitting 100% by Stage 1 of the curriculum), confirming that deep nesting and exact key constraints are the primary skills learned during this RLHF process.
* **Error type analysis & Convergence.** Despite a 14× parameter gap (135M vs 2B) and varying baseline capabilities, all models converge to a remarkably tight **87–99% performance band** after 2500 steps. The dominant baseline error across most models was `no_code_block` (generating raw JSON without the required markdown fences). Post-training, this error is almost entirely eradicated. The few remaining failures are predominantly minor `json_error` instances (e.g., trailing commas or missing closing brackets in deeply nested structures), rather than complete structural misunderstandings.

---

## 6. Conclusion and Limitations

### Summary

We presented a reproducible pipeline for aligning small LLMs to strict JSON generation using GRPO with rule-based rewards. The approach is lightweight (single GPU, LoRA, 4-bit quantisation) and does not require human annotations or a neural reward model. We extensively evaluated this approach across four configurations: Standard vs. Curriculum Learning, and Think vs. No-Think modalities.

Across five models spanning a 14× parameter range (135 M – 2 B), the GRPO training consistently and drastically improved JSON compliance. The weakest model (SmolLM2-135M) improved from an unreliable ~31-39% baseline to a robust ~87-90% pass rate (peaking at +58.67 pp in Think Standard). Meanwhile, the strongest baselines (Gemma-2-2B and Qwen2.5-0.5B) were successfully polished to near-perfection (~98%) without catastrophic forgetting, provided Curriculum Learning was employed. All models converged to a tight 87–99% band after 2 500 training steps, demonstrating that rule-based GRPO—especially when paired with structured curriculum pacing—is an exceptionally effective method for teaching strict structured output. This effectively neutralises the practical capability gap between extreme small-scale models and larger 2B architectures for this specific task.

### Limitations

- **Synthetic prompts only.** The template-based dataset, while diverse (24 templates × random parametrisation), does not cover all real-world JSON use cases. Models may not generalise perfectly to free-form user requests outside the template distribution.
- **Single evaluation metric.** Pass@1 measures whether the output is valid JSON conforming to the requested structural schema, but it does not assess semantic quality (e.g., whether the generated field values are factually accurate or contextually plausible).
- **No comparison with DPO/PPO.** We solely tested GRPO; comparing these results against other alignment methods like DPO or PPO would help contextualise the efficiency of the RLHF algorithm itself.
- **Fixed hyperparameters.** The same LoRA rank, learning rate, and reward weights were applied universally. Per-model tuning (especially adjusting the curriculum difficulty stages for already-strong models like Gemma) could likely smooth out training anomalies and improve individual peaks.

### Future work

- **Real-world benchmark.** Evaluate the trained models on established structured-output benchmarks (e.g., complex tool-use datasets, nested function-calling benchmarks) to test out-of-distribution generalisation.
- **Reward ablation.** Systematically disable or reweight individual reward components (such as the schema penalty vs. the strictness penalty) to measure their marginal contribution to the final performance.
- **Dynamic Curriculum Scaling.** Investigate the "U-shaped" degradation observed in highly capable models during early curriculum stages, potentially developing a dynamic curriculum that skips overly simplistic data if the model demonstrates early mastery.
- **Scale up.** Apply the same RLHF pipeline to 7 B+ parameter models to establish an upper bound and assess if rule-based GRPO can push larger models to absolute 100% zero-shot reliability.

## 7. Additional Information

### 7.1 Contribution Breakdown
- **Giuseppe Bellamacina**: Full project design and implementation — synthetic dataset generator, reward functions, curriculum training pipeline, evaluation framework, cluster infrastructure (SLURM scripts, Apptainer container, monitoring tools), documentation, and analysis.

### 7.2 Use of Artificial Intelligence
GitHub Copilot and Claude were used as coding assistants throughout the project for:
- **Boilerplate generation**: SLURM scripts, Docker/Apptainer configuration, YAML configs.
- **Debugging**: Diagnosing CUDA OOM errors, SLURM job failures, tokenizer compatibility issues.
- **Documentation**: Drafting and reviewing markdown documentation.
- **Code review**: Identifying edge cases in reward functions and evaluation logic.

All architectural decisions (reward component design, curriculum staging, model selection, quantisation strategy) were made by the author. The AI tools accelerated implementation but did not originate the methodology.

## References

[1] B. Agarwal, I. Joshi, V. Rojkova, "Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence," *arXiv:2502.14905*, 2025. [Paper](https://arxiv.org/abs/2502.14905) · [PDF](papers/2502.14905v1.pdf)

[2] R. Hu, S. Wu, "RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs," *arXiv:2512.00319*, 2025. [Paper](https://arxiv.org/abs/2512.00319) · [PDF](papers/2512.00319v2.pdf)

[3] F. Pennino, B. Raimondi, M. Rondelli, A. Gurioli, M. Gabbrielli, "From Reasoning to Code: GRPO Optimization for Underrepresented Languages," *arXiv:2506.11027*, 2025. [Paper](https://arxiv.org/abs/2506.11027) · [PDF](papers/2506.11027v2.pdf)

[4] C. Qian, E. C. Acikgoz, Q. He, H. Wang, X. Chen, D. Hakkani-Tür, G. Tur, H. Ji, "ToolRL: Reward is All Tool Learning Needs," *arXiv:2504.13958*, 2025. [Paper](https://arxiv.org/abs/2504.13958) · [PDF](papers/2504.13958v1.pdf)

[5] P. Jayarao, H. Gupta, N. Varshney, C. Dwivedi, "Explicit Reasoning Makes Better Judges: A Systematic Study on Accuracy, Efficiency, and Robustness," *arXiv:2509.13332*, 2025. [Paper](https://arxiv.org/abs/2509.13332) · [PDF](papers/2509.13332v1.pdf)

[6] G. Fang, X. Ma, X. Wang, "Thinkless: LLM Learns When to Think," *arXiv:2505.13379*, 2025. [Paper](https://arxiv.org/abs/2505.13379) · [PDF](papers/2505.13379v2.pdf)

[7] H. Huang, Y. He, H. Zhou, R. Zhang, W. Liu, W. Wang, J. Liu, W. Su, "Think-J: Learning to Think for Generative LLM-as-a-Judge," *arXiv:2505.14268*, 2026. [Paper](https://arxiv.org/abs/2505.14268) · [PDF](papers/2505.14268v2.pdf)

[8] Unsloth Documentation. [https://unsloth.ai/docs](https://unsloth.ai/docs)

[9] Ando AI, "AI GRPO — A Deep Dive into Group Relative Policy Optimization." [https://blog.ando.ai/posts/ai-grpo/](https://blog.ando.ai/posts/ai-grpo/)

[10] L. Bometon, "Fine-Tuning GRPO with LLM Judge: From Zero to Production," Medium, 2025. [https://medium.com/@lbometon2/fine-tuning-grpo-with-llm-judge-from-zero-to-production-69a25a4ab3bd](https://medium.com/@lbometon2/fine-tuning-grpo-with-llm-judge-from-zero-to-production-69a25a4ab3bd)

[11] Patronus AI, "Guide to RL Environments for LLMs." [https://www.patronus.ai/guide-to-rl-environments](https://www.patronus.ai/guide-to-rl-environments)
