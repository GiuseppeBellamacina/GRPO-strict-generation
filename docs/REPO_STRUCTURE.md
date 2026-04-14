# Complete Repository Structure

```text
├── 📁 .devcontainer
│   └── ⚙️ devcontainer.json
├── 📁 .githooks
│   ├── 📝 README.md
│   └── 📄 pre-push
├── 📁 cluster
│   ├── 📄 aliases.sh
│   ├── 📄 chain_next.sh
│   ├── 📄 clean.sh
│   ├── 📄 clean_model.sh
│   ├── 📄 eval.sh
│   ├── 📄 run_all.sh
│   ├── 📄 setup.sh
│   └── 📄 train.sh
├── 📁 data
│   └── ⚙️ .gitkeep
├── 📁 docs
│   ├── 📁 papers
│   │   ├── 📕 2502.14905v1.pdf
│   │   ├── 📕 2504.13958v1.pdf
│   │   ├── 📕 2505.13379v2.pdf
│   │   ├── 📕 2505.14268v2.pdf
│   │   ├── 📕 2506.11027v2.pdf
│   │   ├── 📕 2509.13332v1.pdf
│   │   └── 📕 2512.00319v2.pdf
│   ├── 📝 CLUSTER.md
│   ├── 📝 GUIDA_CLUSTER_DMI.md
│   ├── 📝 MODELS.md
│   ├── 📝 QUICK_SETUP.md
│   ├── 📝 REFERENCES.md
│   ├── 📝 REPORT.md
│   ├── 📝 REPO_STRUCTURE.md
│   └── 📝 SLURM_COMMANDS.md
├── 📁 experiments
│   ├── 📁 configs
│   │   ├── 📁 nothink
│   │   │   ├── 📁 curriculum
│   │   │   │   ├── ⚙️ grpo_gemma2.yaml
│   │   │   │   ├── ⚙️ grpo_qwen05.yaml
│   │   │   │   ├── ⚙️ grpo_smollm2_135m.yaml
│   │   │   │   ├── ⚙️ grpo_smollm2_360m.yaml
│   │   │   │   └── ⚙️ grpo_tinyllama.yaml
│   │   │   └── 📁 standard
│   │   │       ├── ⚙️ grpo_gemma2.yaml
│   │   │       ├── ⚙️ grpo_qwen05.yaml
│   │   │       ├── ⚙️ grpo_smollm2_135m.yaml
│   │   │       ├── ⚙️ grpo_smollm2_360m.yaml
│   │   │       └── ⚙️ grpo_tinyllama.yaml
│   │   ├── 📁 think
│   │   │   ├── 📁 curriculum
│   │   │   │   ├── ⚙️ grpo_gemma2.yaml
│   │   │   │   ├── ⚙️ grpo_qwen05.yaml
│   │   │   │   ├── ⚙️ grpo_smollm2_135m.yaml
│   │   │   │   ├── ⚙️ grpo_smollm2_360m.yaml
│   │   │   │   └── ⚙️ grpo_tinyllama.yaml
│   │   │   └── 📁 standard
│   │   │       ├── ⚙️ grpo_gemma2.yaml
│   │   │       ├── ⚙️ grpo_qwen05.yaml
│   │   │       ├── ⚙️ grpo_smollm2_135m.yaml
│   │   │       ├── ⚙️ grpo_smollm2_360m.yaml
│   │   │       └── ⚙️ grpo_tinyllama.yaml
│   │   └── ⚙️ sft.yaml
│   └── 📁 logs
│       └── 📁 grpo
│           ├── 📁 nothink
│           │   ├── 📁 curriculum
│           │   │   ├── 📁 gemma2-2b
│           │   │   │   ├── 📁 eval_20260411_010201
│           │   │   │   │   ├── 📁 figures
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ curriculum_progression.png
│           │   │   │   │   │   ├── 🖼️ error_evolution.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │   │   │   │   ├── 🖼️ reward_breakdown.png
│           │   │   │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │   │   │   ├── ⚙️ comparison.json
│           │   │   │   │   ├── ⚙️ completions_baseline.json
│           │   │   │   │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │   │   │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │   │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │   │   │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │   │   │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │   │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   │   │   ├── 📁 train_20260410_205437
│           │   │   │   └── ⚙️ baseline_results.json
│           │   │   ├── 📁 qwen25-05b
│           │   │   │   ├── 📁 eval_20260410_174001
│           │   │   │   │   ├── 📁 figures
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ curriculum_progression.png
│           │   │   │   │   │   ├── 🖼️ error_evolution.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │   │   │   │   ├── 🖼️ reward_breakdown.png
│           │   │   │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │   │   │   ├── ⚙️ comparison.json
│           │   │   │   │   ├── ⚙️ completions_baseline.json
│           │   │   │   │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │   │   │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │   │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │   │   │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │   │   │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │   │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   │   │   ├── 📁 train_20260410_153846
│           │   │   │   └── ⚙️ baseline_results.json
│           │   │   ├── 📁 smollm2-135m
│           │   │   │   ├── 📁 eval_20260410_060558
│           │   │   │   │   ├── 📁 figures
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ curriculum_progression.png
│           │   │   │   │   │   ├── 🖼️ error_evolution.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │   │   │   │   ├── 🖼️ reward_breakdown.png
│           │   │   │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │   │   │   ├── ⚙️ comparison.json
│           │   │   │   │   ├── ⚙️ completions_baseline.json
│           │   │   │   │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │   │   │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │   │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │   │   │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │   │   │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │   │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   │   │   ├── 📁 train_20260410_032332
│           │   │   │   └── ⚙️ baseline_results.json
│           │   │   ├── 📁 smollm2-360m
│           │   │   │   ├── 📁 eval_20260410_094244
│           │   │   │   │   ├── 📁 figures
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ curriculum_progression.png
│           │   │   │   │   │   ├── 🖼️ error_evolution.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │   │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │   │   │   │   ├── 🖼️ reward_breakdown.png
│           │   │   │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │   │   │   ├── ⚙️ comparison.json
│           │   │   │   │   ├── ⚙️ completions_baseline.json
│           │   │   │   │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │   │   │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │   │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │   │   │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │   │   │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │   │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   │   │   ├── 📁 train_20260410_065529
│           │   │   │   └── ⚙️ baseline_results.json
│           │   │   └── 📁 tinyllama-11b
│           │   │       ├── 📁 eval_20260410_202131
│           │   │       │   ├── 📁 figures
│           │   │       │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │       │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │       │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │       │   │   ├── 🖼️ curriculum_progression.png
│           │   │       │   │   ├── 🖼️ error_evolution.png
│           │   │       │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │       │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │       │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │       │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │       │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │       │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │       │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │       │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │       │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │       │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │       │   │   ├── 🖼️ reward_breakdown.png
│           │   │       │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │       │   ├── ⚙️ comparison.json
│           │   │       │   ├── ⚙️ completions_baseline.json
│           │   │       │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │       │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │       │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │       │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │       │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │       │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   │       ├── 📁 train_20260410_175911
│           │   │       └── ⚙️ baseline_results.json
│           │   └── 📁 standard
│           │       ├── 📁 gemma2-2b
│           │       │   ├── 📁 eval_20260410_151742
│           │       │   │   ├── 📁 figures
│           │       │   │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│           │       │   │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │       │   │   │   └── 🖼️ reward_breakdown.png
│           │       │   │   ├── ⚙️ comparison.json
│           │       │   │   ├── ⚙️ completions_baseline.json
│           │       │   │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│           │       │   │   └── ⚙️ eval_grpo_checkpoint-2500.json
│           │       │   ├── 📁 train_20260410_110522
│           │       │   └── ⚙️ baseline_results.json
│           │       ├── 📁 qwen25-05b
│           │       │   ├── 📁 eval_20260409_205251
│           │       │   │   ├── 📁 figures
│           │       │   │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│           │       │   │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │       │   │   │   └── 🖼️ reward_breakdown.png
│           │       │   │   ├── ⚙️ comparison.json
│           │       │   │   ├── ⚙️ completions_baseline.json
│           │       │   │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│           │       │   │   └── ⚙️ eval_grpo_checkpoint-2500.json
│           │       │   ├── 📁 train_20260409_184531
│           │       │   └── ⚙️ baseline_results.json
│           │       ├── 📁 smollm2-135m
│           │       │   ├── 📁 eval_20260409_152243
│           │       │   │   ├── 📁 figures
│           │       │   │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│           │       │   │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │       │   │   │   └── 🖼️ reward_breakdown.png
│           │       │   │   ├── ⚙️ comparison.json
│           │       │   │   ├── ⚙️ completions_baseline.json
│           │       │   │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│           │       │   │   └── ⚙️ eval_grpo_checkpoint-2500.json
│           │       │   ├── 📁 train_20260409_124833
│           │       │   └── ⚙️ baseline_results.json
│           │       ├── 📁 smollm2-360m
│           │       │   ├── 📁 eval_20260409_182519
│           │       │   │   ├── 📁 figures
│           │       │   │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│           │       │   │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│           │       │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │       │   │   │   └── 🖼️ reward_breakdown.png
│           │       │   │   ├── ⚙️ comparison.json
│           │       │   │   ├── ⚙️ completions_baseline.json
│           │       │   │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│           │       │   │   └── ⚙️ eval_grpo_checkpoint-2500.json
│           │       │   ├── 📁 train_20260409_154457
│           │       │   └── ⚙️ baseline_results.json
│           │       └── 📁 tinyllama-11b
│           │           ├── 📁 eval_20260410_023046
│           │           │   ├── 📁 figures
│           │           │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│           │           │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│           │           │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│           │           │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│           │           │   │   ├── 🖼️ rescued_vs_regressed.png
│           │           │   │   └── 🖼️ reward_breakdown.png
│           │           │   ├── ⚙️ comparison.json
│           │           │   ├── ⚙️ completions_baseline.json
│           │           │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│           │           │   └── ⚙️ eval_grpo_checkpoint-2500.json
│           │           ├── 📁 train_20260410_000830
│           │           └── ⚙️ baseline_results.json
│           └── 📁 think
│               ├── 📁 curriculum
│               │   ├── 📁 gemma2-2b
│               │   │   ├── 📁 eval_20260413_030321
│               │   │   │   ├── 📁 figures
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ curriculum_progression.png
│               │   │   │   │   ├── 🖼️ error_evolution.png
│               │   │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│               │   │   │   │   ├── 🖼️ reward_breakdown.png
│               │   │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│               │   │   │   ├── ⚙️ comparison.json
│               │   │   │   ├── ⚙️ completions_baseline.json
│               │   │   │   ├── ⚙️ completions_stage_1_format_basics.json
│               │   │   │   ├── ⚙️ completions_stage_2_progressive.json
│               │   │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│               │   │   │   ├── ⚙️ eval_stage_1_format_basics.json
│               │   │   │   ├── ⚙️ eval_stage_2_progressive.json
│               │   │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│               │   │   ├── 📁 train_20260412_222102
│               │   │   └── ⚙️ baseline_results.json
│               │   ├── 📁 qwen25-05b
│               │   │   ├── 📁 eval_20260412_184536
│               │   │   │   ├── 📁 figures
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ curriculum_progression.png
│               │   │   │   │   ├── 🖼️ error_evolution.png
│               │   │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│               │   │   │   │   ├── 🖼️ reward_breakdown.png
│               │   │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│               │   │   │   ├── ⚙️ comparison.json
│               │   │   │   ├── ⚙️ completions_baseline.json
│               │   │   │   ├── ⚙️ completions_stage_1_format_basics.json
│               │   │   │   ├── ⚙️ completions_stage_2_progressive.json
│               │   │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│               │   │   │   ├── ⚙️ eval_stage_1_format_basics.json
│               │   │   │   ├── ⚙️ eval_stage_2_progressive.json
│               │   │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│               │   │   ├── 📁 train_20260412_164117
│               │   │   └── ⚙️ baseline_results.json
│               │   ├── 📁 smollm2-135m
│               │   │   ├── 📁 eval_20260413_034728
│               │   │   │   ├── 📁 figures
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ curriculum_progression.png
│               │   │   │   │   ├── 🖼️ error_evolution.png
│               │   │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│               │   │   │   │   ├── 🖼️ reward_breakdown.png
│               │   │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│               │   │   │   ├── ⚙️ comparison.json
│               │   │   │   ├── ⚙️ completions_baseline.json
│               │   │   │   ├── ⚙️ completions_stage_1_format_basics.json
│               │   │   │   ├── ⚙️ completions_stage_2_progressive.json
│               │   │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│               │   │   │   ├── ⚙️ eval_stage_1_format_basics.json
│               │   │   │   ├── ⚙️ eval_stage_2_progressive.json
│               │   │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│               │   │   ├── 📁 train_20260412_093246
│               │   │   └── ⚙️ baseline_results.json
│               │   ├── 📁 smollm2-360m
│               │   │   ├── 📁 eval_20260412_154205
│               │   │   │   ├── 📁 figures
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ curriculum_progression.png
│               │   │   │   │   ├── 🖼️ error_evolution.png
│               │   │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│               │   │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│               │   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│               │   │   │   │   ├── 🖼️ reward_breakdown.png
│               │   │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│               │   │   │   ├── ⚙️ comparison.json
│               │   │   │   ├── ⚙️ completions_baseline.json
│               │   │   │   ├── ⚙️ completions_stage_1_format_basics.json
│               │   │   │   ├── ⚙️ completions_stage_2_progressive.json
│               │   │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│               │   │   │   ├── ⚙️ eval_stage_1_format_basics.json
│               │   │   │   ├── ⚙️ eval_stage_2_progressive.json
│               │   │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│               │   │   ├── 📁 train_20260412_123900
│               │   │   └── ⚙️ baseline_results.json
│               │   └── 📁 tinyllama-11b
│               │       ├── 📁 eval_20260412_213801
│               │       │   ├── 📁 figures
│               │       │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│               │       │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│               │       │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│               │       │   │   ├── 🖼️ curriculum_progression.png
│               │       │   │   ├── 🖼️ error_evolution.png
│               │       │   │   ├── 🖼️ errors_stage_1_format_basics.png
│               │       │   │   ├── 🖼️ errors_stage_2_progressive.png
│               │       │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│               │       │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│               │       │   │   ├── 🖼️ lengths_stage_2_progressive.png
│               │       │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│               │       │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│               │       │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│               │       │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│               │       │   │   ├── 🖼️ rescued_vs_regressed.png
│               │       │   │   ├── 🖼️ reward_breakdown.png
│               │       │   │   └── 🖼️ stage_difficulty_heatmap.png
│               │       │   ├── ⚙️ comparison.json
│               │       │   ├── ⚙️ completions_baseline.json
│               │       │   ├── ⚙️ completions_stage_1_format_basics.json
│               │       │   ├── ⚙️ completions_stage_2_progressive.json
│               │       │   ├── ⚙️ completions_stage_3_full_difficulty.json
│               │       │   ├── ⚙️ eval_stage_1_format_basics.json
│               │       │   ├── ⚙️ eval_stage_2_progressive.json
│               │       │   └── ⚙️ eval_stage_3_full_difficulty.json
│               │       ├── 📁 train_20260412_190738
│               │       └── ⚙️ baseline_results.json
│               └── 📁 standard
│                   ├── 📁 gemma2-2b
│                   │   ├── 📁 eval_20260412_090902
│                   │   │   ├── 📁 figures
│                   │   │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│                   │   │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│                   │   │   │   └── 🖼️ reward_breakdown.png
│                   │   │   ├── ⚙️ comparison.json
│                   │   │   ├── ⚙️ completions_baseline.json
│                   │   │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│                   │   │   └── ⚙️ eval_grpo_checkpoint-2500.json
│                   │   ├── 📁 train_20260412_040035
│                   │   └── ⚙️ baseline_results.json
│                   ├── 📁 qwen25-05b
│                   │   ├── 📁 eval_20260412_010300
│                   │   │   ├── 📁 figures
│                   │   │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│                   │   │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│                   │   │   │   └── 🖼️ reward_breakdown.png
│                   │   │   ├── ⚙️ comparison.json
│                   │   │   ├── ⚙️ completions_baseline.json
│                   │   │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│                   │   │   └── ⚙️ eval_grpo_checkpoint-2500.json
│                   │   ├── 📁 train_20260411_225940
│                   │   └── ⚙️ baseline_results.json
│                   ├── 📁 smollm2-135m
│                   │   ├── 📁 eval_20260411_192610
│                   │   │   ├── 📁 figures
│                   │   │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│                   │   │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│                   │   │   │   └── 🖼️ reward_breakdown.png
│                   │   │   ├── ⚙️ comparison.json
│                   │   │   ├── ⚙️ completions_baseline.json
│                   │   │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│                   │   │   └── ⚙️ eval_grpo_checkpoint-2500.json
│                   │   ├── 📁 train_20260411_164110
│                   │   └── ⚙️ baseline_results.json
│                   ├── 📁 smollm2-360m
│                   │   ├── 📁 eval_20260411_223435
│                   │   │   ├── 📁 figures
│                   │   │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│                   │   │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│                   │   │   │   ├── 🖼️ rescued_vs_regressed.png
│                   │   │   │   └── 🖼️ reward_breakdown.png
│                   │   │   ├── ⚙️ comparison.json
│                   │   │   ├── ⚙️ completions_baseline.json
│                   │   │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│                   │   │   └── ⚙️ eval_grpo_checkpoint-2500.json
│                   │   ├── 📁 train_20260411_195121
│                   │   └── ⚙️ baseline_results.json
│                   └── 📁 tinyllama-11b
│                       ├── 📁 eval_20260412_034025
│                       │   ├── 📁 figures
│                       │   │   ├── 🖼️ baseline_vs_grpo_comparison.png
│                       │   │   ├── 🖼️ errors_grpo_checkpoint-2500.png
│                       │   │   ├── 🖼️ lengths_grpo_checkpoint-2500.png
│                       │   │   ├── 🖼️ pass_rates_grpo_checkpoint-2500.png
│                       │   │   ├── 🖼️ rescued_vs_regressed.png
│                       │   │   └── 🖼️ reward_breakdown.png
│                       │   ├── ⚙️ comparison.json
│                       │   ├── ⚙️ completions_baseline.json
│                       │   ├── ⚙️ completions_grpo_checkpoint-2500.json
│                       │   └── ⚙️ eval_grpo_checkpoint-2500.json
│                       ├── 📁 train_20260412_011410
│                       └── ⚙️ baseline_results.json
├── 📁 notebooks
│   ├── 📁 reference
│   │   ├── 📄 Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
│   │   └── 📄 Llama3_1_(8B)_GRPO.ipynb
│   ├── 📄 01_test_config_and_train.ipynb
│   ├── 📄 02_test_config_and_train_fast.ipynb
│   └── 📄 03_full_pipeline.ipynb
├── 📁 src
│   ├── 📁 datasets
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 dataloader.py
│   │   ├── 🐍 synthetic_dataset.py
│   │   └── 🐍 templates.py
│   ├── 📁 evaluation
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 __main__.py
│   │   ├── 🐍 eval_baseline.py
│   │   ├── 🐍 eval_dataset.py
│   │   └── 🐍 eval_grpo.py
│   ├── 📁 models
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 model_loader.py
│   ├── 📁 training
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 __main__.py
│   │   ├── 🐍 callbacks.py
│   │   ├── 🐍 grpo_train.py
│   │   ├── 🐍 grpo_vanilla.py
│   │   ├── 🐍 rewards.py
│   │   └── 🐍 sft_train.py
│   ├── 📁 utils
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 chain_monitor.py
│   │   ├── 🐍 compare_think.py
│   │   ├── 🐍 config.py
│   │   ├── 🐍 distributed.py
│   │   ├── 🐍 live_training_table.py
│   │   ├── 🐍 metrics.py
│   │   ├── 🐍 show_training_log.py
│   │   └── 🐍 visualization.py
│   └── 🐍 __init__.py
├── 📁 tests
│   ├── 🐍 __init__.py
│   └── 🐍 test_rewards.py
├── ⚙️ .dockerignore
├── ⚙️ .env.example
├── ⚙️ .gitattributes
├── ⚙️ .gitignore
├── 🐳 Dockerfile
├── 📄 LICENSE
├── 📝 README.md
├── ⚙️ docker-compose.yml
├── 📄 format.ps1
├── 📄 format.sh
├── ⚙️ pyproject.toml
├── 📄 setup.sh
└── 📄 sync_cluster.ps1
```