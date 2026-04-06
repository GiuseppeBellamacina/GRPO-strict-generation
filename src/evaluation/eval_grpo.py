"""Post-training evaluation — evaluate GRPO-trained model and compare with baseline.

Usage:
    # Evaluate GRPO model only
    python -m src.evaluation --config experiments/configs/grpo_smollm2_360m.yaml

    # Evaluate GRPO model + compare with baseline results
    python -m src.evaluation --config experiments/configs/grpo_smollm2_360m.yaml --compare

    # Evaluate a specific checkpoint
    python -m src.evaluation --config experiments/configs/grpo_smollm2_360m.yaml \
        --checkpoint experiments/checkpoints/grpo/checkpoint-480
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import torch
import wandb
from dotenv import load_dotenv

from src.datasets.dataloader import (
    format_prompt_for_model,
)
from src.evaluation.eval_baseline import generate_completions
from src.models.model_loader import (
    load_model_and_tokenizer,
    load_tokenizer,
)
from src.utils.config import load_config
from src.utils.metrics import (
    check_syntax,
    compute_detailed_metrics,
    compute_reward_breakdown,
)
from src.utils.visualization import (
    plot_baseline_vs_grpo_comparison,
    plot_completion_length_distribution,
    plot_completions_error_breakdown,
    plot_curriculum_progression,
    plot_error_evolution,
    plot_per_category_breakdown,
    plot_rescued_vs_regressed,
    plot_reward_breakdown,
    plot_stage_difficulty_heatmap,
)

load_dotenv()


def _evaluate_model(
    config: dict[str, Any],
    model_path: str,
    prompts: list[str],
    difficulties: list[str],
    gen_config: dict[str, Any],
    is_checkpoint: bool = False,
    batch_size: int = 8,
) -> tuple[dict[str, Any], list[str]]:
    """Load a model from path, generate completions, compute metrics.

    Args:
        is_checkpoint: If True, treat model_path as a PEFT/LoRA checkpoint
                       and load adapters on top of the base model.
        batch_size: Number of prompts per generation batch.
    """
    from pathlib import Path as _Path

    model_cfg = config["model"]
    use_unsloth = model_cfg.get("use_unsloth", False)

    # Try Unsloth-accelerated loading (2x faster inference, ~50% less VRAM).
    # fast_inference (vLLM) is always disabled to avoid VRAM saturation.
    _unsloth_ok = False
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            _unsloth_ok = True
        except ImportError:
            _unsloth_ok = False

    if _unsloth_ok:
        # ── Unsloth path ─────────────────────────────────────────────────
        quantization = model_cfg.get("quantization", "4bit")
        base_name = model_cfg["name"]

        if (
            is_checkpoint
            and (_Path(model_path) / "adapter_config.json").exists()
        ):
            # PEFT checkpoint — load base model via Unsloth, attach adapter
            from peft import PeftModel

            print(f"  Loading base model {base_name} (Unsloth)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_name,
                max_seq_length=model_cfg.get("max_seq_length", 2048),
                load_in_4bit=(quantization == "4bit"),
                dtype=None,
            )
            print(f"  Loading LoRA adapters from {model_path}...")
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # Full model or base model (baseline)
            print(f"  Loading model from {model_path} (Unsloth)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=model_cfg.get("max_seq_length", 2048),
                load_in_4bit=(quantization == "4bit"),
                dtype=None,
            )

        FastLanguageModel.for_inference(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        # ── Vanilla HuggingFace path (fallback) ─────────────────────────
        if (
            is_checkpoint
            and (_Path(model_path) / "adapter_config.json").exists()
        ):
            from peft import PeftModel

            print(f"  Loading base model {model_cfg['name']}...")
            base_config = {
                "model": {
                    **model_cfg,
                    "fast_inference": False,
                    "use_unsloth": False,
                }
            }
            model, tokenizer = load_model_and_tokenizer(base_config)
            print(f"  Loading LoRA adapters from {model_path}...")
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
        else:
            eval_config = {
                "model": {
                    **model_cfg,
                    "name": model_path,
                    "fast_inference": False,
                    "use_unsloth": False,
                }
            }
            print(f"  Loading model from {model_path}...")
            model, tokenizer = load_model_and_tokenizer(eval_config)

    print(f"  Generating completions for {len(prompts)} prompts...")
    completions_per_prompt = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=gen_config,
        num_return_sequences=1,
        batch_size=batch_size,
    )

    first_completions = [comps[0] for comps in completions_per_prompt]
    metrics = compute_detailed_metrics(
        first_completions, difficulties
    )

    # Free memory
    del model, tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    return metrics, first_completions


def _safe_filename(label: str) -> str:
    """Turn a human-readable label into a filesystem-safe name."""
    return (
        label.lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .strip("_")
    )


def _print_eval_samples(
    prompts: list[str],
    completions: list[str],
    difficulties: list[str],
    n: int = 5,
    raw_prompts: list[str] | None = None,
) -> None:
    """Print the first *n* eval completions with per-component rewards."""
    from src.training.callbacks import _split_think
    from src.training.rewards import (
        format_reward,
        reasoning_reward,
        repetition_reward,
        schema_reward,
        strictness_reward,
        truncation_reward,
        validity_reward,
    )

    sep = "─" * 70
    print(f"\n{'═' * 70}")
    print("  EVAL COMPLETION SAMPLES")
    print(f"{'═' * 70}")
    for i in range(min(n, len(completions))):
        comp = completions[i]
        diff = difficulties[i]
        # Truncate prompt to user instruction
        prompt_short = prompts[i]
        if len(prompt_short) > 150:
            prompt_short = prompt_short[:150] + " [...]"
        # Per-component rewards
        fmt = format_reward(comp)
        val = validity_reward(comp)
        raw_p = raw_prompts[i] if raw_prompts else ""
        sch = schema_reward(comp, prompts[i], raw_prompt=raw_p)
        reas = reasoning_reward(comp)
        trunc = truncation_reward(comp)
        rep = repetition_reward(comp)
        strict = strictness_reward(comp)

        think, output = _split_think(comp)

        print(f"\n{sep}")
        print(f"  Sample {i + 1}  [difficulty={diff}]")
        print(sep)
        print(f"  PROMPT: {prompt_short}")
        if think:
            think_display = (
                think if len(think) <= 200 else think[:200] + " [...]"
            )
            print("  THINK:")
            for line in think_display.splitlines():
                print(f"    {line}")
        output_display = (
            output if len(output) <= 300 else output[:300] + " [...]"
        )
        print("  OUTPUT:")
        for line in output_display.splitlines():
            print(f"    {line}")
        print(
            f"  REWARDS: format={fmt:+.2f}  validity={val:+.2f}  "
            f"schema={sch:+.2f}  reasoning={reas:+.2f}"
        )
        print(
            f"           truncation={trunc:+.2f}  repetition={rep:+.2f}  "
            f"strictness={strict:+.2f}"
        )
    print(f"{'═' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training evaluation of GRPO model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint (default: output_dir/final)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run baseline eval and generate comparison figures",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Evaluate all curriculum stages (looks for stages/ subdirectory "
        "in output_dir).  Implies --compare.",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Path to existing baseline results.json (default: <log_dir>/baseline_results.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="Max test samples to evaluate (default: 300)",
    )
    parser.add_argument(
        "--skip-stages",
        type=int,
        default=0,
        help="Skip the first N stages (for resuming interrupted eval)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Resolve versioned directories ─────────────────────────────────────
    # Checkpoints: resolve output_dir to the latest training run.
    # Eval results: create a new timestamped eval subdirectory under log_dir.
    from src.utils.config import resolve_latest_run, resolve_run_dir

    base_output = config["training"]["output_dir"]
    config["training"]["output_dir"] = str(
        resolve_latest_run(base_output)
    )

    base_log = config["training"].get(
        "log_dir", "experiments/logs/grpo"
    )

    # Baseline results live at the base log_dir level (shared across evals)
    if args.baseline_results is None:
        args.baseline_results = str(
            Path(base_log) / "baseline_results.json"
        )

    # New versioned subdirectory for this eval run
    eval_run_dir, eval_run_id = resolve_run_dir(
        base_log, prefix="eval"
    )
    config["training"]["log_dir"] = str(eval_run_dir)
    print(f"[eval] Run: {eval_run_id}")
    print(f"[eval]   results → {eval_run_dir}")

    # Detect curriculum from config (curriculum.enabled in YAML)
    # Used for checkpoint discovery (stage_*/checkpoint-* layout).
    curriculum_cfg = config.get("curriculum", {})
    is_curriculum = curriculum_cfg.get("enabled", False)

    # --curriculum implies --compare
    if args.curriculum:
        args.compare = True

    # Determine checkpoint path
    output_dir = config["training"]["output_dir"]
    ckpt_path = None

    if args.checkpoint:
        # Explicit checkpoint always wins
        ckpt_path = args.checkpoint
    elif (Path(output_dir) / "final").exists():
        ckpt_path = str(Path(output_dir) / "final")
    elif is_curriculum:
        # Curriculum training: checkpoints are in stage_X_name/checkpoint-*
        # and final stage models in stages/stage_X_name/.
        # For tokenizer loading we need any valid model dir.
        stages_dir = Path(output_dir) / "stages"
        if stages_dir.exists():
            stage_dirs = sorted(
                [d for d in stages_dir.iterdir() if d.is_dir()]
            )
        else:
            stage_dirs = []

        if stage_dirs:
            ckpt_path = str(stage_dirs[-1])
        else:
            # stages/ not yet saved — fall back to intermediate checkpoints
            # inside stage_X_name/checkpoint-* dirs
            stage_ckpts = sorted(
                Path(output_dir).glob("stage_*/checkpoint-*")
            )
            if stage_ckpts:
                ckpt_path = str(stage_ckpts[-1])
            else:
                print(
                    f"No checkpoint found in {output_dir}/stages/ "
                    f"or {output_dir}/stage_*/checkpoint-*"
                )
                return
    else:
        # Non-curriculum: checkpoints are directly in output_dir
        ckpts = sorted(Path(output_dir).glob("checkpoint-*"))
        if ckpts:
            ckpt_path = str(ckpts[-1])
        else:
            print(f"No checkpoint found in {output_dir}")
            return

    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"[eval] Curriculum: {is_curriculum}")
    print(f"[eval] Using checkpoint: {ckpt_path}")

    # Output directory for eval results
    ckpt_name = Path(ckpt_path).name
    eval_output = Path(
        config["training"].get("log_dir", "experiments/logs/grpo")
    )
    eval_output.mkdir(parents=True, exist_ok=True)
    figures_dir = eval_output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb — save runs inside log_dir instead of random wandb/ folder
    wandb_cfg = config.get("wandb", {})
    import os

    model_short = config["model"]["name"].split("/")[-1]
    os.environ["WANDB_DIR"] = str(eval_output)
    wandb.init(
        project=wandb_cfg.get("project", "grpo-strict-generation"),
        name=f"eval-{model_short}-{eval_run_id}",
        config=config,
        tags=wandb_cfg.get("tags", []) + ["eval", "post-grpo"],
        dir=str(eval_output),
    )

    # Load test dataset
    # Always use the balanced eval dataset (generated if missing)
    from src.evaluation.eval_dataset import load_eval_dataset

    test_ds = load_eval_dataset(config)
    if args.max_samples and args.max_samples < len(test_ds):
        # Stratified sampling to keep difficulty distribution balanced
        import random

        indices_by_diff: dict[str, list[int]] = {}
        for i, d in enumerate(test_ds["difficulty"]):
            indices_by_diff.setdefault(d, []).append(i)

        n_cats = len(indices_by_diff)
        per_cat = args.max_samples // n_cats
        remainder = args.max_samples - per_cat * n_cats

        rng = random.Random(42)
        selected: list[int] = []
        for j, (cat, idxs) in enumerate(
            sorted(indices_by_diff.items())
        ):
            n = per_cat + (1 if j < remainder else 0)
            rng.shuffle(idxs)
            selected.extend(idxs[:n])
        selected.sort()
        test_ds = test_ds.select(selected)

    gen_config = {
        "max_new_tokens": config["grpo"].get(
            "max_completion_length", 512
        ),
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    }
    eval_batch_size = config.get("evaluation", {}).get(
        "batch_size", 8
    )

    # We need a tokenizer for formatting — try loading from the checkpoint
    # (which always includes the tokenizer), falling back to the model name
    # on HuggingFace. This avoids gated-repo errors (e.g. Gemma 2).
    tokenizer_source = config["model"]["name"]
    if (
        ckpt_path
        and (Path(ckpt_path) / "tokenizer_config.json").exists()
    ):
        tokenizer_source = ckpt_path
    tokenizer = load_tokenizer(tokenizer_source)
    prompts = [
        format_prompt_for_model(test_ds[i], tokenizer)
        for i in range(len(test_ds))
    ]
    difficulties = list(test_ds["difficulty"])

    # Register schema metadata for precise reward computation during eval
    if "schema_meta" in test_ds.column_names:
        from src.training.rewards import register_schema_metadata

        register_schema_metadata(
            prompts, list(test_ds["schema_meta"])
        )

    # ── Discover models to evaluate ─────────────────────────────────────
    # In curriculum mode, evaluate all stage_end models; otherwise just the
    # single checkpoint.
    eval_targets: list[tuple[str, str, bool]] = []
    # Each entry: (label, path, is_checkpoint)

    if args.curriculum:
        stages_dir = Path(output_dir) / "stages"
        if not stages_dir.exists():
            print(
                f"No stages/ directory found in {output_dir}. "
                "Run curriculum training first."
            )
            wandb.finish()
            return
        stage_dirs = sorted(
            [d for d in stages_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        if not stage_dirs:
            print(f"No stage directories found in {stages_dir}")
            wandb.finish()
            return
        for sd in stage_dirs:
            # Parse label from dir name: "stage_1_format_basics" → "Stage 1: format_basics"
            parts = sd.name.split("_", 2)
            if len(parts) >= 3:
                label = f"Stage {parts[1]}: {parts[2]}"
            else:
                label = sd.name
            eval_targets.append((label, str(sd), True))
        print(
            f"\n[curriculum] Found {len(eval_targets)} stages to evaluate"
        )
        for label, path, _ in eval_targets:
            print(f"  {label}: {path}")

        # Resume support: skip stages already evaluated
        if args.skip_stages > 0:
            skipped = eval_targets[: args.skip_stages]
            eval_targets = eval_targets[args.skip_stages :]
            print(
                f"\n[resume] Skipping {len(skipped)} already-evaluated stages:"
            )
            for label, path, _ in skipped:
                print(f"  (skip) {label}")
            if not eval_targets:
                print("[resume] All stages already evaluated.")
                wandb.finish()
                return
    else:
        eval_targets.append((f"GRPO ({ckpt_name})", ckpt_path, True))

    # ── Evaluate each target ──────────────────────────────────────────────
    # Free the temp tokenizer before loading eval models
    del tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    all_eval_results: list[dict] = []
    all_completions_data: dict[str, list[dict]] = {}
    all_raw_completions: dict[str, list[str]] = {}

    # ── Baseline evaluation (before stages) ───────────────────────────
    baseline_metrics = None

    if args.compare:
        baseline_path = Path(args.baseline_results)
        if baseline_path.exists():
            print(
                f"\nLoading baseline results from {baseline_path}..."
            )
            baseline_data = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            baseline_metrics = baseline_data["detailed_metrics"]
            print(
                f"  Baseline Pass@1: {baseline_metrics['overall_pass_rate']:.4f}"
            )
        else:
            # Run baseline evaluation
            print(f"\n{'='*50}")
            print("Running baseline evaluation...")
            print(f"{'='*50}")
            baseline_metrics, baseline_completions = _evaluate_model(
                config,
                config["model"]["name"],
                prompts,
                difficulties,
                gen_config,
            )
            print(
                f"\nBaseline Pass@1: {baseline_metrics['overall_pass_rate']:.4f}"
            )
            print("Per-category:")
            for cat, stats in baseline_metrics[
                "per_category"
            ].items():
                print(
                    f"  {cat}: {stats['pass_rate']:.4f} "
                    f"({stats['valid']}/{stats['total']})"
                )
            # Save baseline results next to eval results
            bl_results = {
                "model": config["model"]["name"],
                "detailed_metrics": baseline_metrics,
            }
            bl_path = Path(args.baseline_results)
            bl_path.write_text(
                json.dumps(bl_results, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"Baseline results saved to {bl_path}")

            # Save baseline completions with validation results
            bl_compl_data = []
            for p, d, c in zip(
                prompts, difficulties, baseline_completions
            ):
                valid, error = check_syntax(c)
                entry = {
                    "prompt": p,
                    "difficulty": d,
                    "completion": c,
                    "valid": valid,
                }
                if not valid:
                    entry["error"] = error
                bl_compl_data.append(entry)
            bl_compl_path = eval_output / "completions_baseline.json"
            bl_compl_path.write_text(
                json.dumps(
                    bl_compl_data, indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            print(f"Baseline completions saved to {bl_compl_path}")
            all_completions_data["Baseline"] = bl_compl_data
            all_raw_completions["Baseline"] = baseline_completions

    for label, model_path, is_ckpt in eval_targets:
        print(f"\n{'='*50}")
        print(f"Evaluating: {label}")
        print(f"{'='*50}")

        metrics, completions = _evaluate_model(
            config,
            model_path,
            prompts,
            difficulties,
            gen_config,
            is_checkpoint=is_ckpt,
            batch_size=eval_batch_size,
        )

        print(f"\n{label} Pass@1: {metrics['overall_pass_rate']:.4f}")
        print("Per-category:")
        for cat, stats in metrics["per_category"].items():
            print(
                f"  {cat}: {stats['pass_rate']:.4f} "
                f"({stats['valid']}/{stats['total']})"
            )

        all_eval_results.append(
            {"label": label, "path": model_path, "metrics": metrics}
        )

        # Save individual results
        safe_name = _safe_filename(label)
        result_json = {
            "checkpoint": model_path,
            "label": label,
            "metrics": metrics,
        }
        results_path = eval_output / f"eval_{safe_name}.json"
        results_path.write_text(
            json.dumps(result_json, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Results saved to {results_path}")

        # Print a few completion samples for quick inspection
        _print_eval_samples(
            prompts,
            completions,
            difficulties,
            n=5,
            raw_prompts=prompts,
        )
        completions_data = []
        for p, d, c in zip(prompts, difficulties, completions):
            valid, error = check_syntax(c)
            entry = {
                "prompt": p,
                "difficulty": d,
                "completion": c,
                "valid": valid,
            }
            if not valid:
                entry["error"] = error
            completions_data.append(entry)
        compl_path = eval_output / f"completions_{safe_name}.json"
        compl_path.write_text(
            json.dumps(
                completions_data, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        print(f"Completions saved to {compl_path}")
        all_completions_data[label] = completions_data
        all_raw_completions[label] = completions

        # Per-model figures
        fig_path = str(figures_dir / f"pass_rates_{safe_name}.png")
        plot_per_category_breakdown(
            metrics,
            title=f"Pass Rate — {label}",
            output_path=fig_path,
        )
        wandb.log(
            {f"eval/{safe_name}_pass_rates": wandb.Image(fig_path)}
        )

        err_fig_path = str(figures_dir / f"errors_{safe_name}.png")
        plot_completions_error_breakdown(
            completions_data,
            title=f"Error Breakdown — {label}",
            output_path=err_fig_path,
        )
        wandb.log(
            {f"eval/{safe_name}_errors": wandb.Image(err_fig_path)}
        )

        len_fig_path = str(figures_dir / f"lengths_{safe_name}.png")
        plot_completion_length_distribution(
            completions_data,
            title=f"Completion Lengths — {label}",
            output_path=len_fig_path,
        )
        wandb.log(
            {f"eval/{safe_name}_lengths": wandb.Image(len_fig_path)}
        )

    # Use the last evaluated model as "grpo_metrics" for backward compat
    grpo_metrics = all_eval_results[-1]["metrics"]

    # ── Baseline vs GRPO comparison ─────────────────────────────────────
    if baseline_metrics:
        model_name = config["model"]["name"].split("/")[-1]

        # ── Standard 2-way comparison (baseline vs final) ─────────────
        # In curriculum mode, individual baseline-vs-stage charts are
        # generated below — skip the redundant overall chart.
        print(f"\n{'='*50}")
        print("Comparison: Baseline vs GRPO (final stage)")
        print(f"{'='*50}")
        print(
            f"  Baseline Pass@1: {baseline_metrics['overall_pass_rate']:.4f}"
        )
        print(
            f"  GRPO Pass@1:     {grpo_metrics['overall_pass_rate']:.4f}"
        )
        delta = (
            grpo_metrics["overall_pass_rate"]
            - baseline_metrics["overall_pass_rate"]
        )
        print(f"  Delta:           {delta:+.4f}")

        if not args.curriculum:
            comp_fig_path = str(
                figures_dir / "baseline_vs_grpo_comparison.png"
            )
            plot_baseline_vs_grpo_comparison(
                baseline_metrics=baseline_metrics,
                grpo_metrics=grpo_metrics,
                model_name=model_name,
                output_path=comp_fig_path,
            )
            wandb.log(
                {"eval/baseline_vs_grpo": wandb.Image(comp_fig_path)}
            )

        # Save comparison JSON
        comparison: dict = {
            "baseline_pass_rate": baseline_metrics[
                "overall_pass_rate"
            ],
            "grpo_pass_rate": grpo_metrics["overall_pass_rate"],
            "delta": delta,
            "baseline_per_category": baseline_metrics["per_category"],
            "grpo_per_category": grpo_metrics["per_category"],
        }

        # ── Curriculum progression chart (baseline + all stages) ──────
        if args.curriculum and len(all_eval_results) > 1:
            print(f"\n{'='*50}")
            print("Curriculum progression: Baseline → all stages")
            print(f"{'='*50}")

            stage_results = [
                {"label": "Baseline", "metrics": baseline_metrics}
            ] + [
                {"label": r["label"], "metrics": r["metrics"]}
                for r in all_eval_results
            ]

            for r in stage_results:
                print(
                    f"  {r['label']}: "
                    f"Pass@1={r['metrics']['overall_pass_rate']:.4f}"
                )

            # Individual baseline-vs-stage comparison charts
            for r in all_eval_results:
                stage_safe = _safe_filename(r["label"])
                stage_comp_path = str(
                    figures_dir / f"baseline_vs_{stage_safe}.png"
                )
                plot_baseline_vs_grpo_comparison(
                    baseline_metrics=baseline_metrics,
                    grpo_metrics=r["metrics"],
                    model_name=f"{model_name} — {r['label']}",
                    output_path=stage_comp_path,
                )
                wandb.log(
                    {
                        f"eval/baseline_vs_{stage_safe}": wandb.Image(
                            stage_comp_path
                        )
                    }
                )

            # Unified curriculum progression chart
            curric_fig_path = str(
                figures_dir / "curriculum_progression.png"
            )
            plot_curriculum_progression(
                stage_results=stage_results,
                model_name=model_name,
                output_path=curric_fig_path,
            )
            wandb.log(
                {
                    "eval/curriculum_progression": wandb.Image(
                        curric_fig_path
                    )
                }
            )

            # Error evolution across stages
            stage_completions_list: list[tuple[str, list[dict]]] = []
            if "Baseline" in all_completions_data:
                stage_completions_list.append(
                    ("Baseline", all_completions_data["Baseline"])
                )
            for r in all_eval_results:
                if r["label"] in all_completions_data:
                    stage_completions_list.append(
                        (r["label"], all_completions_data[r["label"]])
                    )
            if len(stage_completions_list) >= 2:
                evo_fig_path = str(
                    figures_dir / "error_evolution.png"
                )
                plot_error_evolution(
                    stage_completions_list,
                    model_name=model_name,
                    output_path=evo_fig_path,
                )
                wandb.log(
                    {
                        "eval/error_evolution": wandb.Image(
                            evo_fig_path
                        )
                    }
                )

                # Stage × Difficulty heatmap
                heatmap_path = str(
                    figures_dir / "stage_difficulty_heatmap.png"
                )
                plot_stage_difficulty_heatmap(
                    stage_completions_list,
                    model_name=model_name,
                    output_path=heatmap_path,
                )
                wandb.log(
                    {
                        "eval/stage_difficulty_heatmap": wandb.Image(
                            heatmap_path
                        )
                    }
                )

            # Extend comparison JSON with per-stage data
            comparison["curriculum_stages"] = [
                {
                    "label": r["label"],
                    "pass_rate": r["metrics"]["overall_pass_rate"],
                    "per_category": r["metrics"]["per_category"],
                }
                for r in stage_results
            ]

        comp_path = eval_output / "comparison.json"
        comp_path.write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        print(f"Comparison saved to {comp_path}")

        # Rescued vs Regressed analysis (baseline vs final GRPO)
        final_label = all_eval_results[-1]["label"]
        if (
            "Baseline" in all_completions_data
            and final_label in all_completions_data
        ):
            rr_path = str(figures_dir / "rescued_vs_regressed.png")
            plot_rescued_vs_regressed(
                all_completions_data["Baseline"],
                all_completions_data[final_label],
                model_name=model_name,
                output_path=rr_path,
            )
            wandb.log(
                {"eval/rescued_vs_regressed": wandb.Image(rr_path)}
            )

    # ── Reward component breakdown chart ──────────────────────────────────
    # Build breakdown for every evaluated model (and baseline if available).
    reward_weights_cfg = config.get("grpo", {}).get("reward", {})
    rw_map = {
        "format": reward_weights_cfg.get("weight_format", 0.20),
        "validity": reward_weights_cfg.get("weight_validity", 0.35),
        "schema": reward_weights_cfg.get("weight_schema", 0.35),
        "reasoning": reward_weights_cfg.get("weight_reasoning", 0.10),
        "truncation": reward_weights_cfg.get(
            "weight_truncation", 0.0
        ),
        "repetition": reward_weights_cfg.get(
            "weight_repetition", 0.0
        ),
        "strictness": reward_weights_cfg.get(
            "weight_strictness", 0.0
        ),
    }
    # Only include components with non-zero weight in the chart
    active_rw = {k: v for k, v in rw_map.items() if v > 0}

    stage_breakdowns: list[dict] = []
    bd_order: list[str] = []

    if "Baseline" in all_raw_completions:
        bd_order.append("Baseline")
    for r in all_eval_results:
        bd_order.append(r["label"])

    for lbl in bd_order:
        if lbl not in all_raw_completions:
            continue
        comps = all_raw_completions[lbl]
        breakdown = compute_reward_breakdown(comps, prompts, prompts)
        # Filter to active components only
        filtered = {
            k: v for k, v in breakdown.items() if k in active_rw
        }
        stage_breakdowns.append({"label": lbl, "scores": filtered})

    if stage_breakdowns:
        model_name_bd = config["model"]["name"].split("/")[-1]
        bd_fig_path = str(figures_dir / "reward_breakdown.png")
        plot_reward_breakdown(
            stage_breakdowns,
            reward_weights=active_rw,
            model_name=model_name_bd,
            output_path=bd_fig_path,
        )
        wandb.log({"eval/reward_breakdown": wandb.Image(bd_fig_path)})

    wandb.finish()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
