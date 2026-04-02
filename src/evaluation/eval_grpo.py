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
from src.models.model_loader import load_model_and_tokenizer
from src.utils.config import load_config
from src.utils.metrics import compute_detailed_metrics
from src.utils.visualization import (
    plot_baseline_vs_grpo_comparison,
    plot_curriculum_progression,
    plot_per_category_breakdown,
)

load_dotenv()


def _evaluate_model(
    config: dict[str, Any],
    model_path: str,
    test_ds: Any,
    prompts: list[str],
    difficulties: list[str],
    gen_config: dict[str, Any],
    is_checkpoint: bool = False,
) -> dict[str, Any]:
    """Load a model from path, generate completions, compute metrics.

    Args:
        is_checkpoint: If True, treat model_path as a PEFT/LoRA checkpoint
                       and load adapters on top of the base model.
    """
    from pathlib import Path as _Path

    model_cfg = config["model"]

    if (
        is_checkpoint
        and (_Path(model_path) / "adapter_config.json").exists()
    ):
        # PEFT checkpoint — load base model + merge LoRA adapters
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
        # Full model or base model (baseline)
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
        batch_size=4,
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

    return metrics


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
    args = parser.parse_args()

    config = load_config(args.config)

    # Derive baseline results path from config log_dir if not explicitly set
    if args.baseline_results is None:
        log_dir = config["training"].get(
            "log_dir", "experiments/logs/grpo"
        )
        args.baseline_results = str(
            Path(log_dir) / "baseline_results.json"
        )

    # Detect curriculum from config (curriculum.enabled in YAML)
    curriculum_cfg = config.get("curriculum", {})
    is_curriculum = curriculum_cfg.get("enabled", False)

    # --curriculum flag follows the config by default; explicit flag overrides
    if is_curriculum:
        args.curriculum = True

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

    os.environ["WANDB_DIR"] = str(eval_output)
    wandb.init(
        project=wandb_cfg.get("project", "grpo-strict-generation"),
        name=f"eval-grpo-{ckpt_name}",
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

    # We need a tokenizer for formatting — load temporarily
    temp_config = {
        "model": {
            **config["model"],
            "name": ckpt_path,
            "fast_inference": False,
            "use_unsloth": False,
        }
    }
    _, tokenizer = load_model_and_tokenizer(temp_config)
    prompts = [
        format_prompt_for_model(test_ds[i], tokenizer)
        for i in range(len(test_ds))
    ]
    difficulties = list(test_ds["difficulty"])

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
    else:
        eval_targets.append((f"GRPO ({ckpt_name})", ckpt_path, True))

    # ── Evaluate each target ──────────────────────────────────────────────
    # Free the temp tokenizer before loading eval models
    del tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    all_eval_results: list[dict] = []

    for label, model_path, is_ckpt in eval_targets:
        print(f"\n{'='*50}")
        print(f"Evaluating: {label}")
        print(f"{'='*50}")

        metrics = _evaluate_model(
            config,
            model_path,
            test_ds,
            prompts,
            difficulties,
            gen_config,
            is_checkpoint=is_ckpt,
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
        safe_name = Path(model_path).name
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

        # Per-model figure
        fig_path = str(figures_dir / f"pass_rates_{safe_name}.png")
        plot_per_category_breakdown(
            metrics,
            title=f"Pass Rate — {label}",
            output_path=fig_path,
        )
        wandb.log(
            {f"eval/{safe_name}_pass_rates": wandb.Image(fig_path)}
        )

    # Use the last evaluated model as "grpo_metrics" for backward compat
    grpo_metrics = all_eval_results[-1]["metrics"]

    # ── Baseline comparison ───────────────────────────────────────────────
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
        else:
            # Run baseline evaluation
            print(f"\n{'='*50}")
            print("Running baseline evaluation...")
            print(f"{'='*50}")
            baseline_metrics = _evaluate_model(
                config,
                config["model"]["name"],
                test_ds,
                prompts,
                difficulties,
                gen_config,
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
                safe_name = Path(r["path"]).name
                stage_comp_path = str(
                    figures_dir / f"baseline_vs_{safe_name}.png"
                )
                plot_baseline_vs_grpo_comparison(
                    baseline_metrics=baseline_metrics,
                    grpo_metrics=r["metrics"],
                    model_name=f"{model_name} — {r['label']}",
                    output_path=stage_comp_path,
                )
                wandb.log(
                    {
                        f"eval/baseline_vs_{safe_name}": wandb.Image(
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

    wandb.finish()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
