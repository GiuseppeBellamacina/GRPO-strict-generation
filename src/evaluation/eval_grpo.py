"""Post-training evaluation — evaluate GRPO-trained model and compare with baseline.

Usage:
    # Evaluate GRPO model only
    python -m src.evaluation --config experiments/configs/grpo_cluster.yaml

    # Evaluate GRPO model + compare with baseline results
    python -m src.evaluation --config experiments/configs/grpo_cluster.yaml --compare

    # Evaluate a specific checkpoint
    python -m src.evaluation --config experiments/configs/grpo_cluster.yaml \
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
from dotenv import load_dotenv

import wandb
from src.datasets.dataloader import (
    format_prompt_for_model,
    load_synthetic_dataset,
)
from src.evaluation.eval_baseline import generate_completions
from src.models.model_loader import load_model_and_tokenizer
from src.utils.config import load_config
from src.utils.metrics import compute_detailed_metrics
from src.utils.visualization import (
    plot_baseline_vs_grpo_comparison,
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
) -> dict[str, Any]:
    """Load a model from path, generate completions, compute metrics."""
    eval_config = {
        "model": {
            **config["model"],
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
    metrics = compute_detailed_metrics(first_completions, difficulties)

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
        "--config", type=str, required=True, help="Path to training config YAML"
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
        "--baseline-results",
        type=str,
        default="experiments/logs/baseline/results.json",
        help="Path to existing baseline results.json (skip re-running baseline)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max test samples to evaluate (default: 200)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine checkpoint path
    output_dir = config["training"]["output_dir"]
    ckpt_path = args.checkpoint or str(Path(output_dir) / "final")
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

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
    print("Loading test dataset...")
    ds = load_synthetic_dataset(
        path=config["dataset"]["path"],
        split="test",
    )
    test_ds = ds["test"]
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))

    gen_config = {
        "max_new_tokens": config["grpo"].get("max_completion_length", 512),
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

    # ── Evaluate GRPO model ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Evaluating GRPO model ({ckpt_name})")
    print(f"{'='*50}")

    # Free the temp tokenizer and reload with the model
    del tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    grpo_metrics = _evaluate_model(
        config, ckpt_path, test_ds, prompts, difficulties, gen_config
    )

    print(f"\nGRPO Pass@1: {grpo_metrics['overall_pass_rate']:.4f}")
    print("\nPer-category:")
    for cat, stats in grpo_metrics["per_category"].items():
        print(
            f"  {cat}: {stats['pass_rate']:.4f} ({stats['valid']}/{stats['total']})"
        )

    # Save GRPO results
    grpo_results = {
        "checkpoint": ckpt_path,
        "metrics": grpo_metrics,
    }
    results_path = eval_output / f"eval_{ckpt_name}.json"
    results_path.write_text(
        json.dumps(grpo_results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to {results_path}")

    # GRPO-only figure
    fig_path = str(figures_dir / f"grpo_{ckpt_name}_pass_rates.png")
    plot_per_category_breakdown(grpo_metrics, output_path=fig_path)
    wandb.log({"eval/grpo_pass_rates": wandb.Image(fig_path)})

    # ── Baseline comparison ───────────────────────────────────────────────
    baseline_metrics = None

    if args.compare:
        baseline_path = Path(args.baseline_results)
        if baseline_path.exists():
            print(f"\nLoading baseline results from {baseline_path}...")
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
            # Save baseline results
            bl_output = Path("experiments/logs/baseline")
            bl_output.mkdir(parents=True, exist_ok=True)
            bl_results = {
                "model": config["model"]["name"],
                "detailed_metrics": baseline_metrics,
            }
            bl_path = bl_output / "results.json"
            bl_path.write_text(
                json.dumps(bl_results, indent=2, default=str), encoding="utf-8"
            )
            print(f"Baseline results saved to {bl_path}")

    if baseline_metrics:
        print(f"\n{'='*50}")
        print("Comparison: Baseline vs GRPO")
        print(f"{'='*50}")
        print(f"  Baseline Pass@1: {baseline_metrics['overall_pass_rate']:.4f}")
        print(f"  GRPO Pass@1:     {grpo_metrics['overall_pass_rate']:.4f}")
        delta = (
            grpo_metrics["overall_pass_rate"]
            - baseline_metrics["overall_pass_rate"]
        )
        print(f"  Delta:           {delta:+.4f}")

        # Comparison figure
        model_name = config["model"]["name"].split("/")[-1]
        comp_fig_path = str(figures_dir / "baseline_vs_grpo_comparison.png")
        plot_baseline_vs_grpo_comparison(
            baseline_metrics=baseline_metrics,
            grpo_metrics=grpo_metrics,
            model_name=model_name,
            output_path=comp_fig_path,
        )
        wandb.log({"eval/baseline_vs_grpo": wandb.Image(comp_fig_path)})

        # Save comparison JSON
        comparison = {
            "baseline_pass_rate": baseline_metrics["overall_pass_rate"],
            "grpo_pass_rate": grpo_metrics["overall_pass_rate"],
            "delta": delta,
            "baseline_per_category": baseline_metrics["per_category"],
            "grpo_per_category": grpo_metrics["per_category"],
        }
        comp_path = eval_output / "comparison.json"
        comp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"Comparison saved to {comp_path}")

    wandb.finish()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
