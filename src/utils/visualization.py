"""Visualization utilities for training curves, comparison charts, and error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pass_at_k_comparison(
    results: dict[str, dict[str, float]],
    output_path: str = "experiments/logs/figures/pass_at_k_comparison.png",
) -> None:
    """Bar chart comparing Pass@k across models/methods.

    Args:
        results: Dict like {"Baseline 0.5B": {"pass@1": 0.2, ...}, "GRPO 1.5B": {...}}.
        output_path: Where to save the figure.
    """
    sns.set_theme(style="whitegrid")

    models = list(results.keys())
    metrics = sorted({m for r in results.values() for m in r})

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.8 / len(models)

    for i, model_name in enumerate(models):
        values = [results[model_name].get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width,
            label=model_name,
            alpha=0.85,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Pass@k Comparison: Baseline vs GRPO vs SFT")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_category_breakdown(
    detailed_metrics: dict,
    title: str = "Pass Rate by Task Type and Difficulty",
    output_path: str = "experiments/logs/figures/per_category_breakdown.png",
) -> None:
    """Grouped bar chart of pass rates per category (json/simple, json/hard, etc.)."""
    sns.set_theme(style="whitegrid")
    categories = detailed_metrics.get("per_category", {})

    if not categories:
        print("No per_category data to plot.")
        return

    labels = list(categories.keys())
    pass_rates = [categories[k]["pass_rate"] for k in labels]
    totals = [categories[k]["total"] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(labels))
    bars = ax.bar(labels, pass_rates, color=colors, alpha=0.85)

    for bar, rate, total in zip(bars, pass_rates, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.2f}\n(n={total})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Pass Rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.15)
    plt.xticks(rotation=30, ha="right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_distribution(
    detailed_metrics: dict,
    output_path: str = "experiments/logs/figures/error_distribution.png",
) -> None:
    """Horizontal bar chart of error type distribution."""
    sns.set_theme(style="whitegrid")
    errors = detailed_metrics.get("error_distribution", {})

    if not errors:
        print("No errors to plot.")
        return

    # Top 10 errors
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    labels, counts = zip(*sorted_errors)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        labels,
        counts,
        color=sns.color_palette("Reds_r", len(labels)),
        alpha=0.85,
    )
    ax.set_xlabel("Count")
    ax.set_title("Top Error Types")
    ax.invert_yaxis()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_reward_curve(
    log_dir: str,
    output_path: str = "experiments/logs/figures/training_reward_curve.png",
) -> None:
    """Plot reward over training steps from wandb logs.

    Falls back to reading trainer_state.json if wandb events aren't available.
    """
    sns.set_theme(style="whitegrid")
    state_path = Path(log_dir).parent / "trainer_state.json"

    if not state_path.exists():
        # Try looking one level up or in checkpoints
        for candidate in [
            Path(log_dir) / "trainer_state.json",
            Path(log_dir).parent / "checkpoint-latest" / "trainer_state.json",
        ]:
            if candidate.exists():
                state_path = candidate
                break

    if not state_path.exists():
        print(
            f"No trainer_state.json found near {log_dir}. Skipping reward curve."
        )
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])

    steps = [entry["step"] for entry in log_history if "reward" in entry]
    rewards = [entry["reward"] for entry in log_history if "reward" in entry]

    if not steps:
        # Try loss instead
        steps = [entry["step"] for entry in log_history if "loss" in entry]
        rewards = [entry["loss"] for entry in log_history if "loss" in entry]
        ylabel = "Loss"
    else:
        ylabel = "Mean Reward"

    if not steps:
        print("No reward or loss data in training logs.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, linewidth=2, color="#2196F3")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title("Training Progress")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_baseline_vs_grpo_comparison(
    baseline_metrics: dict,
    grpo_metrics: dict,
    model_name: str = "",
    output_path: str = "experiments/logs/figures/baseline_vs_grpo_comparison.png",
) -> None:
    """Side-by-side grouped bar + delta chart: baseline vs post-GRPO pass rates.

    Args:
        baseline_metrics: Output of compute_detailed_metrics for the baseline model.
        grpo_metrics: Output of compute_detailed_metrics for the post-GRPO model.
        model_name: Short model name shown in the figure suptitle.
        output_path: Where to save the figure.
    """
    all_cats = sorted(
        set(
            list(baseline_metrics["per_category"].keys())
            + list(grpo_metrics["per_category"].keys())
        )
    )
    labels = ["overall"] + all_cats
    b_values = [baseline_metrics["overall_pass_rate"]] + [
        baseline_metrics["per_category"].get(c, {}).get("pass_rate", 0.0)
        for c in all_cats
    ]
    g_values = [grpo_metrics["overall_pass_rate"]] + [
        grpo_metrics["per_category"].get(c, {}).get("pass_rate", 0.0)
        for c in all_cats
    ]
    deltas = [g - b for g, b in zip(g_values, b_values)]

    x = np.arange(len(labels))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    suptitle = "Baseline vs Post-GRPO"
    if model_name:
        suptitle += f" — {model_name}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # ── Grouped bar: pass rate per category ──────────────────────────────────
    ax = axes[0]
    bars_b = ax.bar(x - width / 2, b_values, width, label="Baseline", color="#4C72B0", alpha=0.85)
    bars_g = ax.bar(x + width / 2, g_values, width, label="Post-GRPO", color="#DD8452", alpha=0.85)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Pass@1")
    ax.set_title("Pass Rate per Category")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.bar_label(bars_b, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars_g, fmt="%.2f", padding=3, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)

    # ── Delta bar: improvement per category ──────────────────────────────────
    ax2 = axes[1]
    bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    bars_d = ax2.bar(x, deltas, width * 1.8, color=bar_colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Δ Pass@1 (GRPO − Baseline)")
    ax2.set_title("Delta per Category")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.bar_label(bars_d, fmt="%+.3f", padding=3, fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

