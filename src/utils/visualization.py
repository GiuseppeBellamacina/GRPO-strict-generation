"""Visualization utilities for training curves, comparison charts, and error analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_per_category_breakdown(
    detailed_metrics: dict[str, Any],
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
    plt.xticks(rotation=0, ha="center")

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
        baseline_metrics["per_category"]
        .get(c, {})
        .get("pass_rate", 0.0)
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
    bars_b = ax.bar(
        x - width / 2,
        b_values,
        width,
        label="Baseline",
        color="#4C72B0",
        alpha=0.85,
    )
    bars_g = ax.bar(
        x + width / 2,
        g_values,
        width,
        label="Post-GRPO",
        color="#DD8452",
        alpha=0.85,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Pass@1")
    ax.set_title("Pass Rate per Category")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.legend()
    ax.bar_label(bars_b, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars_g, fmt="%.2f", padding=3, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)

    # ── Delta bar: improvement per category ──────────────────────────────────
    ax2 = axes[1]
    bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    bars_d = ax2.bar(
        x, deltas, width * 1.8, color=bar_colors, alpha=0.85
    )
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Δ Pass@1 (GRPO − Baseline)")
    ax2.set_title("Delta per Category")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, ha="center")
    ax2.bar_label(bars_d, fmt="%+.3f", padding=3, fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_curriculum_progression(
    stage_results: list[dict],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/curriculum_progression.png",
) -> None:
    """Multi-stage curriculum comparison: baseline + each stage side by side.

    Args:
        stage_results: List of dicts, each with keys:
            - ``"label"``: display name (e.g. "Baseline", "Stage 1: format_basics")
            - ``"metrics"``: output of ``compute_detailed_metrics``
        model_name: Short model name for the figure title.
        output_path: Where to save the figure.
    """
    if len(stage_results) < 2:
        print("Need at least 2 results (baseline + 1 stage) to plot.")
        return

    sns.set_theme(style="whitegrid")

    # Collect categories from all results
    all_cats: set[str] = set()
    for r in stage_results:
        all_cats.update(r["metrics"].get("per_category", {}).keys())
    all_cats_sorted = sorted(all_cats)
    labels = ["overall"] + all_cats_sorted

    n_stages = len(stage_results)
    x = np.arange(len(labels))
    width = 0.75 / n_stages

    palette = sns.color_palette("husl", n_stages)

    # ── Figure with 2 subplots: absolute pass rates + deltas vs baseline ──
    fig, axes = plt.subplots(
        1, 2, figsize=(max(14, 4 + n_stages * 2), 6)
    )
    suptitle = "Curriculum Progression"
    if model_name:
        suptitle += f" — {model_name}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    # Left: grouped bar of pass rates
    ax = axes[0]
    all_bars = []
    for i, r in enumerate(stage_results):
        m = r["metrics"]
        values = [m["overall_pass_rate"]] + [
            m["per_category"].get(c, {}).get("pass_rate", 0.0)
            for c in all_cats_sorted
        ]
        offset = (i - n_stages / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=r["label"],
            color=palette[i],
            alpha=0.85,
        )
        all_bars.append((bars, values))

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Pass@1")
    ax.set_title("Pass Rate per Category")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=n_stages,
    )
    for bars, values in all_bars:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)

    # Right: delta bars (each stage vs baseline)
    ax2 = axes[1]
    baseline_m = stage_results[0]["metrics"]
    b_values = [baseline_m["overall_pass_rate"]] + [
        baseline_m["per_category"].get(c, {}).get("pass_rate", 0.0)
        for c in all_cats_sorted
    ]

    for i, r in enumerate(stage_results[1:], start=1):
        m = r["metrics"]
        s_values = [m["overall_pass_rate"]] + [
            m["per_category"].get(c, {}).get("pass_rate", 0.0)
            for c in all_cats_sorted
        ]
        deltas = [s - b for s, b in zip(s_values, b_values)]
        offset = (i - 1 - (n_stages - 1) / 2 + 0.5) * width
        bars_d = ax2.bar(
            x + offset,
            deltas,
            width,
            label=r["label"],
            color=palette[i],
            alpha=0.85,
        )
        ax2.bar_label(bars_d, fmt="%+.3f", padding=2, fontsize=7)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Δ Pass@1 (vs Baseline)")
    ax2.set_title("Improvement over Baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, ha="center")
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_completions_error_breakdown(
    completions_data: list[dict],
    title: str = "Completion Validity",
    output_path: str = "experiments/logs/figures/error_breakdown.png",
) -> None:
    """Pie chart of error types + per-difficulty error type distribution.

    Args:
        completions_data: List of dicts with keys: valid, error (optional), difficulty.
        title: Figure suptitle.
        output_path: Where to save the figure.
    """
    from collections import Counter

    sns.set_theme(style="whitegrid")

    # Classify each completion
    labels_list: list[str] = []
    for entry in completions_data:
        if entry["valid"]:
            labels_list.append("valid")
        else:
            err = entry.get("error", "unknown")
            if err.startswith("json_error"):
                labels_list.append("json_error")
            else:
                labels_list.append(err)

    counts = Counter(labels_list)
    total = len(labels_list)

    # Per-difficulty error breakdown
    diff_errors: dict[str, Counter] = {}
    for entry, lbl in zip(completions_data, labels_list):
        d = entry["difficulty"]
        if d not in diff_errors:
            diff_errors[d] = Counter()
        diff_errors[d][lbl] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: pie chart
    ax = axes[0]
    pie_labels = list(counts.keys())
    pie_values = list(counts.values())
    color_map = {
        "valid": "#2ca02c",
        "no_code_block": "#d62728",
        "json_error": "#ff7f0e",
    }
    colors = [color_map.get(lbl, "#9467bd") for lbl in pie_labels]
    wedges, _, autotexts = ax.pie(
        pie_values,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 5 else "",
        startangle=90,
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(
        wedges,
        [
            f"{lbl} ({v}/{total} — {v/total*100:.1f}%)"
            for lbl, v in zip(pie_labels, pie_values)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        ncol=2,
    )
    ax.set_title("Error Type Distribution")

    # Right: stacked bar — error types per difficulty
    ax2 = axes[1]
    diffs_sorted = sorted(diff_errors.keys())
    all_types = sorted(set(labels_list))
    x = np.arange(len(diffs_sorted))
    bottom = np.zeros(len(diffs_sorted))

    for err_type in all_types:
        values = np.array(
            [diff_errors[d].get(err_type, 0) for d in diffs_sorted],
            dtype=float,
        )
        color = color_map.get(err_type, "#9467bd")
        ax2.bar(
            x,
            values,
            bottom=bottom,
            label=err_type,
            color=color,
            alpha=0.85,
        )
        for i, v in enumerate(values):
            if v > 0:
                ax2.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        bottom += values

    ax2.set_xticks(x)
    ax2.set_xticklabels(diffs_sorted)
    ax2.set_ylabel("Count")
    ax2.set_title("Error Types by Difficulty")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_evolution(
    stage_completions: list[tuple[str, list[dict]]],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/error_evolution.png",
) -> None:
    """Stacked bar chart showing how error types evolve across curriculum stages.

    Args:
        stage_completions: List of (label, completions_data) tuples, one per stage.
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    sns.set_theme(style="whitegrid")

    # Classify errors per stage
    all_error_types: set[str] = {"valid"}
    stage_counts: list[dict[str, int]] = []
    stage_labels: list[str] = []
    for label, comps in stage_completions:
        stage_labels.append(label)
        counts: dict[str, int] = {"valid": 0}
        for entry in comps:
            if entry["valid"]:
                counts["valid"] = counts.get("valid", 0) + 1
            else:
                err = entry.get("error", "unknown")
                if err.startswith("json_error"):
                    err = "json_error"
                counts[err] = counts.get(err, 0) + 1
                all_error_types.add(err)
        stage_counts.append(counts)

    error_types = sorted(all_error_types - {"valid"})
    categories = ["valid"] + error_types

    x = np.arange(len(stage_labels))
    fig, ax = plt.subplots(
        figsize=(max(8, len(stage_labels) * 2.5), 5)
    )

    color_map = {
        "valid": "#2ca02c",
        "no_code_block": "#d62728",
        "json_error": "#ff7f0e",
    }
    bottom = np.zeros(len(stage_labels))

    for cat in categories:
        values = []
        for sc in stage_counts:
            total = sum(sc.values())
            values.append(sc.get(cat, 0) / max(total, 1) * 100)
        values_arr = np.array(values)
        color = color_map.get(cat, "#9467bd")
        ax.bar(
            x,
            values_arr,
            bottom=bottom,
            label=cat,
            color=color,
            alpha=0.85,
        )
        for i, v in enumerate(values_arr):
            if v > 5:
                ax.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
        bottom += values_arr

    suptitle = "Error Evolution Across Stages"
    if model_name:
        suptitle += f" — {model_name}"
    ax.set_title(suptitle, fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=0, ha="center")
    ax.set_ylim(0, 105)
    n_cats = len(categories)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        fontsize=8,
        ncol=n_cats,
    )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_completion_length_distribution(
    completions_data: list[dict],
    title: str = "Completion Length Distribution",
    output_path: str = "experiments/logs/figures/completion_lengths.png",
) -> None:
    """Histogram of completion lengths (chars), split by valid vs invalid.

    Args:
        completions_data: List of dicts with keys: completion, valid.
        title: Figure title.
        output_path: Where to save the figure.
    """
    sns.set_theme(style="whitegrid")

    valid_lens = [
        len(e["completion"]) for e in completions_data if e["valid"]
    ]
    invalid_lens = [
        len(e["completion"])
        for e in completions_data
        if not e["valid"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(
        0,
        max((valid_lens or [0]) + (invalid_lens or [0])) * 1.05,
        30,
    )
    if valid_lens:
        ax.hist(
            valid_lens,
            bins=bins,
            alpha=0.7,
            label=f"Valid (n={len(valid_lens)})",
            color="#2ca02c",
        )
    if invalid_lens:
        ax.hist(
            invalid_lens,
            bins=bins,
            alpha=0.7,
            label=f"Invalid (n={len(invalid_lens)})",
            color="#d62728",
        )

    ax.set_xlabel("Completion Length (chars)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_stage_difficulty_heatmap(
    stage_completions: list[tuple[str, list[dict]]],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/stage_difficulty_heatmap.png",
) -> None:
    """Heatmap of pass rates: rows = stages, columns = difficulty levels.

    Args:
        stage_completions: List of (label, completions_data) tuples.
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    sns.set_theme(style="whitegrid")

    # Collect all difficulties
    all_diffs: set[str] = set()
    for _, comps in stage_completions:
        for e in comps:
            all_diffs.add(e["difficulty"])
    diffs_sorted = sorted(all_diffs)

    stage_labels = [label for label, _ in stage_completions]
    matrix = np.zeros((len(stage_labels), len(diffs_sorted)))

    for i, (_, comps) in enumerate(stage_completions):
        diff_stats: dict[str, dict[str, int]] = {}
        for e in comps:
            d = e["difficulty"]
            if d not in diff_stats:
                diff_stats[d] = {"valid": 0, "total": 0}
            diff_stats[d]["total"] += 1
            if e["valid"]:
                diff_stats[d]["valid"] += 1
        for j, d in enumerate(diffs_sorted):
            s = diff_stats.get(d, {"valid": 0, "total": 1})
            matrix[i, j] = s["valid"] / max(s["total"], 1)

    fig, ax = plt.subplots(
        figsize=(
            max(6, len(diffs_sorted) * 2),
            max(4, len(stage_labels) * 0.8),
        )
    )
    im = ax.imshow(
        matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"
    )
    ax.grid(False)

    ax.set_xticks(range(len(diffs_sorted)))
    ax.set_xticklabels(diffs_sorted)
    ax.set_yticks(range(len(stage_labels)))
    ax.set_yticklabels(stage_labels)

    # Annotate cells with pass rate values
    for i in range(len(stage_labels)):
        for j in range(len(diffs_sorted)):
            val = matrix[i, j]
            text_color = (
                "white" if val < 0.4 or val > 0.8 else "black"
            )
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, label="Pass@1")
    suptitle = "Pass Rate: Stage × Difficulty"
    if model_name:
        suptitle += f" — {model_name}"
    ax.set_title(suptitle, fontsize=13, fontweight="bold")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_reward_breakdown(
    stage_breakdowns: list[dict],
    reward_weights: dict[str, float] | None = None,
    model_name: str = "",
    output_path: str = "experiments/logs/figures/reward_breakdown.png",
) -> None:
    """Stacked bar chart showing weighted reward contributions per stage.

    Each bar is composed of segments (one per reward component).  The height
    of each segment equals ``weight_i * avg_score_i``, i.e. the effective
    contribution of that component to the total weighted reward.  Components
    that can go negative (truncation, repetition, strictness) may produce
    downward segments.

    Args:
        stage_breakdowns: List of dicts, each with:
            - ``"label"`` : stage display name (e.g. "Baseline", "Stage 1: format_basics")
            - ``"scores"`` : dict mapping component name → average raw score
        reward_weights: Optional mapping of component name → weight.  If
            ``None``, all components are shown with equal weight (1.0).
        model_name: Short model name for the figure title.
        output_path: Where to save the figure.
    """
    sns.set_theme(style="whitegrid")

    if not stage_breakdowns:
        print("No reward breakdown data to plot.")
        return

    # Component ordering and colors — keep consistent across charts
    component_order = [
        "format",
        "validity",
        "schema",
        "reasoning",
        "truncation",
        "repetition",
        "strictness",
    ]
    component_colors = {
        "format": "#4C72B0",
        "validity": "#55A868",
        "schema": "#C44E52",
        "reasoning": "#DD8452",
        "truncation": "#8172B2",
        "repetition": "#CCB974",
        "strictness": "#64B5CD",
    }

    # Determine which components actually have data
    all_components = set()
    for sb in stage_breakdowns:
        all_components.update(sb["scores"].keys())
    components = [c for c in component_order if c in all_components]

    if reward_weights is None:
        reward_weights = {c: 1.0 for c in components}

    stage_labels = [sb["label"] for sb in stage_breakdowns]
    n_stages = len(stage_labels)

    # Compute weighted contributions: weight_i * avg_score_i
    weighted: dict[str, list[float]] = {c: [] for c in components}
    totals: list[float] = []
    for sb in stage_breakdowns:
        total = 0.0
        for c in components:
            w = reward_weights.get(c, 0.0)
            val = sb["scores"].get(c, 0.0) * w
            weighted[c].append(val)
            total += val
        totals.append(total)

    x = np.arange(n_stages)
    fig, ax = plt.subplots(figsize=(max(8, n_stages * 2.2), 6))

    # Split positive and negative contributions for proper stacking
    pos_bottom = np.zeros(n_stages)
    neg_bottom = np.zeros(n_stages)

    for c in components:
        values = np.array(weighted[c])
        pos_vals = np.where(values >= 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)
        color = component_colors.get(c, "#999999")

        w_val = reward_weights.get(c, 0.0)
        label = f"{c} (w={w_val:.2f})" if w_val != 1.0 else c

        if np.any(pos_vals > 0):
            ax.bar(
                x,
                pos_vals,
                bottom=pos_bottom,
                label=label,
                color=color,
                alpha=0.85,
                width=0.6,
            )
            pos_bottom += pos_vals

        if np.any(neg_vals < 0):
            ax.bar(
                x,
                neg_vals,
                bottom=neg_bottom,
                label=label if not np.any(pos_vals > 0) else "",
                color=color,
                alpha=0.85,
                width=0.6,
                hatch="//",
            )
            neg_bottom += neg_vals

    # Total reward marker on top of each bar
    for i, total in enumerate(totals):
        ax.text(
            x[i],
            max(pos_bottom[i], 0) + 0.02,
            f"{total:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=0, ha="center")
    ax.set_ylabel("Weighted Reward Contribution")

    suptitle = "Reward Component Breakdown"
    if model_name:
        suptitle += f" \u2014 {model_name}"
    ax.set_title(suptitle, fontsize=13, fontweight="bold")

    # De-duplicate legend entries (avoid double labels from pos/neg split)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    unique_handles = []
    unique_labels = []
    for h, lbl in zip(handles, labels):
        if lbl and lbl not in seen:
            seen[lbl] = True
            unique_handles.append(h)
            unique_labels.append(lbl)
    ax.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
        ncol=min(len(unique_labels), 6),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_rescued_vs_regressed(
    baseline_completions: list[dict],
    grpo_completions: list[dict],
    model_name: str = "",
    output_path: str = "experiments/logs/figures/rescued_vs_regressed.png",
) -> None:
    """Bar chart showing per-prompt outcome changes: rescued, regressed, both-pass, both-fail.

    Args:
        baseline_completions: List of dicts with valid, difficulty (baseline model).
        grpo_completions: List of dicts with valid, difficulty (GRPO model).
        model_name: Short model name for the title.
        output_path: Where to save the figure.
    """
    sns.set_theme(style="whitegrid")

    categories = {
        "rescued": 0,
        "regressed": 0,
        "both_pass": 0,
        "both_fail": 0,
    }
    diff_cats: dict[str, dict[str, int]] = {}

    for bl, gr in zip(baseline_completions, grpo_completions):
        bl_ok = bl["valid"]
        gr_ok = gr["valid"]
        diff = bl["difficulty"]

        if diff not in diff_cats:
            diff_cats[diff] = {
                "rescued": 0,
                "regressed": 0,
                "both_pass": 0,
                "both_fail": 0,
            }

        if not bl_ok and gr_ok:
            cat = "rescued"
        elif bl_ok and not gr_ok:
            cat = "regressed"
        elif bl_ok and gr_ok:
            cat = "both_pass"
        else:
            cat = "both_fail"

        categories[cat] += 1
        diff_cats[diff][cat] += 1

    total = sum(categories.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    suptitle = "Rescued vs Regressed"
    if model_name:
        suptitle += f" — {model_name}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # Left: overall pie
    ax = axes[0]
    color_map = {
        "rescued": "#2ca02c",
        "regressed": "#d62728",
        "both_pass": "#1f77b4",
        "both_fail": "#7f7f7f",
    }
    pie_labels = [k for k, v in categories.items() if v > 0]
    pie_values = [categories[k] for k in pie_labels]
    pie_colors = [color_map[k] for k in pie_labels]
    wedges, _, autotexts = ax.pie(
        pie_values,
        colors=pie_colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 5 else "",
        startangle=90,
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(
        wedges,
        [
            f"{lbl} ({v}/{total} — {v/total*100:.1f}%)"
            for lbl, v in zip(pie_labels, pie_values)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        ncol=2,
    )
    ax.set_title("Overall")

    # Right: stacked bar per difficulty
    ax2 = axes[1]
    diffs_sorted = sorted(diff_cats.keys())
    x = np.arange(len(diffs_sorted))
    bottom = np.zeros(len(diffs_sorted))
    cat_order = ["both_pass", "rescued", "regressed", "both_fail"]

    for cat in cat_order:
        values = np.array(
            [diff_cats[d].get(cat, 0) for d in diffs_sorted],
            dtype=float,
        )
        if values.sum() > 0:
            ax2.bar(
                x,
                values,
                bottom=bottom,
                label=cat,
                color=color_map[cat],
                alpha=0.85,
            )
            for i, v in enumerate(values):
                if v > 0:
                    ax2.text(
                        x[i],
                        bottom[i] + v / 2,
                        f"{int(v)}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            bottom += values

    ax2.set_xticks(x)
    ax2.set_xticklabels(diffs_sorted)
    ax2.set_ylabel("Count")
    ax2.set_title("By Difficulty")
    ax2.legend(fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
