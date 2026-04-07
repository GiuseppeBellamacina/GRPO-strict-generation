"""Compare evaluation metrics between no-think and think runs.

Loads comparison.json (or individual eval_*.json) from both run variants
and produces side-by-side bar charts + delta analysis.

Usage:
    python -m src.utils.compare_think \
        --no-think experiments/logs/grpo/nothink/curriculum/gemma2-2b \
        --think    experiments/logs/grpo/think/curriculum/gemma2-2b

    # Or compare all 5 models at once (curriculum variant):
    python -m src.utils.compare_think --all --curriculum

    # Standard variant:
    python -m src.utils.compare_think --all --standard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Model registry (maps short name → log dir name) ─────────────────────────
MODELS = {
    "SmolLM2-135M": "smollm2-135m",
    "SmolLM2-360M": "smollm2-360m",
    "TinyLlama-1.1B": "tinyllama-11b",
    "Qwen2.5-0.5B": "qwen25-05b",
    "Gemma-2-2B": "gemma2-2b",
}

LOGS_NOTHINK = Path("experiments/logs/grpo/nothink")
LOGS_THINK = Path("experiments/logs/grpo/think")


def _logs_base(variant: str) -> tuple[Path, Path]:
    """Return (nothink_base, think_base) for the given curriculum variant."""
    return (
        LOGS_NOTHINK / variant,
        LOGS_THINK / variant,
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def _resolve_latest_eval(log_dir: Path) -> Path | None:
    """Find the most recent eval_* subdirectory."""
    if not log_dir.exists():
        return None
    eval_dirs = sorted(
        [
            d
            for d in log_dir.iterdir()
            if d.is_dir() and d.name.startswith("eval_")
        ],
        key=lambda d: d.name,
    )
    return eval_dirs[-1] if eval_dirs else None


def load_comparison(log_dir: Path) -> dict[str, Any] | None:
    """Load comparison.json from the latest eval run."""
    eval_dir = _resolve_latest_eval(log_dir)
    if eval_dir is None:
        return None
    comp = eval_dir / "comparison.json"
    if comp.exists():
        return json.loads(comp.read_text(encoding="utf-8"))
    return None


def load_stage_evals(log_dir: Path) -> list[dict[str, Any]]:
    """Load all eval_stage_*.json files from the latest eval run."""
    eval_dir = _resolve_latest_eval(log_dir)
    if eval_dir is None:
        return []
    results = []
    for f in sorted(eval_dir.glob("eval_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        results.append(data)
    return results


def load_baseline(log_dir: Path) -> dict[str, Any] | None:
    """Load baseline_results.json."""
    bl = log_dir / "baseline_results.json"
    if bl.exists():
        return json.loads(bl.read_text(encoding="utf-8"))
    return None


def extract_metrics_summary(log_dir: Path) -> dict[str, Any] | None:
    """Extract a flat metrics dict from one run variant.

    Returns dict with keys:
      - overall_pass_rate (float)
      - baseline_pass_rate (float)
      - delta (float)
      - per_category: {simple: float, medium: float, hard: float}
      - curriculum_stages: list of {label, pass_rate, per_category}
    """
    comp = load_comparison(log_dir)
    if comp is None:
        return None

    summary: dict[str, Any] = {
        "baseline_pass_rate": comp["baseline_pass_rate"],
        "grpo_pass_rate": comp["grpo_pass_rate"],
        "delta": comp["delta"],
        "grpo_per_category": {
            k: v["pass_rate"]
            for k, v in comp.get("grpo_per_category", {}).items()
        },
        "baseline_per_category": {
            k: v["pass_rate"]
            for k, v in comp.get("baseline_per_category", {}).items()
        },
    }

    if "curriculum_stages" in comp:
        summary["curriculum_stages"] = comp["curriculum_stages"]

    return summary


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_nothink_vs_think(
    nothink: dict[str, Any],
    think: dict[str, Any],
    model_name: str,
    output_path: str | Path,
) -> None:
    """Side-by-side comparison: no-think vs think pass rates + deltas."""
    sns.set_theme(style="whitegrid")

    categories = sorted(
        set(
            list(nothink["grpo_per_category"])
            + list(think["grpo_per_category"])
        )
    )
    labels = ["overall"] + categories

    nt_vals = [nothink["grpo_pass_rate"]] + [
        nothink["grpo_per_category"].get(c, 0.0) for c in categories
    ]
    tk_vals = [think["grpo_pass_rate"]] + [
        think["grpo_per_category"].get(c, 0.0) for c in categories
    ]
    deltas = [t - n for t, n in zip(tk_vals, nt_vals)]

    x = np.arange(len(labels))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"No-Think vs Think — {model_name}",
        fontsize=13,
        fontweight="bold",
    )

    # Left: grouped bars
    ax = axes[0]
    bars_nt = ax.bar(
        x - width / 2,
        nt_vals,
        width,
        label="No-Think",
        color="#4C72B0",
        alpha=0.85,
    )
    bars_tk = ax.bar(
        x + width / 2,
        tk_vals,
        width,
        label="Think",
        color="#DD8452",
        alpha=0.85,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Pass@1")
    ax.set_title("Post-GRPO Pass Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(bars_nt, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars_tk, fmt="%.3f", padding=3, fontsize=8)

    # Right: delta bars
    ax2 = axes[1]
    bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    bars_d = ax2.bar(
        x, deltas, width * 1.8, color=bar_colors, alpha=0.85
    )
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Δ Pass@1 (Think − No-Think)")
    ax2.set_title("Delta per Category")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.bar_label(bars_d, fmt="%+.3f", padding=3, fontsize=8)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_all_models_summary(
    results: dict[str, dict[str, dict[str, Any]]],
    output_path: (
        str | Path
    ) = "experiments/logs/figures/think_vs_nothink_all_models.png",
) -> None:
    """Summary chart: all models, no-think vs think overall pass rate."""
    sns.set_theme(style="whitegrid")

    model_names = list(results.keys())
    nt_rates = []
    tk_rates = []
    for name in model_names:
        nt = results[name].get("nothink")
        tk = results[name].get("think")
        nt_rates.append(nt["grpo_pass_rate"] if nt else 0.0)
        tk_rates.append(tk["grpo_pass_rate"] if tk else 0.0)

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_nt = ax.bar(
        x - width / 2,
        nt_rates,
        width,
        label="No-Think",
        color="#4C72B0",
        alpha=0.85,
    )
    bars_tk = ax.bar(
        x + width / 2,
        tk_rates,
        width,
        label="Think",
        color="#DD8452",
        alpha=0.85,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Overall Pass@1")
    ax.set_title(
        "No-Think vs Think — All Models (Post-GRPO)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.bar_label(bars_nt, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars_tk, fmt="%.3f", padding=3, fontsize=9)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_curriculum_think_comparison(
    nothink: dict[str, Any],
    think: dict[str, Any],
    model_name: str,
    output_path: str | Path,
) -> None:
    """Compare curriculum progression curves: no-think vs think."""
    nt_stages = nothink.get("curriculum_stages", [])
    tk_stages = think.get("curriculum_stages", [])

    if not nt_stages or not tk_stages:
        print(
            f"Skipping curriculum comparison for {model_name} (no stage data)."
        )
        return

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Curriculum Progression — {model_name}",
        fontsize=13,
        fontweight="bold",
    )

    nt_labels = [s["label"] for s in nt_stages]
    nt_rates = [s["pass_rate"] for s in nt_stages]
    tk_labels = [s["label"] for s in tk_stages]
    tk_rates = [s["pass_rate"] for s in tk_stages]

    ax.plot(
        nt_labels,
        nt_rates,
        "o-",
        label="No-Think",
        color="#4C72B0",
        linewidth=2,
    )
    ax.plot(
        tk_labels,
        tk_rates,
        "s--",
        label="Think",
        color="#DD8452",
        linewidth=2,
    )

    for i, (lbl, r) in enumerate(zip(nt_labels, nt_rates)):
        ax.annotate(
            f"{r:.3f}",
            (i, r),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=8,
        )
    for i, (lbl, r) in enumerate(zip(tk_labels, tk_rates)):
        ax.annotate(
            f"{r:.3f}",
            (i, r),
            textcoords="offset points",
            xytext=(0, -15),
            fontsize=8,
        )

    ax.set_ylabel("Pass@1")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare no-think vs think evaluation metrics"
    )
    parser.add_argument(
        "--no-think",
        type=str,
        default=None,
        help="Log dir for no-think run",
    )
    parser.add_argument(
        "--think",
        type=str,
        default=None,
        help="Log dir for think run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all 5 models using default log dir structure",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum variant logs (default when --all)",
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Use standard variant logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/logs/figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.all:
        # Determine curriculum variant
        if args.curriculum and args.standard:
            parser.error("--curriculum and --standard are mutually exclusive.")
        variant = "standard" if args.standard else "curriculum"
        logs_nt, logs_tk = _logs_base(variant)

        all_results: dict[str, dict[str, dict[str, Any]]] = {}
        for display_name, dir_suffix in MODELS.items():
            nt_dir = logs_nt / dir_suffix
            tk_dir = logs_tk / dir_suffix

            nt_summary = extract_metrics_summary(nt_dir)
            tk_summary = extract_metrics_summary(tk_dir)

            all_results[display_name] = {
                "nothink": nt_summary,
                "think": tk_summary,
            }

            if nt_summary and tk_summary:
                plot_nothink_vs_think(
                    nt_summary,
                    tk_summary,
                    display_name,
                    out_dir / f"think_vs_nothink_{dir_suffix}.png",
                )
                plot_curriculum_think_comparison(
                    nt_summary,
                    tk_summary,
                    display_name,
                    out_dir / f"curriculum_think_{dir_suffix}.png",
                )
            else:
                missing = []
                if not nt_summary:
                    missing.append(f"no-think ({nt_dir})")
                if not tk_summary:
                    missing.append(f"think ({tk_dir})")
                print(
                    f"[{display_name}] Skipping — missing: {', '.join(missing)}"
                )

        # Summary chart (only models with both variants)
        complete = {
            k: v
            for k, v in all_results.items()
            if v["nothink"] and v["think"]
        }
        if complete:
            plot_all_models_summary(
                complete, out_dir / "think_vs_nothink_all_models.png"
            )

        # Print tabular summary
        print("\n" + "=" * 70)
        print(
            f"{'Model':<20} {'No-Think':>10} {'Think':>10} {'Delta':>10}"
        )
        print("-" * 70)
        for name, data in all_results.items():
            nt_rate = (
                f"{data['nothink']['grpo_pass_rate']:.3f}"
                if data["nothink"]
                else "—"
            )
            tk_rate = (
                f"{data['think']['grpo_pass_rate']:.3f}"
                if data["think"]
                else "—"
            )
            if data["nothink"] and data["think"]:
                delta = (
                    data["think"]["grpo_pass_rate"]
                    - data["nothink"]["grpo_pass_rate"]
                )
                delta_str = f"{delta:+.3f}"
            else:
                delta_str = "—"
            print(
                f"{name:<20} {nt_rate:>10} {tk_rate:>10} {delta_str:>10}"
            )
        print("=" * 70)

    elif args.no_think and args.think:
        nt_dir = Path(args.no_think)
        tk_dir = Path(args.think)
        model_name = nt_dir.name

        nt_summary = extract_metrics_summary(nt_dir)
        tk_summary = extract_metrics_summary(tk_dir)

        if nt_summary is None:
            print(f"No comparison.json found in {nt_dir}")
            return
        if tk_summary is None:
            print(f"No comparison.json found in {tk_dir}")
            return

        plot_nothink_vs_think(
            nt_summary,
            tk_summary,
            model_name,
            out_dir / f"think_vs_nothink_{model_name}.png",
        )
        plot_curriculum_think_comparison(
            nt_summary,
            tk_summary,
            model_name,
            out_dir / f"curriculum_think_{model_name}.png",
        )
    else:
        parser.error(
            "Provide --no-think and --think paths, or use --all"
        )


if __name__ == "__main__":
    main()
