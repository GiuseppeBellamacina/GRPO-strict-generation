"""Display training log from trainer_state.json as a formatted table or plot.

Usage:
    python -m src.utils.show_training_log experiments/checkpoints/grpo/nothink/curriculum/smollm2-135m/stage_1_format_basics/checkpoint-720
    python -m src.utils.show_training_log experiments/checkpoints/grpo/nothink/curriculum/smollm2-135m/stage_1_format_basics/checkpoint-720 --cols step,loss,reward,learning_rate
    python -m src.utils.show_training_log experiments/checkpoints/grpo/nothink/standard/ --last
    python -m src.utils.show_training_log experiments/checkpoints/grpo/nothink/curriculum/ --plot          # genera grafici PNG
    python -m src.utils.show_training_log experiments/checkpoints/grpo/nothink/curriculum/ --plot --deg 5  # grado regressione
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Default columns to show (most useful for GRPO training)
_DEFAULT_COLS = [
    "step",
    "loss",
    "reward",
    "reward_std",
    "rewards/format_reward/mean",
    "rewards/validity_reward/mean",
    "rewards/schema_reward/mean",
    "rewards/reasoning_reward/mean",
    "rewards/truncation_reward/mean",
    "completion_length",
    "learning_rate",
    "grad_norm",
]


def _find_trainer_state(path: str) -> Path | None:
    """Find trainer_state.json from a checkpoint or output dir path."""
    p = Path(path)

    # Direct file
    if p.name == "trainer_state.json" and p.exists():
        return p

    # Inside a checkpoint dir
    ts = p / "trainer_state.json"
    if ts.exists():
        return ts

    # --last: find the latest checkpoint-* in a directory
    ckpts = sorted(p.glob("checkpoint-*"))
    if ckpts:
        ts = ckpts[-1] / "trainer_state.json"
        if ts.exists():
            return ts

    # Search in stage subdirs — use the LAST stage's latest checkpoint
    stage_dirs = sorted(p.glob("stage_*"))
    if stage_dirs:
        # Iterate in reverse to find the latest stage with a checkpoint
        for stage_dir in reversed(stage_dirs):
            ckpts = sorted(stage_dir.glob("checkpoint-*"))
            if ckpts:
                ts = ckpts[-1] / "trainer_state.json"
                if ts.exists():
                    return ts

    return None


def _format_value(val: object) -> str:
    """Format a value for display."""
    if val is None:
        return "-"
    if isinstance(val, float):
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def show_log(
    path: str,
    columns: list[str] | None = None,
    tail: int | None = None,
) -> None:
    ts_path = _find_trainer_state(path)
    if ts_path is None:
        print(f"No trainer_state.json found in {path}")
        return

    print(f"Source: {ts_path.parent.name}/{ts_path.name}")

    data = json.loads(ts_path.read_text(encoding="utf-8"))
    log_history = data.get("log_history", [])
    if not log_history:
        print("No log entries found.")
        return

    # Filter to training logs only (skip eval entries)
    train_logs = [
        e for e in log_history if "loss" in e or "reward" in e
    ]
    if not train_logs:
        print("No training log entries found.")
        return

    if tail:
        train_logs = train_logs[-tail:]

    cols = columns or _DEFAULT_COLS
    # Filter columns to those that actually exist
    available = set()
    for entry in train_logs:
        available.update(entry.keys())
    cols = [c for c in cols if c in available]

    if not cols:
        print("No matching columns found. Available:")
        for k in sorted(available):
            print(f"  {k}")
        return

    # Shorten column headers for display
    short_names = []
    for c in cols:
        # "rewards/format_reward/mean" → "format"
        if c.startswith("rewards/") and c.endswith("/mean"):
            short_names.append(c.split("/")[1].replace("_reward", ""))
        elif c == "completion_length":
            short_names.append("comp_len")
        elif c == "learning_rate":
            short_names.append("lr")
        else:
            short_names.append(c)

    # Compute column widths
    rows = []
    for entry in train_logs:
        row = [_format_value(entry.get(c)) for c in cols]
        rows.append(row)

    widths = [
        max(len(h), max(len(r[i]) for r in rows))
        for i, h in enumerate(short_names)
    ]

    # Print header
    header = " │ ".join(
        h.rjust(w) for h, w in zip(short_names, widths)
    )
    sep = "─┼─".join("─" * w for w in widths)
    print(f" {header}")
    print(f" {sep}")

    # Print rows
    for row in rows:
        line = " │ ".join(v.rjust(w) for v, w in zip(row, widths))
        print(f" {line}")

    print(
        f"\n{len(rows)} entries, global_step={data.get('global_step', '?')}"
    )


_PLOT_METRICS = [
    ("reward", "Mean Reward"),
    ("loss", "Loss"),
    ("rewards/format_reward/mean", "Format Reward"),
    ("rewards/validity_reward/mean", "Validity Reward"),
    ("rewards/schema_reward/mean", "Schema Reward"),
    ("rewards/truncation_reward/mean", "Truncation Reward"),
    ("completion_length", "Completion Length"),
]


def plot_training_curves(
    path: str,
    degree: int = 4,
    output_dir: str | None = None,
) -> None:
    """Generate training curve plots with polynomial regression overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    ts_path = _find_trainer_state(path)
    if ts_path is None:
        print(f"No trainer_state.json found in {path}")
        return

    data = json.loads(ts_path.read_text(encoding="utf-8"))
    log_history = data.get("log_history", [])
    train_logs = [
        e for e in log_history if "loss" in e or "reward" in e
    ]
    if not train_logs:
        print("No training log entries found.")
        return

    if output_dir is None:
        output_dir = str(ts_path.parent.parent / "figures")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Determine which metrics are available
    available_metrics = [
        (key, label)
        for key, label in _PLOT_METRICS
        if any(key in e for e in train_logs)
    ]

    if not available_metrics:
        print("No plottable metrics found.")
        return

    # Multi-panel figure: rewards on top row, others on bottom
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for idx, (key, label) in enumerate(available_metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        steps = [e["step"] for e in train_logs if key in e]
        values = [e[key] for e in train_logs if key in e]

        if not steps:
            ax.set_visible(False)
            continue

        x = np.array(steps, dtype=float)
        y = np.array(values, dtype=float)

        # Raw data (faded)
        ax.scatter(x, y, alpha=0.15, s=8, color="#1f77b4", zorder=1)

        # Polynomial regression (smooth trend)
        if len(x) > degree + 1:
            coeffs = np.polyfit(x, y, degree)
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = np.polyval(coeffs, x_smooth)
            ax.plot(
                x_smooth,
                y_smooth,
                color="#d62728",
                linewidth=2,
                zorder=2,
                label=f"poly deg={degree}",
            )
            ax.legend(fontsize=7)

        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(label)

    # Hide empty axes
    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    model_name = ts_path.parent.parent.name
    fig.suptitle(
        f"Training Curves — {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = out_path / "training_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show training log as table or plot"
    )
    parser.add_argument(
        "path", help="Path to checkpoint dir or output dir"
    )
    parser.add_argument(
        "--cols",
        type=str,
        default=None,
        help="Comma-separated column names",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=None,
        help="Show only last N entries",
    )
    parser.add_argument(
        "--all-cols",
        action="store_true",
        help="List all available columns",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training curve plots with polynomial regression",
    )
    parser.add_argument(
        "--deg",
        type=int,
        default=4,
        help="Polynomial regression degree (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if args.all_cols:
        ts_path = _find_trainer_state(args.path)
        if ts_path:
            data = json.loads(ts_path.read_text(encoding="utf-8"))
            cols = set()
            for e in data.get("log_history", []):
                cols.update(e.keys())
            print("Available columns:")
            for c in sorted(cols):
                print(f"  {c}")
        return

    if args.plot:
        plot_training_curves(
            args.path, degree=args.deg, output_dir=args.output_dir
        )
        return

    columns = args.cols.split(",") if args.cols else None
    show_log(args.path, columns=columns, tail=args.tail)


if __name__ == "__main__":
    main()
