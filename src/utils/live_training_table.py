#!/usr/bin/env python3
"""Parse HuggingFace Trainer log lines from stdin and display as a live table.

Usage:
    tail -f logs/slurm-train-1234.log | python -u -m src.utils.live_training_table
    tail -f logs/slurm-train-1234.log | python -u -m src.utils.live_training_table --cols step,reward,loss
"""

from __future__ import annotations

import ast
import re
import sys

_DEFAULT_COLS = [
    "step",
    "loss",
    "reward",
    "reward_std",
    "rewards/format_reward/mean",
    "rewards/validity_reward/mean",
    "rewards/schema_reward/mean",
    "rewards/truncation_reward/mean",
    "completion_length",
    "learning_rate",
    "grad_norm",
]

_SHORT_NAMES = {
    "rewards/format_reward/mean": "format",
    "rewards/validity_reward/mean": "validity",
    "rewards/schema_reward/mean": "schema",
    "rewards/truncation_reward/mean": "truncation",
    "completion_length": "comp_len",
    "learning_rate": "lr",
}

_DICT_PATTERN = re.compile(r"\{.*\}")
# Matches "step=240  loss=0.001  reward=0.55 ..." key=value lines
_KV_PATTERN = re.compile(r"step=\d+")


def _parse_kv_line(line: str) -> dict | None:
    """Parse 'key=value  key=value ...' formatted log lines."""
    if not _KV_PATTERN.search(line):
        return None
    # Extract the key=value portion (after any prefix like timestamps/progress bars)
    m = re.search(r"(step=\d+.*)", line)
    if not m:
        return None
    kv_part = m.group(1)
    entry: dict = {}
    for pair in re.finditer(
        r"([\w/]+)=([-+]?\d+\.?\d*(?:e[+-]?\d+)?)", kv_part
    ):
        key, val = pair.group(1), pair.group(2)
        try:
            entry[key] = float(val)
        except ValueError:
            entry[key] = val
    return entry if "step" in entry else None


def _format_val(key: str, val: object) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        if key == "step":
            return str(int(val))
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=str, default=None)
    args = parser.parse_args()

    cols = args.cols.split(",") if args.cols else _DEFAULT_COLS
    short = [_SHORT_NAMES.get(c, c) for c in cols]

    header_printed = False
    widths = [max(8, len(s)) for s in short]

    try:
        for line in sys.stdin:
            line = line.strip()
            entry = None

            # Try key=value format first (never truncated by SLURM)
            entry = _parse_kv_line(line)

            # Fall back to dict format
            if entry is None:
                m = _DICT_PATTERN.search(line)
                if m:
                    try:
                        entry = ast.literal_eval(m.group(0))
                    except (ValueError, SyntaxError):
                        pass

            if not entry or "step" not in entry:
                continue

            # Filter to requested columns that exist
            active_cols = [c for c in cols if c in entry]
            if not active_cols:
                continue

            if not header_printed:
                # Use all requested cols for header
                active_short = [
                    _SHORT_NAMES.get(c, c) for c in active_cols
                ]
                widths = [max(8, len(s)) for s in active_short]
                header = " │ ".join(
                    s.rjust(w) for s, w in zip(active_short, widths)
                )
                sep = "─┼─".join("─" * w for w in widths)
                print(f" {header}", flush=True)
                print(f" {sep}", flush=True)
                header_printed = True

            vals = [_format_val(c, entry.get(c)) for c in active_cols]
            # Update widths if needed
            for i, v in enumerate(vals):
                if len(v) > widths[i]:
                    widths[i] = len(v)
            row = " │ ".join(v.rjust(w) for v, w in zip(vals, widths))
            print(f" {row}", flush=True)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
