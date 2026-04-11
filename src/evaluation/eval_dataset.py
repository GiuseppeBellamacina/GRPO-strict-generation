"""Balanced eval dataset loader — generates on first use, caches to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

# Defaults for the balanced eval dataset
_DEFAULT_NUM_SAMPLES = 999
_DEFAULT_WEIGHTS = {"simple": 0.334, "medium": 0.333, "hard": 0.333}
_SEED = 0


def load_eval_dataset(
    config: dict[str, Any],
) -> Dataset:
    """Load the balanced eval dataset, generating it if missing.

    The dataset is saved to ``{dataset.path}/eval/`` and reused
    across eval runs.  Parameters (num_samples, difficulty_weights) come
    from ``config["curriculum"]["eval_dataset"]`` when present, otherwise
    sensible defaults are used (999 samples, uniform 33/33/33).

    Returns the ``"test"`` split (all samples are in test).
    """
    base_path = Path(config["dataset"]["path"])
    eval_dir = base_path / "eval"

    # Read config overrides (if any)
    eval_cfg = config.get("curriculum", {}).get("eval_dataset", {})
    num_samples = eval_cfg.get("num_samples", _DEFAULT_NUM_SAMPLES)
    difficulty_weights = eval_cfg.get(
        "difficulty_weights", _DEFAULT_WEIGHTS
    )

    expected_meta = {
        "difficulty_weights": difficulty_weights,
        "num_samples": num_samples,
        "seed": _SEED,
    }

    # Try loading from cache
    meta_path = eval_dir / "curriculum_meta.json"
    if eval_dir.exists() and meta_path.exists():
        try:
            existing = json.loads(
                meta_path.read_text(encoding="utf-8")
            )
            if existing == expected_meta:
                from datasets import load_from_disk

                print(
                    f"[eval] Loading balanced eval dataset from {eval_dir} "
                    f"({num_samples} samples)"
                )
                raw_ds = load_from_disk(str(eval_dir))
                return raw_ds["test"]
        except Exception:
            pass

    # Generate fresh
    from src.datasets.synthetic_dataset import generate_dataset

    print(
        f"[eval] Generating balanced eval dataset: {num_samples} samples, "
        f"weights={difficulty_weights}"
    )
    ds = generate_dataset(
        num_samples=num_samples,
        seed=_SEED,
        test_ratio=1.0,
        difficulty_weights=difficulty_weights,
    )

    # Save to disk
    eval_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(eval_dir))
    meta_path.write_text(
        json.dumps(expected_meta, indent=2),
        encoding="utf-8",
    )
    print(f"[eval] Eval dataset saved to {eval_dir}")

    return ds["test"]
