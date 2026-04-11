"""Synthetic dataset generator for strict JSON generation tasks.

Generates prompt-instruction pairs at three difficulty levels (simple, medium, hard)
for JSON tasks. Used as the training/eval dataset for GRPO alignment.

Usage:
    python -m src.datasets.synthetic_dataset --output data/synthetic --num_samples 5000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict
from src.datasets.templates import DIFFICULTY_WEIGHTS, POOLS


def generate_sample(
    rng: random.Random | None = None,
    difficulty_weights: dict[str, float] | None = None,
) -> dict[str, str]:
    """Generate a single prompt sample with metadata."""
    if rng is None:
        rng = random.Random()

    weights = (
        difficulty_weights
        if difficulty_weights is not None
        else DIFFICULTY_WEIGHTS
    )
    difficulty = rng.choices(
        list(weights.keys()),
        weights=list(weights.values()),
        k=1,
    )[0]

    pool_key = f"json_{difficulty}"
    pool = POOLS[pool_key]
    template = rng.choice(pool["templates"])  # type: ignore[arg-type]

    params = template["params"](rng)  # type: ignore[operator]
    instruction = template["instruction"].format(**params)  # type: ignore[union-attr]

    # Compute structured schema metadata from template (if defined)
    schema_fn = template.get("schema")
    schema_meta = ""
    if schema_fn is not None:
        schema_meta = json.dumps(
            schema_fn(params), separators=(",", ":")
        )

    return {
        "prompt": instruction,
        "difficulty": difficulty,
        "schema_meta": schema_meta,
    }


def generate_dataset(
    num_samples: int = 5000,
    seed: int = 42,
    test_ratio: float = 0.2,
    difficulty_weights: dict[str, float] | None = None,
) -> DatasetDict:
    """Generate the full synthetic dataset as a HuggingFace DatasetDict."""
    rng = random.Random(seed)
    samples = [
        generate_sample(
            rng,
            difficulty_weights=difficulty_weights,
        )
        for _ in range(num_samples)
    ]

    # Deterministic split
    rng.shuffle(samples)
    split_idx = int(len(samples) * (1 - test_ratio))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    def _to_columnar(rows: list[dict]) -> dict:
        if not rows:
            return {
                k: []
                for k in [
                    "prompt",
                    "difficulty",
                    "schema_meta",
                ]
            }
        return {k: [r[k] for r in rows] for k in rows[0]}

    splits: dict[str, Dataset] = {
        "train": Dataset.from_dict(_to_columnar(train_samples)),
    }
    if test_samples:
        splits["test"] = Dataset.from_dict(_to_columnar(test_samples))

    return DatasetDict(splits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic JSON prompt dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Total number of samples",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction for test split",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config",
    )
    args = parser.parse_args()

    print(
        f"Generating {args.num_samples} samples (seed={args.seed})..."
    )
    ds = generate_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))

    # Also save a human-readable preview
    preview_path = out_path / "preview.json"
    preview = [
        ds["train"][i] for i in range(min(10, len(ds["train"])))
    ]
    preview_path.write_text(
        json.dumps(preview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Dataset saved to {out_path}")
    print(f"  Train: {len(ds['train'])} samples")
    print(f"  Test:  {len(ds['test'])} samples")

    # Print distribution stats
    for split_name in ["train", "test"]:
        split = ds[split_name]
        diffs = split["difficulty"]
        print(f"\n  {split_name} distribution:")
        for dd in ["simple", "medium", "hard"]:
            count = sum(1 for d in diffs if d == dd)
            print(f"    json/{dd}: {count}")


if __name__ == "__main__":
    main()
