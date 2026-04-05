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


def _build_system_prompt(thinking: bool = True) -> str:
    """Build the system prompt instructing the model to output valid JSON.

    Args:
        thinking: If True, the model is asked to reason in <think>...</think>
            before producing the JSON block. If False, only the JSON block is
            expected (stricter, no chain-of-thought).
    """
    if thinking:
        return (
            "You are a helpful assistant that generates valid JSON. "
            "Think through the problem inside <think>...</think> tags, then respond "
            "with a JSON code block wrapped in ```json and ``` markers. "
            "No other text is allowed outside the think block and the JSON block.\n\n"
            "Example structure:\n"
            "<think>\nYour reasoning here.\n</think>\n```json\n{...}\n```"
        )
    return (
        "You are a helpful assistant that generates valid JSON. "
        "Respond ONLY with a JSON code block. Do not include any explanation "
        "before or after the JSON. Wrap your output in ```json and ``` markers."
    )


def generate_sample(
    rng: random.Random | None = None,
    thinking: bool = True,
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
    system_prompt = _build_system_prompt(thinking)

    return {
        "system_prompt": system_prompt,
        "prompt": instruction,
        "difficulty": difficulty,
    }


def generate_dataset(
    num_samples: int = 5000,
    seed: int = 42,
    test_ratio: float = 0.2,
    thinking: bool = True,
    difficulty_weights: dict[str, float] | None = None,
) -> DatasetDict:
    """Generate the full synthetic dataset as a HuggingFace DatasetDict."""
    rng = random.Random(seed)
    samples = [
        generate_sample(
            rng,
            thinking=thinking,
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
                for k in ["system_prompt", "prompt", "difficulty"]
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
        help="Path to YAML config (legge dataset.thinking)",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        default=None,
        help="Abilita <think>...</think> nel system prompt (override config)",
    )
    thinking_group.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disabilita thinking (override config)",
    )
    args = parser.parse_args()

    # Resolve thinking flag: CLI > config file > default (True)
    thinking: bool = True
    if args.config is not None:
        from src.utils.config import load_config as _load_config

        cfg = _load_config(args.config)
        thinking = cfg.get("dataset", {}).get("thinking", True)
    if args.thinking is not None:  # CLI flag overrides config
        thinking = args.thinking

    print(
        f"Generating {args.num_samples} samples (seed={args.seed}, thinking={thinking})..."
    )
    ds = generate_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        test_ratio=args.test_ratio,
        thinking=thinking,
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
