"""Dataset loading utilities compatible with trl trainers."""

from __future__ import annotations

from pathlib import Path

import yaml

from datasets import DatasetDict, load_from_disk


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_synthetic_dataset(
    path: str = "data/synthetic",
    split: str | None = None,
    max_samples: int | None = None,
) -> DatasetDict:
    """Load the synthetic dataset from disk.

    Args:
        path: Path to the saved DatasetDict.
        split: If given, return only this split (still wrapped in DatasetDict).
        max_samples: If given, truncate each split to this many samples.

    Returns:
        A DatasetDict with 'train' and/or 'test' splits.
    """
    ds_path = Path(path)
    if not ds_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {ds_path}. " "Run: python -m src.datasets.synthetic_dataset --output data/synthetic"
        )

    ds: DatasetDict = load_from_disk(str(ds_path))  # type: ignore[assignment]

    if split is not None:
        if split not in ds:
            raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
        ds = DatasetDict({split: ds[split]})  # type: ignore[arg-type]

    if max_samples is not None:
        ds = DatasetDict({k: v.select(range(min(max_samples, len(v)))) for k, v in ds.items()})

    return ds


def format_prompt_for_model(
    sample: dict,
    tokenizer=None,
) -> str:
    """Format a dataset sample into a chat-template prompt string.

    If a tokenizer with apply_chat_template is provided, uses it.
    Otherwise falls back to a generic ChatML-style format.
    """
    messages = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": sample["prompt"]},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback: ChatML format
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def prepare_grpo_dataset(ds, tokenizer=None) -> list[dict]:
    """Prepare dataset for GRPOTrainer — returns list of dicts with 'prompt' key."""
    rows = []
    for i in range(len(ds)):
        sample = ds[i]
        prompt_text = format_prompt_for_model(sample, tokenizer)
        rows.append(
            {
                "prompt": prompt_text,
                "task_type": sample["task_type"],
                "difficulty": sample["difficulty"],
            }
        )
    return rows


def prepare_sft_dataset(ds, gold_completions: list[str], tokenizer=None) -> list[dict]:
    """Prepare dataset for SFTTrainer — returns list of dicts with full conversations."""
    rows = []
    for i in range(len(ds)):
        sample = ds[i]
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": gold_completions[i]},
        ]

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            parts = []
            for msg in messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            text = "\n".join(parts)

        rows.append({"text": text, "task_type": sample["task_type"]})
    return rows
