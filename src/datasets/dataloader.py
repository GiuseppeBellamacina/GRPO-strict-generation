"""Dataset loading utilities compatible with trl trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

# Re-export for backward compatibility
from src.utils.config import load_config  # noqa: F401
from src.utils.distributed import is_main_process


def build_system_prompt(thinking: bool = True) -> str:
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
            f"Dataset not found at {ds_path}. "
            "Run: python -m src.datasets.synthetic_dataset --output data/synthetic"
        )

    ds: DatasetDict = load_from_disk(str(ds_path))  # type: ignore[assignment]
    if is_main_process():
        print(
            f"[dataset] Loaded from {ds_path}: {{{', '.join(f'{k}: {len(v)}' for k, v in ds.items())}}}"
        )

    if split is not None:
        if split not in ds:
            raise ValueError(
                f"Split '{split}' not found. Available: {list(ds.keys())}"
            )
        ds = DatasetDict({split: ds[split]})  # type: ignore[arg-type]
        if is_main_process():
            print(
                f"[dataset] Filtered to split='{split}' ({len(ds[split])} samples)"
            )

    if max_samples is not None:
        ds = DatasetDict(
            {
                k: v.select(range(min(max_samples, len(v))))
                for k, v in ds.items()
            }
        )
        if is_main_process():
            print(f"[dataset] Truncated to max_samples={max_samples}")

    return ds


def _supports_system_role(tokenizer: Any) -> bool:
    """Check if a tokenizer's chat template accepts the system role.

    Cached per tokenizer class to avoid re-running on every sample.
    """
    key = id(tokenizer)
    if key in _SYSTEM_ROLE_CACHE:
        return _SYSTEM_ROLE_CACHE[key]
    try:
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "test"},
            ],
            tokenize=False,
        )
        result = True
    except Exception:
        result = False
    _SYSTEM_ROLE_CACHE[key] = result
    return result


_SYSTEM_ROLE_CACHE: dict[int, bool] = {}


def format_prompt_for_model(
    sample: dict[str, Any],
    tokenizer: Any = None,
    thinking: bool = True,
) -> str:
    """Format a dataset sample into a chat-template prompt string.

    The system prompt is built from *thinking* — any ``system_prompt``
    field already present in *sample* is ignored.

    If a tokenizer with apply_chat_template is provided, uses it.
    Otherwise falls back to a generic ChatML-style format.
    """
    system_prompt = build_system_prompt(thinking)

    supports_system = (
        tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and _supports_system_role(tokenizer)
    )

    if supports_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"]},
        ]
    else:
        # Models like Gemma 2 don't support system role —
        # merge system prompt into user message
        messages = [
            {
                "role": "user",
                "content": system_prompt + "\n\n" + sample["prompt"],
            },
        ]

    if tokenizer is not None and hasattr(
        tokenizer, "apply_chat_template"
    ):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback: ChatML format
    parts = []
    for msg in messages:
        parts.append(
            f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
        )
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def prepare_grpo_dataset(
    ds: Any, tokenizer: Any = None, thinking: bool = True
) -> list[dict[str, str]]:
    """Prepare dataset for GRPOTrainer — returns list of dicts with 'prompt' key."""
    rows: list[dict[str, str]] = []
    for i in range(len(ds)):
        sample = ds[i]
        prompt_text = format_prompt_for_model(
            sample, tokenizer, thinking=thinking
        )
        rows.append(
            {
                "prompt": prompt_text,
                "difficulty": sample["difficulty"],
            }
        )
    return rows


def prepare_sft_dataset(
    ds: Any,
    gold_completions: list[str],
    tokenizer: Any = None,
    thinking: bool = True,
) -> list[dict[str, str]]:
    """Prepare dataset for SFTTrainer — returns list of dicts with full conversations."""
    system_prompt = build_system_prompt(thinking)
    rows = []
    for i in range(len(ds)):
        sample = ds[i]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": gold_completions[i]},
        ]

        if tokenizer is not None and hasattr(
            tokenizer, "apply_chat_template"
        ):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False
            )
        else:
            parts = []
            for msg in messages:
                parts.append(
                    f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
                )
            text = "\n".join(parts)

        rows.append({"text": text})
    return rows
