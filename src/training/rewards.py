"""Reward functions for GRPO training.

All rewards are rule-based (no neural reward model).  Four components:
  1. format_reward    — proper ```json code block present
  2. validity_reward  — JSON parses without error (partial credit based on
                        how far into the text the first error occurs)
  3. schema_reward    — structural compliance with constraints stated in the
                        prompt: exact/minimum array count, required key
                        presence, nesting depth, top-level array vs object
  4. reasoning_reward — bonus for <think>...</think> with real content

Security: NEVER uses exec() or eval(). Only static parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def extract_code_block(text: str, language: str) -> str | None:
    """Extract the first fenced code block of the given language from text.

    Looks for ```language ... ``` patterns. Falls back to the first
    generic ``` ... ``` block if no language-specific block is found.
    """
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    stripped = text.strip()
    if language == "json" and (
        stripped.startswith("{") or stripped.startswith("[")
    ):
        return stripped

    return None


def _parse_json_safe(text: str) -> tuple[Any, str | None]:
    """Try to parse JSON. Returns (parsed_value, error_message_or_None)."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def _max_nesting_depth(obj: Any, _depth: int = 0) -> int:
    """Return the maximum nesting depth of a parsed JSON value."""
    if isinstance(obj, dict):
        if not obj:
            return _depth
        return max(
            _max_nesting_depth(v, _depth + 1) for v in obj.values()
        )
    if isinstance(obj, list):
        if not obj:
            return _depth
        return max(_max_nesting_depth(v, _depth) for v in obj)
    return _depth


def _collect_array_lengths(obj: Any) -> list[int]:
    """Collect the lengths of all JSON arrays in the object (recursive)."""
    lengths: list[int] = []
    if isinstance(obj, list):
        lengths.append(len(obj))
        for item in obj:
            lengths.extend(_collect_array_lengths(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            lengths.extend(_collect_array_lengths(v))
    return lengths


def _collect_all_keys(obj: Any) -> set[str]:
    """Collect all dictionary keys recursively from a parsed JSON value."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        keys.update(obj.keys())
        for v in obj.values():
            keys.update(_collect_all_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_all_keys(item))
    return keys


# ---------------------------------------------------------------------------
# Constraint extraction from natural-language instructions
# ---------------------------------------------------------------------------


def _extract_exact_count(instruction: str) -> int | None:
    """Extract the primary exact count from phrases like '3 items', '5 steps'."""
    m = re.search(r"exactly\s+(\d+)", instruction, re.IGNORECASE)
    if m:
        return int(m.group(1))
    units = r"items?|objects?|elements?|steps?|widgets?|operations?|tasks?|entries?"
    m = re.search(rf"(\d+)\s+(?:{units})", instruction, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_min_count(instruction: str) -> int | None:
    """Extract minimum count from 'at least N' phrases."""
    m = re.search(r"at\s+least\s+(\d+)", instruction, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_required_keys(instruction: str) -> list[str]:
    """Extract JSON key names that must appear in the output.

    Looks for two reliable patterns:
      - ``"key_name" (type description)`` — e.g. ``"id" (integer)``
      - Explicit field lists in parentheses after words like "fields" or
        "including": ``fields (page, per_page, total, total_pages)``
    """
    keys: list[str] = []

    # Pattern 1: "key_name" (type ...) — key followed by parenthesised type
    keys.extend(re.findall(r'"(\w+)"\s*\(', instruction))

    # Pattern 2: unquoted word list inside parens after "fields"/"including"
    m = re.search(
        r"(?:fields?|including?|attributes?|properties)\s*\(([^)]+)\)",
        instruction,
        re.IGNORECASE,
    )
    if m:
        for part in m.group(1).split(","):
            k = part.strip()
            if re.match(r"^\w+$", k):
                keys.append(k)

    # Deduplicate preserving first-seen order
    seen: set[str] = set()
    result: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result


def _extract_required_depth(instruction: str) -> int | None:
    """Return the minimum nesting depth stated in the instruction, or None."""
    if not re.search(
        r"deeply\s+nested|levels?\s+of\s+nesting|at\s+least\s+\d+\s+levels?",
        instruction,
        re.IGNORECASE,
    ):
        return None
    m = re.search(r"(\d+)\s+levels?", instruction, re.IGNORECASE)
    return int(m.group(1)) if m else 4  # default for "deeply nested"


def _requires_array_toplevel(instruction: str) -> bool | None:
    """Return True/False if instruction unambiguously requires array/object at
    the top level; None if unspecified."""
    m = re.match(
        r"Generate\s+a\s+(JSON\s+array|list\s+of|array\s+of|JSON\s+object)",
        instruction,
        re.IGNORECASE,
    )
    if not m:
        return None
    val = m.group(1).lower()
    if "object" in val:
        return False
    return True  # "array" or "list"


# ---------------------------------------------------------------------------
# Individual reward components (all return floats in [0.0, 1.0])
# ---------------------------------------------------------------------------


def format_reward(completion: str) -> float:
    """1.0 if a proper ```json ... ``` block is present, 0.5 for a generic
    ``` ... ``` block, 0.0 otherwise."""
    if re.search(
        r"```json\s*\n[\s\S]*?```", completion, re.IGNORECASE
    ):
        return 1.0
    if re.search(r"```\s*\n[\s\S]*?```", completion):
        return 0.5
    return 0.0


def validity_reward(completion: str) -> float:
    """Graduated reward for producing valid, parseable JSON.

    Scale:
      1.00 — JSON parses without error
      0.70 — parse error occurs in the last 15 % of the string (almost valid)
      0.40 — parse error in the middle section (40–85 %)
      0.20 — parse error in the first 40 % (early structural mistake)
      0.10 — code block found but content cannot be parsed at all
      0.00 — no code block
    """
    code = extract_code_block(completion, "json")
    if code is None:
        return 0.0

    parsed, err = _parse_json_safe(code)
    if parsed is not None:
        return 1.0

    if err:
        m = re.search(r"char (\d+)", err)
        if m:
            ratio = int(m.group(1)) / max(len(code), 1)
            if ratio > 0.85:
                return 0.70
            if ratio > 0.40:
                return 0.40
            return 0.20

    return 0.10


def schema_reward(completion: str, instruction: str) -> float:
    """Score structural compliance with the constraints stated in the instruction.

    Checks (each contributing equally to the final average):
      1. Exact array count     — if "N items/steps/widgets/…" in instruction
      2. Minimum count         — if "at least N" in instruction
      3. Required key presence — keys annotated as ``"key" (type)``
      4. Nesting depth         — for "deeply nested" / "at least N levels"
      5. Top-level type        — array vs object when unambiguous

    Returns 0.0 if JSON is unparseable.
    Returns 1.0 if the JSON is valid but no constraints can be extracted.
    """
    code = extract_code_block(completion, "json")
    if code is None:
        return 0.0

    parsed, _ = _parse_json_safe(code)
    if parsed is None:
        return 0.0

    scores: list[float] = []

    # 1. Exact count
    exact = _extract_exact_count(instruction)
    if exact is not None:
        lengths = _collect_array_lengths(parsed)
        # Fallback: if no arrays found and top-level is a dict, use the
        # number of keys (handles "exactly N key-value pairs" templates).
        if not lengths and isinstance(parsed, dict):
            lengths.append(len(parsed))
        if lengths:
            closest = min(lengths, key=lambda n: abs(n - exact))
            diff = abs(closest - exact)
            if diff == 0:
                scores.append(1.0)
            elif diff == 1:
                scores.append(0.70)
            elif diff <= max(1, exact // 3):
                scores.append(0.40)
            else:
                scores.append(0.10)
        else:
            scores.append(0.0)

    # 2. Minimum count
    minimum = _extract_min_count(instruction)
    if minimum is not None:
        lengths = _collect_array_lengths(parsed)
        best = (
            max(lengths)
            if lengths
            else (len(parsed) if isinstance(parsed, dict) else 0)
        )
        scores.append(min(best / minimum, 1.0))

    # 3. Required keys
    req_keys = _extract_required_keys(instruction)
    if req_keys:
        all_keys = _collect_all_keys(parsed)
        present = sum(1 for k in req_keys if k in all_keys)
        scores.append(present / len(req_keys))

    # 4. Nesting depth
    req_depth = _extract_required_depth(instruction)
    if req_depth is not None:
        scores.append(
            min(_max_nesting_depth(parsed) / req_depth, 1.0)
        )

    # 5. Top-level type
    tl = _requires_array_toplevel(instruction)
    if tl is True:
        scores.append(1.0 if isinstance(parsed, list) else 0.0)
    elif tl is False:
        scores.append(1.0 if isinstance(parsed, dict) else 0.0)

    # No extractable constraints → full credit if JSON is valid
    return sum(scores) / len(scores) if scores else 1.0


def reasoning_reward(completion: str) -> float:
    """1.0 if <think>...</think> tags with ≥20 chars of content are present,
    0.0 otherwise."""
    m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    if m and len(m.group(1).strip()) >= 20:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Combined reward and factory
# ---------------------------------------------------------------------------


def combined_reward(
    completion: str,
    instruction: str = "",
    *,
    weight_format: float = 0.20,
    weight_validity: float = 0.35,
    weight_schema: float = 0.35,
    weight_reasoning: float = 0.10,
) -> float:
    """Combine all reward components into a single scalar in [0.0, 1.0].

    All four components contribute additively so the function always
    produces a real signal rather than collapsing to 0 when no code block
    is present.  This is critical for GRPO: if every completion in the
    group scored 0 (hard gate triggered for all), the group advantage
    std = 0 and the policy gradient is exactly 0 — the model never learns
    to produce code blocks.

    Without a code block:
      * ``format_reward`` → 0.0  (no penalty elsewhere)
      * ``validity_reward`` / ``schema_reward`` still try the raw text
        fallback (``extract_code_block`` parses bare ``{``/``[`` text),
        so completions containing raw valid JSON still receive partial
        credit and create the variance GRPO needs.
      * A generic ``` block (format=0.5) penalises only the format share.
    """
    return (
        weight_format * format_reward(completion)
        + weight_validity * validity_reward(completion)
        + weight_schema * schema_reward(completion, instruction)
        + weight_reasoning * reasoning_reward(completion)
    )


def build_reward_function(
    reward_config: dict[str, Any] | None = None,
    *,
    weight_format: float = 0.20,
    weight_validity: float = 0.35,
    weight_schema: float = 0.35,
    weight_reasoning: float = 0.10,
    thinking: bool = True,
) -> Callable[..., list[float]]:
    """Build a reward function compatible with trl GRPOTrainer.

    GRPOTrainer calls: ``reward_fn(completions, prompts=None, **kwargs) -> list[float]``

    The ``prompts`` argument (list of chat-message dicts or plain strings) is
    used to extract the user instruction for structural-compliance checking.

    Args:
        reward_config: If provided, read weights from this dict (keys:
            ``weight_format``, ``weight_validity``, ``weight_schema``,
            ``weight_reasoning``).  Any key absent in the dict falls back to
            the keyword-argument default.
        weight_format: Fraction of score from format check.
        weight_validity: Fraction of score from JSON validity.
        weight_schema: Fraction of score from structural compliance.
        weight_reasoning: Fraction of score from reasoning tags.
        thinking: If False, ``weight_reasoning`` is forced to 0 and its
            share is redistributed proportionally to validity and schema,
            so the weights always sum to 1.0.
    """
    if reward_config is not None:
        weight_format = reward_config.get(
            "weight_format", weight_format
        )
        weight_validity = reward_config.get(
            "weight_validity", weight_validity
        )
        weight_schema = reward_config.get(
            "weight_schema", weight_schema
        )
        weight_reasoning = reward_config.get(
            "weight_reasoning", weight_reasoning
        )

    if not thinking:
        # Redistribute the reasoning share to validity and schema proportionally
        reasoning_share = weight_reasoning
        weight_reasoning = 0.0
        total_vs = weight_validity + weight_schema
        if total_vs > 0:
            weight_validity += reasoning_share * (
                weight_validity / total_vs
            )
            weight_schema += reasoning_share * (
                weight_schema / total_vs
            )

    print(
        f"Reward weights — format={weight_format:.2f} validity={weight_validity:.2f} "
        f"schema={weight_schema:.2f} reasoning={weight_reasoning:.2f} "
        f"(thinking={'on' if thinking else 'off'})"
    )

    def _instruction_from_prompt(prompt: Any) -> str:
        if isinstance(prompt, list):
            for msg in reversed(prompt):
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "user"
                ):
                    return msg.get("content", "")
        return str(prompt) if prompt is not None else ""

    def reward_fn(
        completions: list[Any],
        prompts: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        instructions = (
            [_instruction_from_prompt(p) for p in prompts]
            if prompts
            else [""] * len(completions)
        )
        rewards: list[float] = []
        for completion, instruction in zip(completions, instructions):
            text: str = (
                completion[0]["content"]
                if isinstance(completion, list)
                else completion
            )
            rewards.append(
                combined_reward(
                    text,
                    instruction,
                    weight_format=weight_format,
                    weight_validity=weight_validity,
                    weight_schema=weight_schema,
                    weight_reasoning=weight_reasoning,
                )
            )
        return rewards

    return reward_fn
