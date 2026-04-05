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
# Schema metadata registry
# ---------------------------------------------------------------------------
# Maps prompt text (first 200 chars) → parsed schema dict.
# Populated by ``register_schema_metadata()`` at dataset load time so that
# ``schema_reward`` can use exact structural constraints without embedding
# them in the prompt visible to the LLM.

_schema_registry: dict[str, dict[str, Any]] = {}


def register_schema_metadata(
    prompts: list[str], schema_metas: list[str]
) -> None:
    """Populate the schema registry from formatted prompts and JSON metadata.

    Called once after loading the dataset, before training starts.
    """
    _schema_registry.clear()
    count = 0
    for prompt, meta in zip(prompts, schema_metas):
        if not meta:
            continue
        key = str(prompt)[:200]
        try:
            _schema_registry[key] = json.loads(meta)
            count += 1
        except json.JSONDecodeError:
            pass
    print(
        f"[schema] Registered {count}/{len(prompts)} schema metadata entries"
    )


def _lookup_schema(prompt: str) -> dict[str, Any] | None:
    """Look up schema metadata for a prompt.

    Priority:
      1. Module-level registry (populated from dataset at training time)
      2. Inline ``[SCHEMA:{...}]`` tag in the text (used in tests / eval)
    """
    key = str(prompt)[:200]
    result = _schema_registry.get(key)
    if result is not None:
        return result
    # Fallback: parse inline tag (for tests, eval, and backward compat)
    m = re.search(r"\[SCHEMA:(.+)\]", prompt)
    if m:
        try:
            return json.loads(m.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from text."""
    return re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL
    ).strip()


def extract_code_block(text: str, language: str) -> str | None:
    """Extract the first fenced code block of the given language from text.

    Looks for ```language ... ``` patterns. Falls back to the first
    generic ``` ... ``` block if no language-specific block is found.

    Any ``<think>...</think>`` blocks are stripped before the bare-JSON
    fallback so that reasoning tags don't prevent JSON detection.
    """
    pattern = rf"```{re.escape(language)}\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r"```\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strip <think> tags before bare-JSON fallback
    stripped = _strip_think_tags(text)
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
    units = r"items?|objects?|elements?|steps?|widgets?|operations?|tasks?|entries?|pairs?|records?|values?|keys?|fields?|abbreviations?|names?|strings?|numbers?|things?|examples?"
    m = re.search(rf"(\d+)\s+(?:{units})", instruction, re.IGNORECASE)
    if not m:
        # Handle "N <words> <unit>" e.g. "5 unit of measurement abbreviations"
        m = re.search(
            rf"(\d+)\s+(?:\w+\s+){{1,4}}(?:{units})",
            instruction,
            re.IGNORECASE,
        )
    if m:
        return int(m.group(1))
    return None


def _extract_min_count(instruction: str) -> int | None:
    """Extract minimum count from 'at least N' phrases."""
    m = re.search(r"at\s+least\s+(\d+)", instruction, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_required_keys(instruction: str) -> list[str]:
    """Extract JSON key names that must appear in the output.

    Looks for three reliable patterns:
      - ``"key_name" (type)`` or ``"key_name" key (type)`` — with parenthesised type
      - ``"key_name" key containing/which/...`` — key designator without type
      - Explicit field lists in parentheses after words like "fields" or
        "including": ``fields (page, per_page, total, total_pages)``
    """
    keys: list[str] = []

    # Pattern 1: "key_name" [optional words] (type ...) — key followed by
    # parenthesised type, with optional intervening words like "key"
    keys.extend(re.findall(r'"(\w+)"(?:\s+\w+)*\s*\(', instruction))

    # Pattern 2: "key_name" key ... — explicit "key" designator without type
    # annotation (e.g. '"items" key containing a flat array')
    keys.extend(re.findall(r'"(\w+)"\s+key\b', instruction))

    # Pattern 3: unquoted word list inside parens after "fields"/"including"
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
        r"Generate\s+a\s+(?:\w+\s+)*(JSON\s+array|list\s+of|array\s+of|JSON\s+object)",
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
    if re.search(r"```json\s*[\s\S]*?```", completion, re.IGNORECASE):
        return 1.0
    if re.search(r"```\s*[\s\S]*?```", completion):
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


def schema_reward(
    completion: str,
    instruction: str,
    raw_prompt: str = "",
) -> float:
    """Score structural compliance with the constraints stated in the instruction.

    If schema metadata is available in the registry (populated at dataset
    load time from template metadata), uses exact constraints for precise
    validation.  The registry is keyed on the full formatted prompt
    (``raw_prompt``).  Otherwise falls back to regex-based extraction from
    the instruction text.

    Checks (each contributing equally to the final average):
      1. Exact array/item count
      2. Minimum count
      3. Required top-level keys  (strict when from registry, lenient when from regex)
      4. Item keys (arrays of objects)  — *registry-only*
      5. Nesting depth
      6. Top-level type (array vs object)

    Returns 0.0 if JSON is unparseable.
    Returns 1.0 if the JSON is valid but no constraints can be extracted.
    """
    code = extract_code_block(completion, "json")
    if code is None:
        return 0.0

    parsed, _ = _parse_json_safe(code)
    if parsed is None:
        return 0.0

    # Try registry lookup using full formatted prompt, then instruction text
    schema = (
        _lookup_schema(raw_prompt) if raw_prompt else None
    ) or _lookup_schema(instruction)
    scores: list[float] = []

    # 1. Exact count — from tag or regex
    exact = (
        schema.get("count")
        if schema
        else _extract_exact_count(instruction)
    )
    if exact is not None:
        lengths = _collect_array_lengths(parsed)
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

    # 2. Minimum count — from tag or regex
    minimum = (
        schema.get("min_count")
        if schema
        else _extract_min_count(instruction)
    )
    if minimum is not None:
        lengths = _collect_array_lengths(parsed)
        best = (
            max(lengths)
            if lengths
            else (len(parsed) if isinstance(parsed, dict) else 0)
        )
        scores.append(min(best / minimum, 1.0))

    # 3. Required top-level keys
    if schema:
        req_keys = schema.get("keys", [])
    else:
        req_keys = _extract_required_keys(instruction)
    if req_keys:
        if schema:
            # Strict: keys must be at the TOP LEVEL of the object
            if isinstance(parsed, dict):
                present = sum(1 for k in req_keys if k in parsed)
                scores.append(present / len(req_keys))
            else:
                scores.append(0.0)
        else:
            # Lenient fallback: search recursively
            all_keys = _collect_all_keys(parsed)
            present = sum(1 for k in req_keys if k in all_keys)
            scores.append(present / len(req_keys))

    # 4. Item keys — only from schema tag (for arrays of objects)
    item_keys: list[str] = (
        schema.get("item_keys", []) if schema else []
    )
    if item_keys and isinstance(parsed, list) and len(parsed) > 0:
        item_scores: list[float] = []
        for item in parsed:
            if isinstance(item, dict):
                present = sum(1 for k in item_keys if k in item)
                item_scores.append(present / len(item_keys))
            else:
                item_scores.append(0.0)
        scores.append(sum(item_scores) / len(item_scores))

    # 5. Nesting depth — from tag or regex
    req_depth = (
        schema.get("depth")
        if schema
        else _extract_required_depth(instruction)
    )
    if req_depth is not None:
        scores.append(
            min(_max_nesting_depth(parsed) / req_depth, 1.0)
        )

    # 6. Top-level type — from tag or regex
    if schema:
        tl_str = schema.get("toplevel")
        if tl_str == "array":
            scores.append(1.0 if isinstance(parsed, list) else 0.0)
        elif tl_str == "object":
            scores.append(1.0 if isinstance(parsed, dict) else 0.0)
    else:
        tl = _requires_array_toplevel(instruction)
        if tl is True:
            scores.append(1.0 if isinstance(parsed, list) else 0.0)
        elif tl is False:
            scores.append(1.0 if isinstance(parsed, dict) else 0.0)

    return sum(scores) / len(scores) if scores else 1.0


def reasoning_reward(completion: str) -> float:
    """Graduated reward for chain-of-thought in <think>...</think> tags.

    Scale (based on stripped content length):
      -0.5 — no <think>...</think> block at all (penalise skipping)
      -0.2 — <think> present but content < 10 chars (lazy/empty think)
       0.0 — 10 chars (minimum useful reasoning)
       linear ramp 10→80 chars toward 1.0
       1.0 — 80+ chars of reasoning (plateau — avoid rewarding overthinking)

    The cap is intentionally low: we want the model to *think*, not to
    pad its reasoning.  ~80 chars is roughly one well-formed sentence
    of planning, which is sufficient signal.

    Examples:
      no think   → -0.50
      empty think → -0.20
      30 chars  →  0.29
      50 chars  →  0.57
      80+ chars →  1.00
    """
    m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    if not m:
        return -0.5
    content = m.group(1).strip()
    length = len(content)
    if length < 10:
        return -0.2
    # Linear ramp: 10→80 maps to 0.0→1.0
    return min((length - 10) / 70, 1.0)


def truncation_reward(completion: str) -> float:
    """Detect and penalise completions that appear truncated mid-generation.

    Truncation typically happens when the model hits ``max_completion_length``
    before closing JSON structures.  The reward is designed to NOT overlap
    with ``format_reward`` (which already penalises missing code fences):

      1.0 — completion looks structurally complete
      0.0 — no JSON detected (neutral; other rewards already handle this)
     -1.0 — bare JSON (no fence) that is clearly truncated: unclosed
            braces/brackets, unterminated strings, or trailing commas

    Only fires negatively on the ``extract_code_block`` bare-JSON fallback
    path (text starting with ``{`` or ``[`` without code fences), because
    fenced blocks that are truncated already score 0.0 on *all* other
    rewards (the regex never matches an unclosed fence).
    """
    stripped = _strip_think_tags(completion)

    # If there's a matched code fence pair (open + close ```), not truncated.
    if re.search(
        r"```(?:json)?\s*[\s\S]*?```", stripped, re.IGNORECASE
    ):
        return 1.0

    # Only care about bare JSON (the fallback path in extract_code_block)
    if not (stripped.startswith("{") or stripped.startswith("[")):
        # No JSON at all — format_reward already gives 0.0, don't
        # double-penalise.
        return 0.0

    # Bare JSON detected — check for truncation signals
    # Count unmatched structural characters
    open_braces = stripped.count("{") - stripped.count("}")
    open_brackets = stripped.count("[") - stripped.count("]")

    if open_braces > 0 or open_brackets > 0:
        return -1.0  # unclosed { or [

    # Trailing comma or colon (value was about to follow)
    if stripped.rstrip()[-1:] in (",", ":"):
        return -1.0

    # Unterminated string: odd number of unescaped quotes
    unescaped_quotes = len(re.findall(r'(?<!\\)"', stripped))
    if unescaped_quotes % 2 != 0:
        return -1.0

    return 1.0


def repetition_reward(completion: str) -> float:
    """Penalize degenerate repetitive completions.

    Catches four common failure modes of small language models:
      1. Duplicate code blocks — the same JSON repeated in multiple fenced
         blocks (model re-generates the answer)
      2. Substring token loop — a short token/word repeats consecutively
         many times within a value (e.g. ``"detail_detail_detail_..."``
         or ``"the the the the ..."``)
      3. Line-level repetition — same key-value pairs or structural lines
         appearing many times (model stuck producing similar dict entries)
      4. Word-trigram repetition — repeated phrases within values or after
         the code block (model stuck in a token loop)

    Scale:
      1.0 — normal output, low repetition
      0.0 — moderate repetition (warning signal)
     -1.0 — severe degenerate loop
    """
    text = completion.strip()

    # --- Duplicate code blocks ---
    # If there are multiple fenced code blocks, it's almost always a
    # degenerate re-generation of the same JSON.
    code_blocks = re.findall(r"```(?:\w*)\s*([\s\S]*?)```", text)
    if len(code_blocks) >= 2:
        return -1.0

    # --- Substring token loop ---
    # Catches patterns like "word_word_word_word_..." or "abc abc abc abc"
    # where a short token (2-30 chars) repeats 5+ times consecutively.
    if re.search(r"(.{2,30})\1{4,}", text):
        return -1.0

    if len(text) < 80:
        return 1.0  # too short for statistical checks below

    # --- Line-level uniqueness ---
    # Ignore very short lines (braces, commas) that naturally repeat in JSON
    lines = [
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) > 5
    ]
    line_ratio = (
        len(set(lines)) / len(lines) if len(lines) >= 6 else 1.0
    )

    # --- Word-trigram uniqueness ---
    words = text.split()
    if len(words) >= 20:
        trigrams = [
            tuple(words[i : i + 3]) for i in range(len(words) - 2)
        ]
        trigram_ratio = len(set(trigrams)) / len(trigrams)
    else:
        trigram_ratio = 1.0

    # Use the worse of the two signals
    ratio = min(line_ratio, trigram_ratio)

    if ratio > 0.5:
        return 1.0
    if ratio > 0.3:
        return 0.0
    return -1.0


def strictness_reward(completion: str) -> float:
    """Penalize extra text outside the code block.

    The system prompt instructs the model to respond ONLY with a JSON code
    block (plus optional ``<think>`` tags).  Any additional text — preambles
    like "Here's the JSON:", explanations, bullet-point lists, duplicate
    code blocks — wastes tokens and violates instructions.

    Measures the fraction of the completion that lies *outside* the first
    matched code block and any ``<think>`` block.

    Scale:
      1.0 — clean: nothing extra (0 chars outside)
      0.5 — trivial: whitespace or up to 10 chars (e.g. a newline)
      0.0 — minor: any noticeable extra text (> 10 chars)
     -0.5 — moderate: extra text > 20 % of total
     -1.0 — severe: more explanation than JSON (> 50 %)
      0.0 — no code block at all (neutral; format_reward handles this)
    """
    stripped = _strip_think_tags(completion).strip()
    if not stripped:
        return 0.0

    m = re.search(
        r"```(?:json)?\s*[\s\S]*?```", stripped, re.IGNORECASE
    )
    if not m:
        return 0.0  # no code block — format_reward already penalises

    remaining = (stripped[: m.start()] + stripped[m.end() :]).strip()
    extra_len = len(remaining)
    extra_ratio = extra_len / max(len(stripped), 1)

    if extra_len == 0:
        return 1.0
    if extra_len <= 10:
        return 0.5
    if extra_ratio < 0.20:
        return 0.0
    if extra_ratio < 0.50:
        return -0.5
    return -1.0


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


def _make_single_reward_fn(
    component_fn: Callable[..., float],
    needs_instruction: bool = False,
) -> Callable[..., list[float]]:
    """Wrap a single-sample reward component into the GRPOTrainer-compatible
    signature: ``fn(completions, prompts=None, **kwargs) -> list[float]``.

    For ``schema_reward`` (``needs_instruction=True``), the full formatted
    prompt is passed so that ``_lookup_schema`` can find the registered
    metadata by key.  The extracted user instruction is used as a fallback
    for regex-based constraint extraction.
    """

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
        raw_prompts = (
            [str(p) if p is not None else "" for p in prompts]
            if prompts
            else [""] * len(completions)
        )
        instructions = (
            [_instruction_from_prompt(p) for p in prompts]
            if prompts
            else raw_prompts
        )
        results: list[float] = []
        for completion, instruction, raw_prompt in zip(
            completions, instructions, raw_prompts
        ):
            text: str = (
                completion[0]["content"]
                if isinstance(completion, list)
                else completion
            )
            if needs_instruction:
                # Pass raw prompt for schema registry lookup
                results.append(
                    component_fn(text, instruction, raw_prompt)
                )
            else:
                results.append(component_fn(text))
        return results

    # GRPOTrainer uses fn.__name__ for wandb metric names
    reward_fn.__name__ = component_fn.__name__
    return reward_fn


def build_reward_functions(
    reward_config: dict[str, Any] | None = None,
    *,
    weight_format: float = 0.20,
    weight_validity: float = 0.35,
    weight_schema: float = 0.35,
    weight_reasoning: float = 0.10,
    weight_truncation: float = 0.0,
    weight_repetition: float = 0.0,
    weight_strictness: float = 0.0,
    thinking: bool = True,
) -> tuple[list[Callable[..., list[float]]], list[float]]:
    """Build separate reward functions + weights for GRPOTrainer multi-reward.

    Returns:
        A tuple ``(reward_funcs, reward_weights)`` where:
        - ``reward_funcs`` is a list of callables (one per active component)
        - ``reward_weights`` is a matching list of floats

    GRPOTrainer logs each function independently on wandb using ``fn.__name__``
    as the metric key, giving per-component visibility.
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
        weight_truncation = reward_config.get(
            "weight_truncation", weight_truncation
        )
        weight_repetition = reward_config.get(
            "weight_repetition", weight_repetition
        )
        weight_strictness = reward_config.get(
            "weight_strictness", weight_strictness
        )

    if not thinking:
        reasoning_share = weight_reasoning
        weight_reasoning = 0.0
        # Redistribute the reasoning share proportionally across all
        # remaining *positive* components so relative ratios are preserved.
        remaining = {
            "format": weight_format,
            "validity": weight_validity,
            "schema": weight_schema,
            "truncation": weight_truncation,
            "repetition": weight_repetition,
            "strictness": weight_strictness,
        }
        total_remaining = sum(remaining.values())
        if total_remaining > 0 and reasoning_share > 0:
            scale = reasoning_share / total_remaining
            weight_format += remaining["format"] * scale
            weight_validity += remaining["validity"] * scale
            weight_schema += remaining["schema"] * scale
            weight_truncation += remaining["truncation"] * scale
            weight_repetition += remaining["repetition"] * scale
            weight_strictness += remaining["strictness"] * scale

    funcs: list[Callable[..., list[float]]] = []
    weights: list[float] = []

    if weight_format > 0:
        funcs.append(_make_single_reward_fn(format_reward))
        weights.append(weight_format)

    if weight_validity > 0:
        funcs.append(_make_single_reward_fn(validity_reward))
        weights.append(weight_validity)

    if weight_schema > 0:
        funcs.append(
            _make_single_reward_fn(
                schema_reward, needs_instruction=True
            )
        )
        weights.append(weight_schema)

    if weight_reasoning > 0:
        funcs.append(_make_single_reward_fn(reasoning_reward))
        weights.append(weight_reasoning)

    if weight_truncation > 0:
        funcs.append(_make_single_reward_fn(truncation_reward))
        weights.append(weight_truncation)

    if weight_repetition > 0:
        funcs.append(_make_single_reward_fn(repetition_reward))
        weights.append(weight_repetition)

    if weight_strictness > 0:
        funcs.append(_make_single_reward_fn(strictness_reward))
        weights.append(weight_strictness)

    names = [f.__name__ for f in funcs]
    weight_strs = [f"{w:.2f}" for w in weights]
    print(
        f"Reward functions: {', '.join(f'{n}={w}' for n, w in zip(names, weight_strs))} "
        f"(thinking={'on' if thinking else 'off'})"
    )

    return funcs, weights
