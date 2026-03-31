"""Evaluation metrics for strict JSON generation.

Computes Pass@k, error type distribution, and per-task breakdowns.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from src.training.rewards import extract_code_block


def check_syntax(completion: str) -> tuple[bool, str]:
    """Check if a completion contains syntactically valid JSON.

    Returns:
        (is_valid, error_message) — error_message is "" if valid.
    """
    code = extract_code_block(completion, "json")

    if code is None:
        return False, "no_code_block"

    try:
        json.loads(code)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"json_error: {e.msg}"


def pass_at_k(
    completions_per_prompt: list[list[str]],
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Pass@k metrics.

    Args:
        completions_per_prompt: For each prompt, a list of k completions.
        k_values: List of k values to compute.

    Returns:
        Dict like {"pass@1": 0.72, "pass@5": 0.88, "pass@10": 0.93}.
    """
    results = {}
    n_prompts = len(completions_per_prompt)

    for k in k_values:
        passes = 0
        for comps in completions_per_prompt:
            # Take first k completions (or all if fewer)
            subset = comps[:k]
            if any(check_syntax(c)[0] for c in subset):
                passes += 1
        results[f"pass@{k}"] = passes / max(n_prompts, 1)

    return results


def compute_detailed_metrics(
    completions: list[str],
    difficulties: list[str],
) -> dict[str, Any]:
    """Compute detailed evaluation metrics.

    Args:
        completions: One completion per prompt.
        difficulties: Difficulty level for each prompt.

    Returns:
        Dict with overall pass rate, per-type, per-difficulty breakdowns,
        and error type distribution.
    """
    total = len(completions)
    valid_count = 0
    error_types: Counter = Counter()

    type_counts: dict[str, dict[str, int]] = {}

    for comp, diff in zip(completions, difficulties):
        is_valid, error_msg = check_syntax(comp)
        key = f"{diff}"
        type_counts.setdefault(key, {"total": 0, "valid": 0})
        type_counts[key]["total"] += 1

        if is_valid:
            valid_count += 1
            type_counts[key]["valid"] += 1
        else:
            error_types[error_msg] += 1

    # Build result
    result = {
        "overall_pass_rate": valid_count / max(total, 1),
        "total_samples": total,
        "valid_samples": valid_count,
        "per_category": {},
        "error_distribution": dict(error_types.most_common(20)),
    }

    for key, counts in sorted(type_counts.items()):
        result["per_category"][key] = {
            "total": counts["total"],
            "valid": counts["valid"],
            "pass_rate": counts["valid"] / max(counts["total"], 1),
        }

    return result
