"""Reward functions for GRPO training.

All rewards are rule-based (no neural reward model):
  - JSON: validated via json.loads
  - Python: validated via ast.parse
  - Reasoning bonus: checks for <think>...</think> tags

Security: NEVER uses exec() or eval(). Only static parsing.
"""

from __future__ import annotations

import ast
import json
import re


def extract_code_block(text: str, language: str) -> str | None:
    """Extract the first fenced code block of the given language from text.

    Looks for ```language ... ``` patterns. Falls back to the first
    generic ``` ... ``` block if no language-specific block is found.
    """
    # Try language-specific block first
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: generic fenced block
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: if the entire output looks like raw JSON/code without fences
    stripped = text.strip()
    if language == "json" and (stripped.startswith("{") or stripped.startswith("[")):
        return stripped
    if language == "python" and (
        stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("import ")
    ):
        return stripped

    return None


def json_reward(completion: str, partial_credit: bool = False) -> float:
    """Compute reward for JSON generation.

    Args:
        completion: The model's full completion text.
        partial_credit: If True, award 0.5 for structurally close but invalid JSON.

    Returns:
        1.0 if valid JSON, 0.0 if invalid (or 0.5 for partial credit).
    """
    code = extract_code_block(completion, "json")
    if code is None:
        return 0.0

    try:
        json.loads(code)
        return 1.0
    except json.JSONDecodeError as e:
        if not partial_credit:
            return 0.0
        # Partial credit: if the error is near the end (e.g., trailing comma,
        # missing closing brace), the structure is mostly correct.
        error_pos_ratio = e.pos / max(len(code), 1) if hasattr(e, "pos") and e.pos else 0.0
        return 0.5 if error_pos_ratio > 0.8 else 0.0


def python_reward(completion: str, partial_credit: bool = False) -> float:
    """Compute reward for Python code generation.

    Args:
        completion: The model's full completion text.
        partial_credit: If True, award 0.5 for code with only minor syntax errors.

    Returns:
        1.0 if valid Python, 0.0 if invalid (or 0.5 for partial credit).
    """
    code = extract_code_block(completion, "python")
    if code is None:
        return 0.0

    try:
        ast.parse(code)
        return 1.0
    except SyntaxError as e:
        if not partial_credit:
            return 0.0
        # Partial credit: if the error is on the last few lines, structure is mostly OK.
        if e.lineno is not None:
            total_lines = code.count("\n") + 1
            if total_lines > 0 and e.lineno / total_lines > 0.85:
                return 0.5
        return 0.0


def reasoning_reward(completion: str) -> float:
    """Bonus reward for including reasoning before the code block.

    Checks for <think>...</think> tags containing non-trivial content.

    Returns:
        0.2 if reasoning tags found with content, 0.0 otherwise.
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Require at least 20 chars of reasoning to count
        if len(content) >= 20:
            return 0.2
    return 0.0


def combined_reward(
    completion: str,
    task_type: str,
    partial_credit: bool = False,
    reasoning_bonus: float = 0.0,
) -> float:
    """Compute the combined reward for a completion.

    Args:
        completion: The model's full completion text.
        task_type: "json" or "python".
        partial_credit: Whether to use partial credit scoring.
        reasoning_bonus: If > 0, add reasoning_reward scaled by this factor.

    Returns:
        Total reward in [0.0, 1.0 + reasoning_bonus].
    """
    if task_type == "json":
        base = json_reward(completion, partial_credit=partial_credit)
    elif task_type == "python":
        base = python_reward(completion, partial_credit=partial_credit)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if reasoning_bonus > 0:
        base += reasoning_reward(completion) * (reasoning_bonus / 0.2)

    return base


def build_reward_function(
    task_types: list[str],
    partial_credit: bool = False,
    reasoning_bonus: float = 0.0,
):
    """Build a reward function compatible with trl GRPOTrainer.

    GRPOTrainer expects: reward_fn(completions: list[str], **kwargs) -> list[float]
    The task_types list must align with the prompts passed to the trainer.
    """
    _task_types = list(task_types)  # capture a copy

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            # task_types cycles if completions > task_types (due to num_generations)
            tt = _task_types[i % len(_task_types)]
            r = combined_reward(
                completion,
                task_type=tt,
                partial_credit=partial_credit,
                reasoning_bonus=reasoning_bonus,
            )
            rewards.append(r)
        return rewards

    return reward_fn
