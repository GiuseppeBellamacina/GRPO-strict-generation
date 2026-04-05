"""Custom TrainerCallbacks for GRPO/SFT training."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any, Callable

import wandb
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class HighPrecisionLogCallback(TrainerCallback):
    """Print training metrics with higher float precision (8 decimal places).

    The default HuggingFace Trainer formats floats to 6 decimal places, which
    causes very small loss values (e.g. GRPO policy gradient loss) to appear
    as ``-0.000000``.  This callback reprints every ``on_log`` event to stdout
    with enough precision to see the actual values.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero or not logs:
            return
        parts = [f"step={state.global_step}"]
        for k, v in logs.items():
            parts.append(
                f"{k}={v:.8f}" if isinstance(v, float) else f"{k}={v}"
            )
        print("  " + "  ".join(parts))


class WandbAlertCallback(TrainerCallback):
    """Send wandb alerts at training start, 25%, 50%, 75%, end, and on error.

    Requires ``max_steps`` to be set in the training args so that progress
    percentages can be computed.

    Args:
        stage_label: Optional label (e.g. ``"stage 2/3: progressive"``) to
            include in alert titles for curriculum training.
    """

    def __init__(self, stage_label: str | None = None) -> None:
        self._alerted: set[int] = set()
        self._stage_label = stage_label

    def _title(self, base: str) -> str:
        if self._stage_label:
            return f"{base} [{self._stage_label}]"
        return base

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        wandb.alert(
            title=self._title("Training started"),
            text=f"max_steps={args.max_steps}, lr={args.learning_rate}",
            level=wandb.AlertLevel.INFO,
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero or not args.max_steps:
            return
        pct = int(state.global_step / args.max_steps * 100)
        for milestone in (25, 50, 75):
            if pct >= milestone and milestone not in self._alerted:
                self._alerted.add(milestone)
                reward = logs.get("reward", "n/a") if logs else "n/a"
                wandb.alert(
                    title=self._title(f"Training {milestone}%"),
                    text=f"step {state.global_step}/{args.max_steps}, reward={reward}",
                    level=wandb.AlertLevel.INFO,
                )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        wandb.alert(
            title=self._title("Training completed"),
            text=f"Finished at step {state.global_step}/{args.max_steps}",
            level=wandb.AlertLevel.INFO,
        )


class CurriculumStageCallback(TrainerCallback):
    """Log curriculum stage metadata to wandb.

    Logs ``curriculum/stage`` (1, 2, 3) at every logging step so that wandb
    charts show a clear step-function indicating which stage is active.
    Stage metadata (name, difficulty weights) is written to wandb.config
    once at the start of training.
    """

    def __init__(
        self,
        stage_idx: int,
        stage_name: str,
        difficulty_weights: dict[str, float],
    ) -> None:
        self._stage_idx = stage_idx
        self._stage_name = stage_name
        self._difficulty_weights = difficulty_weights

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        if wandb.run is not None:
            wandb.config.update(
                {
                    "curriculum_stage": self._stage_idx + 1,
                    "curriculum_stage_name": self._stage_name,
                    "curriculum_difficulty_weights": self._difficulty_weights,
                },
                allow_val_change=True,
            )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero or not logs:
            return
        if wandb.run is not None:
            wandb.log(
                {"curriculum/stage": self._stage_idx + 1},
                commit=False,
            )


class SaveWandbRunIdCallback(TrainerCallback):
    """Persist the wandb run ID to a file so ``--resume`` can continue the
    same W&B run instead of opening a new one."""

    def __init__(self, run_id_file: Path) -> None:
        self._run_id_file = run_id_file

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        if wandb.run is not None:
            self._run_id_file.write_text(wandb.run.id)
            print(f"[wandb] Run id saved: {wandb.run.id}")


# ---------------------------------------------------------------------------
# Completion sample logging
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 70


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_think(text: str) -> tuple[str, str]:
    """Split completion into (think_content, output_content)."""
    m = _THINK_RE.search(text)
    if m:
        think = m.group(1).strip()
        output = text[m.end() :].strip()
        return think, output
    return "", text.strip()


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + " [...]"


def _extract_user_instruction(prompt: Any) -> str:
    """Extract the user message from a prompt (chat messages or string)."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return str(prompt) if prompt is not None else ""


class CompletionSampleLogger:
    """Wraps reward functions to capture (prompt, completion, rewards) samples.

    The first reward function is wrapped with an interceptor that stores
    the last batch of completions and prompts.  The callback reads from
    this buffer and prints periodically.

    Usage::

        logger = CompletionSampleLogger(reward_fns, reward_weights, n_samples=3)
        wrapped_fns = logger.wrapped_reward_fns
        # pass wrapped_fns to GRPOTrainer
        # pass logger to CompletionSampleCallback
    """

    def __init__(
        self,
        reward_fns: list[Callable[..., list[float]]],
        reward_weights: list[float],
        n_samples: int = 3,
        thinking: bool = True,
    ) -> None:
        from src.training.rewards import (
            extract_code_block,
            format_reward,
            reasoning_reward,
            repetition_reward,
            schema_reward,
            strictness_reward,
            truncation_reward,
            validity_reward,
        )

        self._reward_fns = list(reward_fns)
        self._reward_weights = list(reward_weights)
        self._n_samples = n_samples
        self._buffer: deque[dict[str, Any]] = deque(maxlen=n_samples)
        self._difficulty_map: dict[str, str] = {}

        # Build component_name → weight mapping from the wrapped reward fns
        # fn.__name__ is e.g. "format_reward" → key "format"
        self._weight_map: dict[str, float] = {}
        for fn, w in zip(reward_fns, reward_weights):
            key = fn.__name__.replace("_reward", "")
            self._weight_map[key] = w

        # Component functions for per-sample breakdown
        self._component_fns = {
            "format": format_reward,
            "validity": validity_reward,
            "schema": schema_reward,
            "truncation": truncation_reward,
            "repetition": repetition_reward,
            "strictness": strictness_reward,
        }
        if thinking:
            self._component_fns["reasoning"] = reasoning_reward
        self._extract = extract_code_block

        # Wrap the first reward function to intercept
        original_fn = self._reward_fns[0]

        def _interceptor(
            completions: list[Any],
            prompts: list[Any] | None = None,
            **kwargs: Any,
        ) -> list[float]:
            self._capture(completions, prompts)
            return original_fn(completions, prompts=prompts, **kwargs)

        _interceptor.__name__ = original_fn.__name__
        self._reward_fns[0] = _interceptor

    def set_difficulty_map(self, dataset: Any) -> None:
        """Build a prompt→difficulty lookup from the training dataset."""
        for row in dataset:
            prompt = (
                row.get("prompt", "")
                if isinstance(row, dict)
                else str(row)
            )
            diff = (
                row.get("difficulty", "")
                if isinstance(row, dict)
                else ""
            )
            if prompt and diff:
                # Use first 200 chars as key to avoid huge memory usage
                self._difficulty_map[prompt[:200]] = diff

    def _capture(
        self,
        completions: list[Any],
        prompts: list[Any] | None,
    ) -> None:
        """Store the first N samples from this batch."""
        self._buffer.clear()
        n = min(self._n_samples, len(completions))
        for i in range(n):
            comp = completions[i]
            text: str = (
                comp[0]["content"] if isinstance(comp, list) else comp
            )
            prompt = prompts[i] if prompts else None
            instruction = _extract_user_instruction(prompt)

            # Look up difficulty from the prompt
            prompt_key = (
                str(prompt)[:200] if prompt is not None else ""
            )
            difficulty = self._difficulty_map.get(prompt_key, "?")

            raw_prompt = str(prompt) if prompt is not None else ""
            breakdown: dict[str, float] = {}
            for name, fn in self._component_fns.items():
                try:
                    breakdown[name] = fn(text)
                except TypeError:
                    breakdown[name] = fn(text, instruction, raw_prompt)  # type: ignore[call-arg]

            # Look up schema metadata used by schema_reward
            from src.training.rewards import _lookup_schema

            schema_info = _lookup_schema(raw_prompt)

            self._buffer.append(
                {
                    "instruction": instruction,
                    "completion": text,
                    "difficulty": difficulty,
                    "breakdown": breakdown,
                    "schema": schema_info,
                }
            )

    @property
    def wrapped_reward_fns(self) -> list[Callable[..., list[float]]]:
        return self._reward_fns

    def format_samples(self) -> str:
        """Format buffered samples as a readable string for logging."""
        if not self._buffer:
            return ""
        lines = [
            f"\n{'═' * 70}",
            "  COMPLETION SAMPLES",
            f"{'═' * 70}",
        ]
        for idx, sample in enumerate(self._buffer, 1):
            instr = sample["instruction"]
            comp = sample["completion"]
            diff = sample.get("difficulty", "?")
            bd = sample["breakdown"]
            bd_items = list(bd.items())
            has_reasoning = any(k == "reasoning" for k, _ in bd_items)
            split = 4 if has_reasoning else 3
            row1 = "  ".join(
                f"{k}={v:+.2f}" for k, v in bd_items[:split]
            )
            row2 = "  ".join(
                f"{k}={v:+.2f}" for k, v in bd_items[split:]
            )
            lines.append(f"\n{_SEPARATOR}")
            lines.append(f"  Sample {idx}  [difficulty={diff}]")
            lines.append(f"{_SEPARATOR}")
            lines.append(f"  PROMPT: {instr}")
            think, output = _split_think(comp)
            if think:
                lines.append("  THINK:")
                for cl in think.splitlines():
                    lines.append(f"    {cl}")
            lines.append("  OUTPUT:")
            for cl in output.splitlines():
                lines.append(f"    {cl}")
            lines.append(f"  REWARDS: {row1}")
            if row2:
                lines.append(f"           {row2}")
            total = sum(
                self._weight_map.get(k, 0.0) * v
                for k, v in bd.items()
            )
            lines.append(f"  TOTAL:   {total:+.4f}")
            schema = sample.get("schema")
            if schema:
                import json as _json

                lines.append(
                    f"  SCHEMA: {_json.dumps(schema, separators=(',', ':'))}"
                )
        lines.append(f"{'═' * 70}\n")
        return "\n".join(lines)


class CompletionSampleCallback(TrainerCallback):
    """Print completion samples every ``every_n_steps`` steps.

    Args:
        logger: The CompletionSampleLogger instance.
        every_n_steps: Print samples every N training steps.
    """

    def __init__(
        self,
        logger: CompletionSampleLogger,
        every_n_steps: int = 5,
    ) -> None:
        self._logger = logger
        self._every_n_steps = every_n_steps
        self._last_printed_step = -1

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        step = state.global_step
        if (
            step > 0
            and step % self._every_n_steps == 0
            and step != self._last_printed_step
        ):
            output = self._logger.format_samples()
            if output:
                print(output)
            self._last_printed_step = step
