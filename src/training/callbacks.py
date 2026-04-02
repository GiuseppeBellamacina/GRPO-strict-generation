"""Custom TrainerCallbacks for GRPO/SFT training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class CurriculumStepOffsetCallback(TrainerCallback):
    """Offset ``global_step`` in wandb logs so steps are continuous across stages.

    Without this callback each curriculum stage logs steps starting from 0,
    making wandb charts discontinuous.  This callback adds ``step_offset``
    to every logged step so stage 2 continues where stage 1 left off.

    It also logs stage metadata (name, difficulty weights) to wandb.config
    at the start of training.
    """

    def __init__(
        self,
        step_offset: int,
        stage_idx: int,
        stage_name: str,
        difficulty_weights: dict[str, float],
    ) -> None:
        self._step_offset = step_offset
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
                    "curriculum_step_offset": self._step_offset,
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
                {
                    "curriculum/global_step": state.global_step
                    + self._step_offset
                },
                step=state.global_step + self._step_offset,
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
