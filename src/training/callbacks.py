"""Custom TrainerCallbacks for GRPO/SFT training."""

from __future__ import annotations

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
    """

    def __init__(self) -> None:
        self._alerted: set[int] = set()

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
            title="Training started",
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
                    title=f"Training {milestone}%",
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
            title="Training completed",
            text=f"Finished at step {state.global_step}/{args.max_steps}",
            level=wandb.AlertLevel.INFO,
        )
