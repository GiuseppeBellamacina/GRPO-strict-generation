"""GRPO training script using trl GRPOTrainer.

Preferred entry point (ensures Unsloth is imported before torch/trl)::

    python -m src.training --config experiments/configs/grpo.yaml

Direct invocation (Unsloth NOT guaranteed to patch before torch)::

    python -m src.training.grpo_train --config experiments/configs/grpo.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from dotenv import load_dotenv
from transformers.integrations.integration_utils import WandbCallback
from transformers.trainer_callback import ProgressCallback
from trl import GRPOConfig, GRPOTrainer  # type: ignore[import]

from datasets import Dataset
from src.datasets.dataloader import (
    format_prompt_for_model,
    load_synthetic_dataset,
)
from src.evaluation.eval_baseline import generate_completions
from src.models.model_loader import load_model_and_tokenizer
from src.training.callbacks import (
    CompletionSampleCallback,
    CompletionSampleLogger,
    GlobalStepWandbCallback,
    HighPrecisionLogCallback,
    SaveWandbRunIdCallback,
    TqdmOnlyProgressCallback,
    WandbAlertCallback,
)
from src.training.rewards import (
    build_reward_functions,
    register_schema_metadata,
)
from src.utils.config import load_config
from src.utils.distributed import is_main_process
from src.utils.metrics import compute_detailed_metrics

load_dotenv()


def _build_grpo_config(
    training_cfg: dict[str, Any],
    grpo_cfg: dict[str, Any],
    full_config: dict[str, Any] | None = None,
    reward_weights: list[float] | None = None,
) -> GRPOConfig:
    """Build a ``GRPOConfig`` from separated training and GRPO config dicts."""
    output_dir = training_cfg["output_dir"]
    log_dir = training_cfg["log_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Warmup: supports both warmup_steps and warmup_ratio
    warmup_kwargs: dict[str, Any] = {}
    if "warmup_ratio" in training_cfg:
        warmup_kwargs["warmup_ratio"] = training_cfg["warmup_ratio"]
    else:
        warmup_kwargs["warmup_steps"] = training_cfg.get(
            "warmup_steps", 50
        )

    # Resolve wandb run_name from config, append datetime for uniqueness
    wandb_cfg = (full_config or {}).get("wandb", {})
    from datetime import datetime

    base_name = wandb_cfg.get("run_name") or "grpo-train"
    run_name = (
        f"{base_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    return GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        max_steps=training_cfg.get("max_steps", 1000),
        per_device_train_batch_size=training_cfg.get(
            "per_device_train_batch_size", 1
        ),
        gradient_accumulation_steps=training_cfg.get(
            "gradient_accumulation_steps", 8
        ),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=training_cfg.get(
            "lr_scheduler_type", "cosine"
        ),
        **warmup_kwargs,
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.1),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        logging_dir=log_dir,
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get(
            "max_completion_length", 512
        ),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 256),
        beta=grpo_cfg.get("beta", 0.04),
        temperature=grpo_cfg.get("temperature", 0.7),
        reward_weights=reward_weights,
        report_to="wandb",
    )


def _prepare_prompt_dataset(
    config: dict[str, Any], tokenizer: Any
) -> Dataset:
    """Load the synthetic dataset and format prompts for the model."""
    ds = load_synthetic_dataset(
        path=config["dataset"]["path"],
        split=config["dataset"].get("split", "train"),
        max_samples=config["dataset"].get("max_samples"),
    )
    train_ds = ds[config["dataset"].get("split", "train")]

    formatted = []
    for i in range(len(train_ds)):
        sample = train_ds[i]
        prompt = format_prompt_for_model(sample, tokenizer)
        entry: dict[str, str] = {
            "prompt": prompt,
            "difficulty": sample["difficulty"],
        }
        if "schema_meta" in sample:
            entry["schema_meta"] = sample["schema_meta"]
        formatted.append(entry)

    result = Dataset.from_list(formatted)

    # Register schema metadata for reward function lookups
    if "schema_meta" in result.column_names:
        register_schema_metadata(
            result["prompt"], result["schema_meta"]
        )

    return result


def _generate_curriculum_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    difficulty_weights: dict[str, float],
    num_samples: int = 1500,
    seed: int = 42,
    save_dir: str | None = None,
) -> Dataset:
    """Generate a training dataset with specific difficulty weights and save to disk.

    Unlike ``_prepare_prompt_dataset`` (which loads from disk), this creates
    fresh samples — used exclusively by curriculum training to produce a
    different difficulty distribution per stage.

    If ``save_dir`` is provided, the raw dataset (before prompt formatting)
    is saved to that directory for reproducibility.  If a dataset already
    exists at that path **with matching weights and sample count**, it is
    loaded from disk instead of being regenerated.
    """
    from src.datasets.synthetic_dataset import generate_dataset

    thinking = config.get("dataset", {}).get("thinking", True)

    # ── Check for cached dataset on disk ──────────────────────────────────
    if save_dir is not None:
        save_path = Path(save_dir)
        meta_path = save_path / "curriculum_meta.json"
        expected_meta = {
            "difficulty_weights": difficulty_weights,
            "num_samples": num_samples,
            "seed": seed,
            "thinking": thinking,
        }

        if save_path.exists() and meta_path.exists():
            try:
                existing_meta = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )
                if existing_meta == expected_meta:
                    if is_main_process():
                        print(
                            f"[curriculum] Loading cached dataset from {save_path}"
                        )
                    from datasets import load_from_disk

                    raw_ds = load_from_disk(str(save_path))
                    train_ds = raw_ds["train"]  # type: ignore[index]
                    formatted = []
                    for i in range(len(train_ds)):
                        sample = train_ds[i]
                        prompt = format_prompt_for_model(
                            sample, tokenizer
                        )
                        entry: dict[str, str] = {
                            "prompt": prompt,
                            "difficulty": sample["difficulty"],
                        }
                        if "schema_meta" in sample:
                            entry["schema_meta"] = sample[
                                "schema_meta"
                            ]
                        formatted.append(entry)
                    result = Dataset.from_list(formatted)
                    if "schema_meta" in result.column_names:
                        register_schema_metadata(
                            result["prompt"], result["schema_meta"]
                        )
                    if is_main_process():
                        diffs = list(train_ds["difficulty"])
                        print(
                            f"[curriculum] Loaded {len(formatted)} samples from cache"
                        )
                        for d in ["simple", "medium", "hard"]:
                            count = sum(1 for x in diffs if x == d)
                            pct = count / len(diffs) * 100
                            print(f"  {d}: {count} ({pct:.1f}%)")
                    return result
                else:
                    if is_main_process():
                        print(
                            f"[curriculum] Cached dataset at {save_path} has "
                            f"different config — regenerating"
                        )
            except Exception:
                if is_main_process():
                    print(
                        f"[curriculum] Failed to load cache from {save_path} "
                        f"— regenerating"
                    )

    # ── Generate fresh dataset ────────────────────────────────────────────
    ds = generate_dataset(
        num_samples=num_samples,
        seed=seed,
        test_ratio=0.0,  # curriculum stages are train-only
        thinking=thinking,
        difficulty_weights=difficulty_weights,
    )
    train_ds = ds["train"]

    # Count distribution
    diff_counts: dict[str, int] = {
        "simple": 0,
        "medium": 0,
        "hard": 0,
    }
    for i in range(len(train_ds)):
        diff_counts[train_ds[i]["difficulty"]] += 1

    if is_main_process():
        print(
            f"[curriculum] Generated {num_samples} samples (seed={seed})"
        )
        for d in ["simple", "medium", "hard"]:
            pct = diff_counts[d] / num_samples * 100
            print(f"  {d}: {diff_counts[d]} ({pct:.1f}%)")

    # ── Save to disk ──────────────────────────────────────────────────────
    if save_dir is not None and is_main_process():
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))
        meta_path = save_path / "curriculum_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "difficulty_weights": difficulty_weights,
                    "num_samples": num_samples,
                    "seed": seed,
                    "thinking": thinking,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[curriculum] Dataset saved to {save_path}")

    # ── Format prompts ────────────────────────────────────────────────────
    formatted: list[dict[str, str]] = []
    for i in range(len(train_ds)):
        sample = train_ds[i]
        prompt = format_prompt_for_model(sample, tokenizer)
        entry_fmt: dict[str, str] = {
            "prompt": prompt,
            "difficulty": sample["difficulty"],
        }
        if "schema_meta" in sample:
            entry_fmt["schema_meta"] = sample["schema_meta"]
        formatted.append(entry_fmt)

    result = Dataset.from_list(formatted)
    if "schema_meta" in result.column_names:
        register_schema_metadata(
            result["prompt"], result["schema_meta"]
        )
    return result


def _load_stage_weights(model: Any, stage_path: Path) -> None:
    """Load trained adapter weights from a completed curriculum stage."""
    from peft import set_peft_model_state_dict

    safetensors_file = stage_path / "adapter_model.safetensors"
    bin_file = stage_path / "adapter_model.bin"

    if safetensors_file.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_file))
    elif bin_file.exists():
        state = torch.load(
            str(bin_file), map_location="cpu", weights_only=True
        )
    else:
        raise FileNotFoundError(
            f"No adapter weights found in {stage_path}"
        )

    set_peft_model_state_dict(model, state)


def _run_curriculum_training(
    config: dict[str, Any],
    model: Any,
    tokenizer: Any,
    resume: bool = False,
) -> None:
    """Run multi-stage curriculum GRPO training.

    Each stage regenerates the training dataset with different difficulty
    weights, allowing the model to first consolidate format basics on
    easier examples before progressing to harder structural tasks.

    The same model object is reused across stages (weights are updated
    in-place by GRPOTrainer), while a fresh optimizer and LR schedule
    are created per stage.

    When *resume* is True, completed stages (whose model has already been
    saved under ``stages/``) are skipped, and the first incomplete stage
    is resumed from its latest checkpoint (if any).
    """
    curriculum = config["curriculum"]
    stages = curriculum["stages"]
    base_output_dir = config["training"]["output_dir"]
    base_data_dir = config["dataset"].get("path", "data/synthetic")
    log_dir = config["training"]["log_dir"]
    num_samples = curriculum.get("num_samples", 1500)
    thinking = config.get("dataset", {}).get("thinking", True)
    total_steps = sum(s["steps"] for s in stages)

    # ── Wandb setup — single run for all stages ─────────────────────────
    wandb_cfg = config.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "grpo-strict-generation")
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_DIR"] = log_dir
    os.environ["WANDB_TAGS"] = ",".join(
        wandb_cfg.get("tags", ["grpo", "curriculum"])
    )

    # Initialize a single wandb run that spans all stages.
    # GRPOTrainer detects the existing run and reuses it.
    run_name = wandb_cfg.get("run_name", "grpo-curriculum")

    # On resume, try to continue the same wandb run
    wandb_init_kwargs: dict[str, Any] = dict(
        project=wandb_project,
        name=run_name,
        config=config,
        tags=wandb_cfg.get("tags", ["grpo", "curriculum"]),
        dir=log_dir,
    )
    if resume:
        prev_run_id: str | None = None
        # 1. Try .wandb_run_id files (most recent stage first)
        for i in range(len(stages), 0, -1):
            rid_file = Path(log_dir) / f".wandb_run_id_stage_{i}"
            if rid_file.exists():
                prev_run_id = rid_file.read_text().strip() or None
                break
        # 2. Fallback: scan offline-run-* dirs
        if not prev_run_id:
            wandb_dir = Path(log_dir) / "wandb"
            if wandb_dir.exists():
                run_dirs = sorted(
                    wandb_dir.glob("offline-run-*"),
                    key=lambda p: p.name,
                )
                if run_dirs:
                    parts = run_dirs[-1].name.split("-")
                    if len(parts) >= 4:
                        prev_run_id = parts[-1]
        if prev_run_id:
            wandb_init_kwargs["id"] = prev_run_id
            wandb_init_kwargs["resume"] = "must"
            if is_main_process():
                print(f"[wandb] Resuming run: {prev_run_id}")
        else:
            if is_main_process():
                print("[wandb] No previous run found, starting new")

    if is_main_process():
        wandb.init(**wandb_init_kwargs)

    if is_main_process():
        print(f"\n{'=' * 60}")
        print(
            f"CURRICULUM TRAINING — {len(stages)} stages, "
            f"{total_steps} total steps"
        )
        print(f"{'=' * 60}")
        for i, s in enumerate(stages):
            print(
                f"  Stage {i + 1}: {s.get('name', f'stage_{i + 1}')} — "
                f"{s['steps']} steps — weights={s['difficulty_weights']}"
            )
        print()

    cumulative_steps = 0

    # ── Resume: detect completed stages ───────────────────────────────────
    stages_root = Path(base_output_dir) / "stages"
    start_stage_idx = 0
    resume_from_checkpoint: str | None = None

    if resume:
        # Check which stages already have a saved model in stages/
        for i, s in enumerate(stages):
            sname = s.get("name", f"stage_{i + 1}")
            stage_model_path = stages_root / f"stage_{i + 1}_{sname}"
            if stage_model_path.exists() and any(
                stage_model_path.iterdir()
            ):
                start_stage_idx = i + 1
                if is_main_process():
                    print(
                        f"[resume] Stage {i + 1} ({sname}) already "
                        f"completed, skipping"
                    )
            else:
                break

        if start_stage_idx >= len(stages):
            if is_main_process():
                print("[resume] All stages already completed!")
                wandb.finish()
            return

        # Look for checkpoints in the first incomplete stage
        inc_stage = stages[start_stage_idx]
        inc_name = inc_stage.get(
            "name", f"stage_{start_stage_idx + 1}"
        )
        inc_output = str(
            Path(base_output_dir)
            / f"stage_{start_stage_idx + 1}_{inc_name}"
        )
        inc_ckpts = (
            sorted(Path(inc_output).glob("checkpoint-*"))
            if Path(inc_output).exists()
            else []
        )

        if inc_ckpts:
            # Resume from the latest checkpoint inside that stage
            resume_from_checkpoint = str(inc_ckpts[-1])
            if is_main_process():
                print(
                    f"[resume] Stage {start_stage_idx + 1}: "
                    f"resuming from {resume_from_checkpoint}"
                )
        elif start_stage_idx > 0:
            # No checkpoint yet — load weights from previous completed stage
            prev = stages[start_stage_idx - 1]
            prev_name = prev.get("name", f"stage_{start_stage_idx}")
            prev_model_path = (
                stages_root / f"stage_{start_stage_idx}_{prev_name}"
            )
            if is_main_process():
                print(
                    f"[resume] Loading model from completed "
                    f"stage {start_stage_idx}: {prev_model_path}"
                )
            _load_stage_weights(model, prev_model_path)

        # Cumulative steps for stages we're skipping
        cumulative_steps = sum(
            stages[i]["steps"] for i in range(start_stage_idx)
        )
        if is_main_process():
            print(
                f"[resume] Resuming from stage {start_stage_idx + 1}"
                f"/{len(stages)} (cumulative_steps={cumulative_steps})"
            )

    # ── Pre-generate balanced eval dataset (so eval scripts find it) ──────
    if is_main_process():
        from src.evaluation.eval_dataset import (
            load_eval_dataset,
        )

        load_eval_dataset(config)

    for stage_idx, stage in enumerate(stages):
        stage_name = stage.get("name", f"stage_{stage_idx + 1}")

        # Skip already-completed stages on resume
        if stage_idx < start_stage_idx:
            continue

        if is_main_process():
            print(f"\n{'=' * 60}")
            print(
                f"STAGE {stage_idx + 1}/{len(stages)}: {stage_name}"
            )
            print(f"{'=' * 60}")

        # 1. Generate (or load cached) dataset with stage-specific difficulty weights
        stage_data_dir = str(
            Path(base_data_dir)
            / f"curriculum_stage_{stage_idx + 1}_{stage_name}"
        )
        if is_main_process():
            print(
                f"\n[curriculum] >>> Switching dataset for stage "
                f"{stage_idx + 1}/{len(stages)}: {stage_name}"
            )
            print(
                f"[curriculum]     Difficulty weights: {stage['difficulty_weights']}"
            )
            print(f"[curriculum]     Dataset dir: {stage_data_dir}")
        stage_dataset = _generate_curriculum_dataset(
            config,
            tokenizer,
            difficulty_weights=stage["difficulty_weights"],
            num_samples=num_samples,
            seed=42 + stage_idx,
            save_dir=stage_data_dir,
        )

        # 2. Stage-specific training config overrides
        stage_training = {**config["training"]}
        stage_training["max_steps"] = stage["steps"]
        stage_output = str(
            Path(base_output_dir)
            / f"stage_{stage_idx + 1}_{stage_name}"
        )
        stage_training["output_dir"] = stage_output
        if "learning_rate" in stage:
            stage_training["learning_rate"] = stage["learning_rate"]

        # 3. Stage-specific GRPO config overrides
        stage_grpo = {**config["grpo"]}
        if "temperature" in stage:
            stage_grpo["temperature"] = stage["temperature"]
        if "num_generations" in stage:
            stage_grpo["num_generations"] = stage["num_generations"]

        # 4. Stage-specific reward weights (optional override)
        stage_reward_cfg = {**config.get("reward", {})}
        if "reward" in stage:
            stage_reward_cfg.update(stage["reward"])
        reward_fns, rw = build_reward_functions(
            stage_reward_cfg, thinking=thinking
        )

        # 4b. Completion sample logger
        sample_logger = CompletionSampleLogger(
            reward_fns, rw, n_samples=3, thinking=thinking
        )
        sample_logger.set_difficulty_map(stage_dataset)
        reward_fns = sample_logger.wrapped_reward_fns

        # 5. Build GRPOConfig
        stage_wandb = {
            **wandb_cfg,
            "run_name": run_name,
        }
        stage_full_config = {**config, "wandb": stage_wandb}
        grpo_config = _build_grpo_config(
            stage_training,
            stage_grpo,
            stage_full_config,
            reward_weights=rw,
        )

        if is_main_process():
            print(
                f"[stage {stage_idx + 1}] steps={stage['steps']} "
                f"temp={stage_grpo.get('temperature', config['grpo'].get('temperature', 0.7))} "
                f"lr={stage_training.get('learning_rate', config['training'].get('learning_rate', 5e-6))} "
                f"output={stage_output}"
            )

        # 6. Callbacks
        wandb_run_id_file = (
            Path(log_dir) / f".wandb_run_id_stage_{stage_idx + 1}"
        )
        stage_label = (
            f"stage {stage_idx + 1}/{len(stages)}: {stage_name}"
        )

        # 7. Create trainer and train
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=stage_dataset,
            reward_funcs=reward_fns,
            processing_class=tokenizer,
            callbacks=[
                HighPrecisionLogCallback(),
                WandbAlertCallback(stage_label=stage_label),
                GlobalStepWandbCallback(
                    step_offset=cumulative_steps,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    difficulty_weights=stage["difficulty_weights"],
                ),
                SaveWandbRunIdCallback(wandb_run_id_file),
                CompletionSampleCallback(sample_logger),
            ],
        )
        trainer.remove_callback(ProgressCallback)
        trainer.remove_callback(WandbCallback)
        trainer.add_callback(TqdmOnlyProgressCallback)

        # Resume from checkpoint if this is the stage being resumed
        if resume_from_checkpoint is not None:
            trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )
            resume_from_checkpoint = (
                None  # only for first resumed stage
            )
        else:
            trainer.train()

        # 8. Update cumulative step counter
        cumulative_steps += stage["steps"]

        # 9. Save stage model to a dedicated directory (outside checkpoint dirs
        #    so save_total_limit never deletes them)
        stages_root = Path(base_output_dir) / "stages"
        stage_end_path = (
            stages_root / f"stage_{stage_idx + 1}_{stage_name}"
        )
        if is_main_process():
            print(
                f"[stage {stage_idx + 1}] Saving model to "
                f"{stage_end_path}..."
            )
            stage_end_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(stage_end_path))
        tokenizer.save_pretrained(str(stage_end_path))

        # 10. Free trainer resources (model stays in memory)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        if is_main_process():
            print(f"[stage {stage_idx + 1}] {stage_name} completed")

    # ── Summary ───────────────────────────────────────────────────────────
    stages_root = Path(base_output_dir) / "stages"
    final_stage = stages[-1]
    final_name = final_stage.get("name", f"stage_{len(stages)}")
    final_model_path = (
        stages_root / f"stage_{len(stages)}_{final_name}"
    )

    if is_main_process():
        print(f"\n{'=' * 60}")
        print("CURRICULUM TRAINING COMPLETE")
        print(f"Final model: {final_model_path}")
        print(f"{'=' * 60}")

    # Skip in-process eval with Unsloth (same rationale as single-stage)
    if config.get("model", {}).get("use_unsloth", False):
        if is_main_process():
            print(
                "\n[curriculum] Skipping in-process checkpoint eval "
                "(Unsloth patches incompatible with vanilla HF loading)."
                "\nUse eval scripts for post-training evaluation."
            )

    # Finish the single wandb run after all stages
    if is_main_process():
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO training for strict code/JSON generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Skip training. Evaluate checkpoints in the given directory "
        "(e.g. experiments/checkpoints/grpo/nothink/curriculum) and select the best one.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Version output directories ────────────────────────────────────────
    # Each run gets its own timestamped subdirectory so re-runs never
    # overwrite previous checkpoints or logs.
    from src.utils.config import (
        _update_latest_symlink,
        resolve_latest_run,
        resolve_run_dir,
    )

    if not args.eval_only:
        base_output = config["training"]["output_dir"]
        base_log = config["training"]["log_dir"]

        if args.resume:
            # Resuming: reuse the latest existing run directories
            config["training"]["output_dir"] = str(
                resolve_latest_run(base_output)
            )
            config["training"]["log_dir"] = str(
                resolve_latest_run(base_log)
            )
            if is_main_process():
                print(
                    f"[run] Resuming in {config['training']['output_dir']}"
                )
        else:
            # New run: create timestamped subdirectories (same run_id)
            ckpt_dir, run_id = resolve_run_dir(
                base_output, prefix="train"
            )
            log_dir_path = Path(base_log) / run_id
            log_dir_path.mkdir(parents=True, exist_ok=True)
            _update_latest_symlink(Path(base_log), run_id)

            config["training"]["output_dir"] = str(ckpt_dir)
            config["training"]["log_dir"] = str(log_dir_path)
            if is_main_process():
                print(f"[run] New run: {run_id}")
                print(f"[run]   checkpoints → {ckpt_dir}")
                print(f"[run]   logs        → {log_dir_path}")

    # ── Auto-disable Unsloth/fast_inference per multi-GPU ─────────────────
    num_gpus = config.get("model", {}).get("num_gpus", 1)
    if num_gpus > 1:
        config["model"]["use_unsloth"] = False
        config["model"]["fast_inference"] = False
        if is_main_process():
            print(
                f"[grpo] num_gpus={num_gpus} → Unsloth e fast_inference disabilitati"
            )

    # ── Eval-only mode ────────────────────────────────────────────────────
    if args.eval_only:
        _select_best_checkpoint(config, args.eval_only)
        return

    # Enable vLLM standby if fast_inference is requested (respect env override)
    if config.get("model", {}).get("fast_inference", False):
        if os.environ.get("UNSLOTH_VLLM_STANDBY") is None:
            os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
        if is_main_process():
            print(
                f"[grpo] UNSLOTH_VLLM_STANDBY={os.environ.get('UNSLOTH_VLLM_STANDBY')} (fast_inference requested)"
            )

    # Load model and tokenizer
    if is_main_process():
        print(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(config)

    # ── Curriculum mode ───────────────────────────────────────────────────
    curriculum = config.get("curriculum")
    if curriculum and curriculum.get("enabled", False):
        _run_curriculum_training(
            config, model, tokenizer, resume=args.resume
        )
        return

    # Prepare dataset
    if is_main_process():
        print("Loading dataset...")
    prompt_dataset = _prepare_prompt_dataset(config, tokenizer)
    if is_main_process():
        print(
            f"[grpo] Training dataset: {len(prompt_dataset)} prompts"
        )

    # Build reward functions (separate per-component for wandb logging)
    thinking = config.get("dataset", {}).get("thinking", True)
    if is_main_process():
        print(f"[grpo] thinking={'on' if thinking else 'off'}")
    reward_fns, reward_weights = build_reward_functions(
        config.get("reward", {}), thinking=thinking
    )

    # Completion sample logger
    sample_logger = CompletionSampleLogger(
        reward_fns, reward_weights, n_samples=3, thinking=thinking
    )
    sample_logger.set_difficulty_map(prompt_dataset)
    reward_fns = sample_logger.wrapped_reward_fns

    # Build GRPO config
    grpo_config = _build_grpo_config(
        config["training"],
        config["grpo"],
        config,
        reward_weights=reward_weights,
    )
    if is_main_process():
        print(
            f"[grpo] Hyperparams: max_steps={grpo_config.max_steps} "
            f"batch={grpo_config.per_device_train_batch_size} "
            f"grad_accum={grpo_config.gradient_accumulation_steps} "
            f"lr={grpo_config.learning_rate} "
            f"num_gen={grpo_config.num_generations} "
            f"beta={grpo_config.beta} "
            f"max_completion={grpo_config.max_completion_length}"
        )

    # ── Find resume checkpoint ────────────────────────────────────────────
    resume_from: str | None = None
    if args.resume:
        ckpts = sorted(
            Path(grpo_config.output_dir).glob("checkpoint-*")
        )
        if ckpts:
            resume_from = str(ckpts[-1])
            if is_main_process():
                print(f"Resuming from {resume_from}")
        else:
            if is_main_process():
                print("No checkpoint found, starting from scratch.")

    # Configure wandb via env vars — the GRPOTrainer handles wandb.init internally
    wandb_cfg = config.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "grpo-strict-generation")
    log_dir = config["training"]["log_dir"]
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_DIR"] = log_dir
    os.environ["WANDB_TAGS"] = ",".join(
        wandb_cfg.get(
            "tags", ["grpo", config["model"]["name"].split("/")[-1]]
        )
    )

    # When resuming, we look for the previous wandb run ID to continue
    # logging to the same run.  The actual wandb.init() is done below,
    # after creating the GRPOConfig.
    wandb_run_id_file = Path(log_dir) / ".wandb_run_id"

    if is_main_process():
        print(
            f"[wandb] project={wandb_project} run={grpo_config.run_name}"
        )

    # Initialize wandb explicitly (since we replace the default WandbCallback
    # with GlobalStepWandbCallback to control the step counter).
    wandb_init_kwargs: dict[str, Any] = dict(
        project=wandb_project,
        name=grpo_config.run_name,
        config=config,
        tags=wandb_cfg.get(
            "tags", ["grpo", config["model"]["name"].split("/")[-1]]
        ),
        dir=log_dir,
    )
    if args.resume:
        run_id_found_val: str | None = None
        if wandb_run_id_file.exists():
            run_id_found_val = (
                wandb_run_id_file.read_text().strip() or None
            )
        if not run_id_found_val:
            wandb_dir = Path(log_dir) / "wandb"
            if wandb_dir.exists():
                run_dirs = sorted(
                    wandb_dir.glob("offline-run-*"),
                    key=lambda p: p.name,
                )
                if run_dirs:
                    parts = run_dirs[-1].name.split("-")
                    if len(parts) >= 4:
                        run_id_found_val = parts[-1]
        if run_id_found_val:
            wandb_init_kwargs["id"] = run_id_found_val
            wandb_init_kwargs["resume"] = "must"

    if is_main_process():
        wandb.init(**wandb_init_kwargs)

    # Initialize trainer
    if is_main_process():
        print("[grpo] Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,  # type: ignore[arg-type]
        args=grpo_config,
        train_dataset=prompt_dataset,
        reward_funcs=reward_fns,  # type: ignore[arg-type]
        processing_class=tokenizer,  # type: ignore[arg-type]
        callbacks=[
            HighPrecisionLogCallback(),
            WandbAlertCallback(),
            GlobalStepWandbCallback(),
            SaveWandbRunIdCallback(wandb_run_id_file),
            CompletionSampleCallback(sample_logger),
        ],
    )
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(WandbCallback)
    trainer.add_callback(TqdmOnlyProgressCallback)

    # Train
    if is_main_process():
        print("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model — skip if the last checkpoint already matches max_steps
    final_path = Path(grpo_config.output_dir) / "final"
    last_ckpt = sorted(
        Path(grpo_config.output_dir).glob("checkpoint-*")
    )
    last_step = (
        int(last_ckpt[-1].name.split("-")[-1]) if last_ckpt else -1
    )
    if last_step == grpo_config.max_steps:
        # Last checkpoint IS the final model — just symlink/copy reference
        if is_main_process():
            print(
                f"[grpo] checkpoint-{last_step} matches max_steps, "
                f"skipping duplicate final save"
            )
    else:
        if is_main_process():
            print(f"[grpo] Saving final model to {final_path}...")
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))  # type: ignore[union-attr]
        if is_main_process():
            print(f"Final model saved to {final_path}")

    if is_main_process():
        wandb.finish()

    # ── Post-training checkpoint evaluation ───────────────────────────────
    # Evaluate all saved checkpoints + final on a fixed test set and pick
    # the one with the highest overall pass rate.
    # Skip when Unsloth was used: it monkey-patches transformer classes at
    # the class level, so loading a vanilla HF model in the same process
    # crashes (e.g. 'LlamaAttention' has no attribute 'apply_qkv').
    # Use eval.sh for proper post-training evaluation instead.
    if config.get("model", {}).get("use_unsloth", False):
        if is_main_process():
            print(
                "\n[grpo] Skipping in-process checkpoint eval (Unsloth patches "
                "are incompatible with vanilla HF loading in the same process)."
                "\nUse eval.sh for post-training evaluation:"
                "\n  COMPARE=1 sbatch cluster/eval.sh"
            )
    else:
        _select_best_checkpoint(config, grpo_config.output_dir)


def _select_best_checkpoint(
    config: dict[str, Any], output_dir: str
) -> None:
    """Evaluate each checkpoint on the test set and symlink the best one."""
    if not is_main_process():
        return

    output_path = Path(output_dir)

    # Collect all candidate directories
    candidates: list[Path] = []
    final = output_path / "final"
    if final.exists():
        candidates.append(final)
    for ckpt in sorted(output_path.glob("checkpoint-*")):
        if ckpt.is_dir():
            candidates.append(ckpt)

    if len(candidates) <= 1:
        print("Only one checkpoint found, skipping selection.")
        return

    # Load test set — always use balanced eval dataset
    from src.evaluation.eval_dataset import load_eval_dataset

    test_ds = load_eval_dataset(config)
    max_eval = min(len(test_ds), 999)
    eval_ds = test_ds.select(range(max_eval))
    difficulties = list(eval_ds["difficulty"])

    gen_config = {
        "max_new_tokens": config["grpo"].get(
            "max_completion_length", 512
        ),
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    }

    best_path: Path | None = None
    best_pass_rate: float = -1.0
    results: list[tuple[str, float]] = []

    # Build a config without LoRA (adapters are merged in checkpoints)
    # and without fast_inference to avoid vLLM conflicts
    # Also disable unsloth to reduce VRAM leaks between checkpoint loads
    eval_model_config = {
        "model": {
            **config["model"],
            "fast_inference": False,
            "use_unsloth": False,
        }
    }
    if "lora" in eval_model_config:
        del eval_model_config["lora"]

    for ckpt_path in candidates:
        print(f"\nEvaluating {ckpt_path.name}...")
        # Check if this is a PEFT checkpoint (has adapter_config.json)
        is_peft = (ckpt_path / "adapter_config.json").exists()

        try:
            if is_peft:
                # Load base model + merge LoRA adapters
                from peft import PeftModel

                base_config = {
                    **eval_model_config,
                    "model": {
                        **eval_model_config["model"],
                    },
                }
                ckpt_model, ckpt_tokenizer = load_model_and_tokenizer(
                    base_config
                )
                ckpt_model = PeftModel.from_pretrained(
                    ckpt_model, str(ckpt_path)
                )
                ckpt_model = ckpt_model.merge_and_unload()  # type: ignore[assignment]
            else:
                # Full model checkpoint
                ckpt_config = {
                    **eval_model_config,
                    "model": {
                        **eval_model_config["model"],
                        "name": str(ckpt_path),
                    },
                }
                ckpt_model, ckpt_tokenizer = load_model_and_tokenizer(
                    ckpt_config
                )
        except Exception as e:
            print(f"  Failed to load {ckpt_path.name}: {e}")
            continue

        prompts = [
            format_prompt_for_model(eval_ds[i], ckpt_tokenizer)
            for i in range(len(eval_ds))
        ]
        completions = generate_completions(
            model=ckpt_model,
            tokenizer=ckpt_tokenizer,
            prompts=prompts,
            generation_config=gen_config,
            num_return_sequences=1,
            batch_size=4,
        )
        first = [c[0] for c in completions]
        metrics = compute_detailed_metrics(first, difficulties)
        pr = metrics["overall_pass_rate"]
        results.append((ckpt_path.name, pr))
        print(f"  {ckpt_path.name}: pass@1 = {pr:.4f}")

        if pr > best_pass_rate:
            best_pass_rate = pr
            best_path = ckpt_path

        # Free memory
        del ckpt_model, ckpt_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*50}")
    print("Checkpoint evaluation results:")
    print(f"{'='*50}")
    for name, pr in results:
        marker = (
            " <-- BEST"
            if name == (best_path.name if best_path else "")
            else ""
        )
        print(f"  {name}: {pr:.4f}{marker}")

    if best_path and best_path.name != "final":
        # Copy best checkpoint as "best"
        best_dest = output_path / "best"
        if best_dest.exists():
            shutil.rmtree(best_dest)
        shutil.copytree(best_path, best_dest)
        print(
            f"\nBest checkpoint ({best_path.name}) copied to {best_dest}"
        )
    elif best_path:
        print(
            f"\nFinal model is already the best (pass@1 = {best_pass_rate:.4f})"
        )

    # ── Save results as JSON ──────────────────────────────────────────
    results_json = {
        "checkpoints": [
            {"name": n, "pass_rate": p} for n, p in results
        ],
        "best": best_path.name if best_path else None,
        "best_pass_rate": best_pass_rate,
    }
    json_path = output_path / "checkpoint_eval_results.json"
    json_path.write_text(
        json.dumps(results_json, indent=2), encoding="utf-8"
    )
    print(f"Checkpoint eval results saved to {json_path}")

    # ── Save comparison figure ────────────────────────────────────────
    if results:
        names = [n for n, _ in results]
        rates = [p for _, p in results]
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
        colors = [
            (
                "#4CAF50"
                if n == (best_path.name if best_path else "")
                else "#2196F3"
            )
            for n in names
        ]
        ax.bar(names, rates, color=colors)
        ax.set_ylabel("Pass@1")
        ax.set_title("Checkpoint Evaluation – Pass@1")
        ax.set_ylim(0, 1)
        for i, v in enumerate(rates):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        figures_dir = output_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "checkpoint_eval_pass_rates.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    if is_main_process():
        print(
            "WARNING: prefer 'python -m src.training --config ...' to ensure "
            "Unsloth is imported before torch/transformers/trl."
        )
    main()
