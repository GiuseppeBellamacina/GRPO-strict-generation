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
from trl import GRPOConfig, GRPOTrainer

from datasets import Dataset
from src.datasets.dataloader import (
    format_prompt_for_model,
    load_synthetic_dataset,
)
from src.evaluation.eval_baseline import generate_completions
from src.models.model_loader import load_model_and_tokenizer
from src.training.callbacks import (
    HighPrecisionLogCallback,
    SaveWandbRunIdCallback,
    WandbAlertCallback,
)
from src.training.rewards import build_reward_function
from src.utils.config import load_config
from src.utils.distributed import is_main_process
from src.utils.metrics import compute_detailed_metrics

load_dotenv()


def _build_grpo_config(
    training_cfg: dict[str, Any],
    grpo_cfg: dict[str, Any],
    full_config: dict[str, Any] | None = None,
) -> GRPOConfig:
    """Build a ``GRPOConfig`` from separated training and GRPO config dicts."""
    output_dir = training_cfg.get(
        "output_dir", "experiments/checkpoints/grpo"
    )
    log_dir = training_cfg.get("log_dir", "experiments/logs/grpo")
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
        formatted.append({"prompt": prompt})

    return Dataset.from_list(formatted)


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
        "(e.g. experiments/checkpoints/grpo) and select the best one.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

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

    # Prepare dataset
    if is_main_process():
        print("Loading dataset...")
    prompt_dataset = _prepare_prompt_dataset(config, tokenizer)
    if is_main_process():
        print(
            f"[grpo] Training dataset: {len(prompt_dataset)} prompts"
        )

    # Build reward function
    thinking = config.get("dataset", {}).get("thinking", True)
    if is_main_process():
        print(f"[grpo] thinking={'on' if thinking else 'off'}")
    reward_fn = build_reward_function(
        config["reward"], thinking=thinking
    )

    # Build GRPO config
    grpo_config = _build_grpo_config(
        config["training"], config["grpo"], config
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
    log_dir = config["training"].get(
        "log_dir", "experiments/logs/grpo"
    )
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_DIR"] = log_dir
    os.environ["WANDB_TAGS"] = ",".join(
        wandb_cfg.get(
            "tags", ["grpo", config["model"]["name"].split("/")[-1]]
        )
    )

    # When resuming, restore the previous wandb run id so metrics continue on
    # the same run in the W&B UI.  We try (in order):
    #   1. A persisted .wandb_run_id file written by SaveWandbRunIdCallback
    #   2. Parsing the most recent offline-run-* directory name (fallback)
    wandb_run_id_file = Path(log_dir) / ".wandb_run_id"
    if args.resume:
        run_id_found: str | None = None
        # 1) Try the persisted file first (most reliable)
        if wandb_run_id_file.exists():
            run_id_found = (
                wandb_run_id_file.read_text().strip() or None
            )
            if run_id_found and is_main_process():
                print(
                    f"[wandb] Resuming run id: {run_id_found} (from file)"
                )
        # 2) Fall back to directory name parsing
        if not run_id_found:
            wandb_dir = Path(log_dir) / "wandb"
            if wandb_dir.exists():
                run_dirs = sorted(
                    wandb_dir.glob("offline-run-*"),
                    key=lambda p: p.name,
                )
                if run_dirs:
                    parts = run_dirs[-1].name.split("-")
                    if len(parts) >= 4:
                        run_id_found = parts[-1]
                        if is_main_process():
                            print(
                                f"[wandb] Resuming run id: {run_id_found} (from dir)"
                            )
        if run_id_found:
            os.environ["WANDB_RUN_ID"] = run_id_found
            os.environ["WANDB_RESUME"] = "must"
        else:
            os.environ["WANDB_RESUME"] = "allow"
            if is_main_process():
                print(
                    "[wandb] No previous run found, starting new run"
                )

    if is_main_process():
        print(
            f"[wandb] project={wandb_project} run={grpo_config.run_name}"
        )

    # Initialize trainer
    if is_main_process():
        print("[grpo] Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,  # type: ignore[arg-type]
        args=grpo_config,
        train_dataset=prompt_dataset,
        reward_funcs=reward_fn,  # type: ignore[arg-type]
        processing_class=tokenizer,  # type: ignore[arg-type]
        callbacks=[
            HighPrecisionLogCallback(),
            WandbAlertCallback(),
            SaveWandbRunIdCallback(wandb_run_id_file),
        ],
    )

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
    # Use eval_grpo.sh for proper post-training evaluation instead.
    if config.get("model", {}).get("use_unsloth", False):
        if is_main_process():
            print(
                "\n[grpo] Skipping in-process checkpoint eval (Unsloth patches "
                "are incompatible with vanilla HF loading in the same process)."
                "\nUse eval_grpo.sh for post-training evaluation:"
                "\n  COMPARE=1 sbatch cluster/eval_grpo.sh"
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

    # Load test set
    ds = load_synthetic_dataset(
        path=config["dataset"]["path"],
        split="test",
    )
    test_ds = ds["test"]
    max_eval = min(200, len(test_ds))
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
                ckpt_model = ckpt_model.merge_and_unload()
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
        figures_dir = Path("figures")
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
