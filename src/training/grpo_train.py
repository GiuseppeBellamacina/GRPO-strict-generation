"""GRPO training script using trl GRPOTrainer.

Usage:
    python -m src.training.grpo_train --config experiments/configs/grpo.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from dotenv import load_dotenv
from trl.trainer.grpo_config import GRPOConfig

load_dotenv()
from trl.trainer.grpo_trainer import GRPOTrainer

from datasets import Dataset
from src.datasets.dataloader import (
    format_prompt_for_model,
    load_config,
    load_synthetic_dataset,
)
from src.models.model_loader import load_model_and_tokenizer
from src.training.rewards import combined_reward


def build_reward_fn(reward_cfg: dict):
    """Build a reward function from config, compatible with GRPOTrainer.

    GRPOTrainer calls: reward_fn(prompts, completions, **kwargs) -> list[float]
    """
    partial_credit = reward_cfg.get("partial_credit", False)
    reasoning_bonus = reward_cfg.get("reasoning_bonus", 0.0)

    def reward_fn(completions, **kwargs):
        """Reward function for GRPOTrainer.

        Args:
            completions: list of completion strings.
            **kwargs: additional keyword arguments (e.g., prompts).

        Returns:
            list of float rewards.
        """
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            # Detect task type from content heuristics
            task_type = _detect_task_type(text)
            r = combined_reward(
                text,
                task_type=task_type,
                partial_credit=partial_credit,
                reasoning_bonus=reasoning_bonus,
            )
            rewards.append(r)
        return rewards

    return reward_fn


def _detect_task_type(text: str) -> str:
    """Detect whether a completion is JSON or Python based on content."""
    text_lower = text.lower()
    if "```json" in text_lower:
        return "json"
    if "```python" in text_lower or "def " in text or "class " in text:
        return "python"
    # Check if it starts like JSON
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    return "python"  # default


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for strict code/JSON generation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    grpo_cfg = config["grpo"]
    reward_cfg = config["reward"]

    # Load model and tokenizer
    print(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(config)

    # Load and prepare dataset
    print("Loading dataset...")
    ds = load_synthetic_dataset(
        path=config["dataset"]["path"],
        split=config["dataset"].get("split", "train"),
        max_samples=config["dataset"].get("max_samples"),
    )
    train_ds = ds[config["dataset"].get("split", "train")]

    # Format prompts for the model
    formatted_prompts = []
    for i in range(len(train_ds)):
        sample = train_ds[i]
        prompt = format_prompt_for_model(sample, tokenizer)
        formatted_prompts.append({"prompt": prompt, "task_type": sample["task_type"]})

    prompt_dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in formatted_prompts])

    # Build reward function
    reward_fn = build_reward_fn(reward_cfg)

    # Configure GRPO
    output_dir = training_cfg.get("output_dir", "experiments/checkpoints/grpo")
    log_dir = training_cfg.get("log_dir", "experiments/logs/grpo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "grpo-strict-generation"),
        name=wandb_cfg.get("run_name", "grpo-train"),
        config=config,
        tags=wandb_cfg.get("tags", ["grpo", config["model"]["name"].split("/")[-1]]),
    )

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=training_cfg.get("max_steps", 1000),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=training_cfg.get("warmup_steps", 50),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        logging_dir=log_dir,
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 512),
        beta=grpo_cfg.get("beta", 0.04),
        temperature=grpo_cfg.get("temperature", 0.7),
        # vLLM — accelerates generation during GRPO training
        use_vllm=grpo_cfg.get("use_vllm", False),
        report_to="wandb",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,  # type: ignore[arg-type]
        args=grpo_config,
        train_dataset=prompt_dataset,
        reward_funcs=reward_fn,  # type: ignore[arg-type]
        processing_class=tokenizer,  # type: ignore[arg-type]
    )

    # Train
    print("Starting GRPO training...")
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))  # type: ignore[union-attr]
    print(f"Final model saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
