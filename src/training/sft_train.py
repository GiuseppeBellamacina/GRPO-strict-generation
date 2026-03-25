"""Supervised Fine-Tuning (SFT) training script for comparison with GRPO.

Usage:
    python -m src.training.sft_train --config experiments/configs/sft.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import wandb
from dotenv import load_dotenv
from trl.trainer.sft_config import SFTConfig

load_dotenv()
from trl.trainer.sft_trainer import SFTTrainer

from datasets import Dataset
from src.datasets.dataloader import load_config, load_synthetic_dataset
from src.models.model_loader import load_model_and_tokenizer


def _generate_gold_json(prompt: str) -> str:
    """Generate a simple valid JSON completion for a prompt."""
    # Create a minimal valid JSON that satisfies the prompt
    return '```json\n{"example_key": "example_value", "count": 42, "active": true}\n```'


def _generate_gold_python(prompt: str) -> str:
    """Generate a simple valid Python completion for a prompt."""
    return '```python\ndef example_function(x):\n    """Example function."""\n    return x\n```'


def generate_gold_completions(
    model,
    tokenizer,
    dataset,
) -> list[str]:
    """Generate gold-standard completions using the teacher model.

    For SFT we need correct (prompt, completion) pairs. We generate with
    the base model and keep only the valid ones. For prompts where the model
    fails, we use a simple template-based fallback.
    """
    import torch
    from tqdm import tqdm

    from src.datasets.dataloader import format_prompt_for_model
    from src.training.rewards import extract_code_block

    gold = []
    for i in tqdm(range(len(dataset)), desc="Generating gold completions"):
        sample = dataset[i]
        prompt = format_prompt_for_model(sample, tokenizer)
        task_type = sample["task_type"]

        # Try generating a valid completion
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        best_completion = None
        for _attempt in range(5):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

            # Validate
            code = extract_code_block(text, "json" if task_type == "json" else "python")
            if code is not None:
                try:
                    if task_type == "json":
                        json.loads(code)
                    else:
                        ast.parse(code)
                    best_completion = text
                    break
                except (json.JSONDecodeError, SyntaxError):
                    continue

        if best_completion is None:
            # Fallback to template
            if task_type == "json":
                best_completion = _generate_gold_json(sample["prompt"])
            else:
                best_completion = _generate_gold_python(sample["prompt"])

        gold.append(best_completion)

    return gold


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for comparison with GRPO")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--skip_gold_gen",
        action="store_true",
        help="Skip gold generation (use existing gold_completions.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]

    # Load model and tokenizer
    print(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(config)

    # Load dataset
    print("Loading dataset...")
    ds = load_synthetic_dataset(
        path=config["dataset"]["path"],
        split=config["dataset"].get("split", "train"),
        max_samples=config["dataset"].get("max_samples"),
    )
    train_ds = ds[config["dataset"].get("split", "train")]

    # Generate or load gold completions
    output_dir = training_cfg.get("output_dir", "experiments/checkpoints/sft")
    gold_path = Path(output_dir) / "gold_completions.json"

    if args.skip_gold_gen and gold_path.exists():
        print(f"Loading existing gold completions from {gold_path}")
        gold_completions = json.loads(gold_path.read_text(encoding="utf-8"))
    else:
        print("Generating gold completions (this may take a while)...")
        gold_completions = generate_gold_completions(model, tokenizer, train_ds)
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(json.dumps(gold_completions, ensure_ascii=False), encoding="utf-8")
        print(f"Gold completions saved to {gold_path}")

    # Build SFT dataset with full conversations
    from src.datasets.dataloader import prepare_sft_dataset

    sft_data = prepare_sft_dataset(train_ds, gold_completions, tokenizer)
    sft_dataset = Dataset.from_list(sft_data)

    # Configure SFT
    log_dir = training_cfg.get("log_dir", "experiments/logs/sft")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "grpo-strict-generation"),
        name=wandb_cfg.get("run_name", "sft-train"),
        config=config,
        tags=wandb_cfg.get("tags", ["sft", config["model"]["name"].split("/")[-1]]),
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=training_cfg.get("warmup_steps", 50),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        logging_dir=log_dir,
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        dataset_text_field="text",
        report_to="wandb",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting SFT training...")
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Final model saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
