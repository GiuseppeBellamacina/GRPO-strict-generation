"""Baseline evaluation — test off-the-shelf LLMs without any fine-tuning.

Usage:
    python -m src.evaluation --config experiments/configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)  # noqa: E402

from src.datasets.dataloader import (  # noqa: E402
    format_prompt_for_model,
    load_synthetic_dataset,
)
from src.models.model_loader import (  # noqa: E402
    load_model,
    load_model_and_tokenizer,
    load_tokenizer,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.metrics import (
    compute_detailed_metrics,
    pass_at_k,
)  # noqa: E402

load_dotenv()


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    generation_config: dict[str, Any],
    num_return_sequences: int = 1,
    batch_size: int = 4,
) -> list[list[str]]:
    """Generate completions for a list of prompts.

    Returns:
        List of lists — for each prompt, a list of num_return_sequences completions.
    """
    all_completions: list[list[str]] = []

    for i in tqdm(
        range(0, len(prompts), batch_size), desc="Generating"
    ):
        batch_prompts = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_config.get(
                    "max_new_tokens", 512
                ),
                temperature=generation_config.get("temperature", 0.7),
                top_p=generation_config.get("top_p", 0.95),
                do_sample=generation_config.get("do_sample", True),
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (skip the input)
        for j in range(len(batch_prompts)):
            input_len = inputs["input_ids"][j].shape[0]
            prompt_completions = []
            for seq_idx in range(num_return_sequences):
                idx = j * num_return_sequences + seq_idx
                generated_ids = outputs[idx][input_len:]
                text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                prompt_completions.append(text)
            all_completions.append(prompt_completions)

    return all_completions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline evaluation of off-the-shelf LLMs"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    gen_cfg = config["generation"]
    eval_cfg = config["evaluation"]

    # Initialize wandb
    wandb_cfg = config.get("wandb", {})
    eval_output_dir = eval_cfg.get(
        "output_dir", "experiments/logs/baseline"
    )
    os.environ["WANDB_DIR"] = eval_output_dir
    wandb.init(
        project=wandb_cfg.get("project", "grpo-strict-generation"),
        name=wandb_cfg.get(
            "run_name", f"baseline-{model_cfg['name'].split('/')[-1]}"
        ),
        config=config,
        tags=wandb_cfg.get(
            "tags", ["baseline", model_cfg["name"].split("/")[-1]]
        ),
        dir=eval_output_dir,
    )

    print(f"Loading model: {model_cfg['name']}")

    # Support both Unsloth and standard HuggingFace loading
    use_unsloth = model_cfg.get("use_unsloth", False)
    if use_unsloth:
        # Load via Unsloth (without LoRA — baseline has no adapters)
        base_config: dict[str, Any] = {"model": model_cfg}
        model, tokenizer = load_model_and_tokenizer(base_config)
    else:
        model = load_model(
            model_name=model_cfg["name"],
            quantization=model_cfg.get("quantization", "4bit"),
            dtype=model_cfg.get("dtype", "bfloat16"),
        )
        tokenizer = load_tokenizer(model_cfg["name"])

    print("Loading dataset...")
    ds = load_synthetic_dataset(
        path=config["dataset"]["path"],
        split=config["dataset"].get("split", "test"),
        max_samples=config["dataset"].get("max_samples"),
    )
    test_ds = ds[config["dataset"].get("split", "test")]

    # Format prompts
    prompts = [
        format_prompt_for_model(test_ds[i], tokenizer)
        for i in range(len(test_ds))
    ]
    difficulties = test_ds["difficulty"]

    num_seqs = gen_cfg.get("num_return_sequences", 1)
    print(
        f"Generating {num_seqs} completion(s) per prompt for {len(prompts)} prompts..."
    )

    completions_per_prompt = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=gen_cfg,
        num_return_sequences=num_seqs,
    )

    # Pass@k evaluation
    k_values = eval_cfg.get("pass_at_k", [1, 5, 10])
    # Only compute pass@k for k <= num_return_sequences
    valid_k = [k for k in k_values if k <= num_seqs]
    if valid_k:
        pass_k = pass_at_k(completions_per_prompt, valid_k)
        print("\nPass@k results:")
        for metric, value in pass_k.items():
            print(f"  {metric}: {value:.4f}")
    else:
        pass_k = {}

    # Detailed metrics (using first completion per prompt)
    first_completions = [comps[0] for comps in completions_per_prompt]
    detailed = compute_detailed_metrics(
        first_completions, difficulties
    )

    print(f"\nOverall Pass@1: {detailed['overall_pass_rate']:.4f}")
    print("\nPer-category breakdown:")
    for cat, stats in detailed["per_category"].items():
        print(
            f"  {cat}: {stats['pass_rate']:.4f} ({stats['valid']}/{stats['total']})"
        )

    if detailed["error_distribution"]:
        print("\nTop error types:")
        for err, count in list(
            detailed["error_distribution"].items()
        )[:10]:
            print(f"  {err}: {count}")

    # Save results
    output_dir = Path(
        eval_cfg.get("output_dir", "experiments/logs/baseline")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": model_cfg["name"],
        "pass_at_k": pass_k,
        "detailed_metrics": detailed,
        "config": config,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to {results_path}")

    # Save raw completions for analysis
    completions_path = output_dir / "completions.json"
    completion_records = []
    for i, comps in enumerate(completions_per_prompt):
        completion_records.append(
            {
                "prompt": prompts[i],
                "difficulty": difficulties[i],
                "completions": comps,
            }
        )
    completions_path.write_text(
        json.dumps(completion_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Completions saved to {completions_path}")

    # Log to wandb — solo la figura verticale, niente scalari sparsi
    categories = list(detailed["per_category"].keys())
    cat_rates = [
        detailed["per_category"][c]["pass_rate"] for c in categories
    ]
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.2), 4))
    ax.bar(categories, cat_rates, color="#2196F3")
    ax.set_ylabel("Pass@1")
    ax.set_title(
        f"Baseline Evaluation – {model_cfg['name'].split('/')[-1]}"
    )
    ax.set_ylim(0, 1)
    for i, v in enumerate(cat_rates):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "baseline_eval_pass_rates.png"
    fig.savefig(fig_path, dpi=150)
    wandb.log({"eval/pass_rates": wandb.Image(fig)})
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
