"""Model loading utilities: HuggingFace model + tokenizer + LoRA + quantization."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_quantization_config(quantization: str) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig based on the quantization string."""
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for generation with batched inputs
    return tokenizer


def load_model(
    model_name: str,
    quantization: str = "4bit",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> Any:
    """Load a causal LM with optional quantization."""
    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    quant_config = get_quantization_config(quantization)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model


def apply_lora(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Apply LoRA adapters to the model via PEFT."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Prepare for k-bit training if quantized
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(
    config: dict,
) -> tuple[Any, Any]:
    """High-level loader: model + tokenizer from a config dict.

    Expected config structure:
        model:
          name: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
          quantization: "4bit"
          dtype: "bfloat16"
        lora:  # optional
          r: 16
          lora_alpha: 32
          ...
    """
    model_cfg = config["model"]
    model = load_model(
        model_name=model_cfg["name"],
        quantization=model_cfg.get("quantization", "4bit"),
        dtype=model_cfg.get("dtype", "bfloat16"),
    )
    tokenizer = load_tokenizer(model_cfg["name"])

    if "lora" in config:
        lora_cfg = config["lora"]
        model = apply_lora(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )

    return model, tokenizer
