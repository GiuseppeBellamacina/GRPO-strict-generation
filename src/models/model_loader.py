"""Model loading utilities: HuggingFace model + tokenizer + LoRA + quantization.

Supports two backends:
  - Standard HuggingFace (transformers + peft + bitsandbytes)
  - Unsloth (2-5x faster training, ~50-70% less VRAM)

Set  model.use_unsloth: true  in your config YAML to enable Unsloth.
Set  model.fast_inference: true  to use vLLM-backed generation.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.utils.distributed import is_main_process


def get_quantization_config(
    quantization: str, dtype: str = "bfloat16"
) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig based on the quantization string."""
    if quantization == "4bit":
        compute_dtype = getattr(torch, dtype, torch.bfloat16)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load and configure the tokenizer."""
    if is_main_process():
        print(f"[tokenizer] Loading tokenizer: {model_name}")
    tokenizer: PreTrainedTokenizerBase = (
        AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if is_main_process():
            print(
                f"[tokenizer] pad_token not set → using eos_token ({tokenizer.eos_token!r})"
            )
    tokenizer.padding_side = (
        "left"  # required for generation with batched inputs
    )
    if is_main_process():
        print(
            f"[tokenizer] padding_side=left, vocab_size={tokenizer.vocab_size}"
        )
    return tokenizer


def load_model(
    model_name: str,
    quantization: str = "4bit",
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> PreTrainedModel:
    """Load a causal LM with optional quantization."""
    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    quant_config = get_quantization_config(quantization, dtype=dtype)
    if is_main_process():
        print(
            f"[model] Loading {model_name} (quantization={quantization}, dtype={dtype}, device_map={device_map})"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return model


def apply_lora(
    model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    task_type: str = "CAUSAL_LM",
) -> PreTrainedModel:
    """Apply LoRA adapters to the model via PEFT."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if is_main_process():
        print(
            f"[lora] Applying LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, targets={target_modules}"
        )

    # Prepare for k-bit training if quantized
    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    ):
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
    config: dict[str, Any],
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """High-level loader: model + tokenizer from a config dict.

    Expected config structure:
        model:
          name: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
          quantization: "4bit"
          dtype: "bfloat16"
          use_unsloth: false  # set true to use Unsloth backend
        lora:  # optional
          r: 16
          lora_alpha: 32
          ...
    """
    model_cfg = config["model"]
    use_unsloth = model_cfg.get("use_unsloth", False)

    if use_unsloth:
        if is_main_process():
            print("[model] Backend: Unsloth")
        return _load_with_unsloth(config)

    if is_main_process():
        print("[model] Backend: HuggingFace (transformers + peft)")
    _dtype = model_cfg.get("dtype", "bfloat16")
    model = load_model(
        model_name=model_cfg["name"],
        quantization=model_cfg.get("quantization", "4bit"),
        dtype=_dtype,
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


# ── Unsloth backend ──────────────────────────────────────────────────────────


def _resolve_fast_inference(model_cfg: dict[str, Any]) -> bool:
    """Determine if fast_inference (vLLM) can be enabled.

    Returns True only when the config flag is set AND vLLM is installed.
    """
    requested = model_cfg.get("fast_inference", False)
    if not requested:
        return False

    try:
        import vllm as _vllm  # noqa: F401
    except ImportError:
        warnings.warn(
            "fast_inference richiede vllm — pacchetto non trovato, disabilitato.",
            stacklevel=2,
        )
        return False

    return True


def _load_with_unsloth(
    config: dict[str, Any],
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model + tokenizer via Unsloth's FastLanguageModel.

    Unsloth patches the model in-place with fused kernels and handles
    LoRA + 4-bit quantization internally, so it replaces the
    transformers + peft + bitsandbytes pipeline entirely.

    When ``model.fast_inference`` is ``True`` and the environment supports
    it, enables vLLM-backed generation for 2-5x faster completions
    during GRPO rollouts.
    """
    from unsloth import FastLanguageModel

    model_cfg = config["model"]
    lora_cfg = config.get("lora", {})

    # Map quantization string to load_in_4bit flag
    quantization = model_cfg.get("quantization", "4bit")
    load_in_4bit = quantization == "4bit"
    if is_main_process():
        print(
            f"[unsloth] Loading {model_cfg['name']} (quantization={quantization}, max_seq_length={model_cfg.get('max_seq_length', 2048)})"
        )

    # Resolve fast_inference availability
    use_fast_inference = _resolve_fast_inference(model_cfg)

    # fast_inference kwargs (vLLM-backed generation)
    fi_kwargs: dict[str, Any] = {}
    if use_fast_inference:
        fi_kwargs["fast_inference"] = True
        fi_kwargs["max_lora_rank"] = lora_cfg.get("r", 16)
        fi_kwargs["gpu_memory_utilization"] = model_cfg.get(
            "gpu_memory_utilization", 0.9
        )
        if is_main_process():
            print("fast_inference abilitato (vLLM backend)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
        **fi_kwargs,
    )

    # Apply LoRA via Unsloth (uses its own optimised implementation)
    if lora_cfg:
        target_modules = lora_cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        if is_main_process():
            print(
                f"[unsloth-lora] r={lora_cfg.get('r', 16)}, alpha={lora_cfg.get('lora_alpha', 32)}, targets={target_modules}"
            )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0),
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",  # 60% less VRAM
            random_state=lora_cfg.get("random_state", 3407),
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer
