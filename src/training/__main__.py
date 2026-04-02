"""Bootstrap entry point for training scripts.

Usage:
    python -m src.training --config experiments/configs/grpo_smollm2_360m.yaml [--resume] [--eval-only DIR]
    python -m src.training --config experiments/configs/sft.yaml

This module reads the config YAML *before* importing heavy libraries so that
Unsloth can be imported first when ``model.use_unsloth: true``.  Unsloth must
monkey-patch torch/transformers/trl internals before they are loaded.

The training mode (grpo or sft) is auto-detected from the config:
  - ``grpo:`` section present → GRPO training
  - ``sft:`` section present  → SFT training
"""

import argparse

import yaml


def _peek_config(config_path: str) -> dict:
    """Lightweight config read without importing anything heavy."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _detect_mode(cfg: dict) -> str:
    """Auto-detect training mode from config sections."""
    if "grpo" in cfg:
        return "grpo"
    if "sft" in cfg:
        return "sft"
    return "grpo"  # default


# Parse --config early to decide bootstrap
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", type=str, default=None)
_early_args, _ = _parser.parse_known_args()

_cfg = _peek_config(_early_args.config) if _early_args.config else {}

# Auto-disable Unsloth/fast_inference when using multiple GPUs
_num_gpus = _cfg.get("model", {}).get("num_gpus", 1)
if _num_gpus > 1:
    _cfg.setdefault("model", {})["use_unsloth"] = False
    _cfg.setdefault("model", {})["fast_inference"] = False
    print(
        f"[bootstrap] num_gpus={_num_gpus} → disabling Unsloth e fast_inference (incompatibili con multi-GPU)"
    )

# Unsloth early import — must happen before torch/transformers/trl
if _cfg.get("model", {}).get("use_unsloth", False):
    print(
        "[bootstrap] use_unsloth=True → importing Unsloth before torch/transformers/trl"
    )
    import unsloth as _unsloth  # noqa: F401

# Dispatch to the correct training script
_mode = _detect_mode(_cfg)
print(f"[bootstrap] mode={_mode} (config={_early_args.config})")

if _mode == "sft":
    from src.training.sft_train import main  # noqa: E402
else:
    from src.training.grpo_train import main  # noqa: E402

main()
