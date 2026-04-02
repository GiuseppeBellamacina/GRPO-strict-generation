"""Bootstrap entry point for evaluation scripts.

Usage:
    python -m src.evaluation --config experiments/configs/baseline.yaml
    python -m src.evaluation --config experiments/configs/grpo_smollm2_360m.yaml [--compare]

The evaluation mode is auto-detected from the config:
  - ``grpo:`` section present → GRPO post-training evaluation
  - otherwise              → baseline evaluation
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
    """Auto-detect evaluation mode from config sections."""
    if "grpo" in cfg:
        return "grpo"
    return "baseline"


# Parse --config early to decide bootstrap
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", type=str, default=None)
_early_args, _ = _parser.parse_known_args()

_cfg = _peek_config(_early_args.config) if _early_args.config else {}

# Unsloth early import — must happen before torch/transformers/trl
if _cfg.get("model", {}).get("use_unsloth", False):
    print(
        "[bootstrap] use_unsloth=True → importing Unsloth before torch/transformers/trl"
    )
    import unsloth as _unsloth  # noqa: F401

# Dispatch to the correct evaluation script
_mode = _detect_mode(_cfg)
print(f"[bootstrap] mode={_mode} (config={_early_args.config})")

if _mode == "baseline":
    from src.evaluation.eval_baseline import main  # noqa: E402
else:
    from src.evaluation.eval_grpo import main  # noqa: E402

main()
