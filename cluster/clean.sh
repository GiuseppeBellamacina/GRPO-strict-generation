#!/bin/bash
# ============================================================================
# Pulizia workspace sul cluster DMI
#
# Uso:
#   bash cluster/clean.sh          # dry-run (mostra cosa cancellerebbe)
#   bash cluster/clean.sh --force  # cancella davvero
# ============================================================================

set -e
cd "$HOME/GRPO-strict-generation"

FORCE=0
if [ "$1" = "--force" ]; then
    FORCE=1
fi

if [ "$FORCE" = "0" ]; then
    echo "=== DRY RUN — aggiungi --force per cancellare davvero ==="
    echo ""
    CMD="echo [DRY] rm -rf"
else
    CMD="rm -rf"
fi

echo "Pulizia workspace: $PWD"
echo ""

# ── Svuota data/ (dataset generato, verrà ricreato da train.sh) ───────────
echo "[1/6] data/ (dataset sintetico)"
if [ -d "data" ]; then
    $CMD data/*
fi

# ── Svuota experiments/ tranne configs/ ───────────────────────────────────
echo "[2/6] experiments/ (checkpoints, logs — preserva configs/)"
if [ -d "experiments/checkpoints" ]; then
    $CMD experiments/checkpoints/*
fi
if [ -d "experiments/logs" ]; then
    $CMD experiments/logs/*
fi

# ── Svuota logs/ (SLURM log) ─────────────────────────────────────────────
echo "[3/6] logs/ (SLURM output)"
if [ -d "logs" ]; then
    $CMD logs/*
fi

# ── Cache Python ─────────────────────────────────────────────────────────
echo "[4/6] __pycache__/"
find . -type d -name "__pycache__" -print -exec $CMD {} + 2>/dev/null || true

# ── Artifact LoRA/Unsloth del GRPOTrainer ────────────────────────────────
echo "[5/6] grpo_trainer_lora_model_*/"
for d in grpo_trainer_lora_model_*; do
    [ -d "$d" ] && $CMD "$d"
done

# ── wandb offline runs ──────────────────────────────────────────────────
echo "[6/6] wandb/ (cartella legacy)"
if [ -d "wandb" ]; then
    $CMD wandb
fi

echo ""
if [ "$FORCE" = "0" ]; then
    echo "=== Nessun file cancellato (dry-run). Usa: bash cluster/clean.sh --force ==="
else
    echo "Pulizia completata."
fi
