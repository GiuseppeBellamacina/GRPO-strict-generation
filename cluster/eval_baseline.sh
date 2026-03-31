#!/bin/bash
# ============================================================================
# SLURM batch script — Baseline Evaluation sul cluster DMI
#
# Uso:
#   sbatch cluster/eval_baseline.sh
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=grpo-baseline
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-large
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 --gres=shard:11000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-baseline-%j.log

CONFIG="experiments/configs/baseline.yaml"

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

echo "============================================"
echo "  Baseline Evaluation — Cluster DMI"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "============================================"

mkdir -p logs

export WANDB_MODE=offline
export UNSLOTH_VLLM_STANDBY=0

cd "$HOME/GRPO-strict-generation"

echo ""
echo "Avvio baseline evaluation..."
echo ""

apptainer run --nv /shared/sifs/latest.sif \
    python -m src.evaluation.baseline_eval --config "${CONFIG}"

echo ""
echo "============================================"
echo "  Baseline evaluation completata!"
echo "  $(date)"
echo "============================================"
