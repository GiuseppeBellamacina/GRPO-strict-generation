#!/bin/bash
# ============================================================================
# SLURM batch script — Post-GRPO Evaluation sul cluster DMI
#
# Uso:
#   sbatch cluster/eval_grpo.sh                # solo eval GRPO
#   COMPARE=1 sbatch cluster/eval_grpo.sh      # eval GRPO + comparison con baseline
#
# Se baseline results.json non esiste e COMPARE=1, lo script
# esegue anche la baseline evaluation automaticamente.
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=grpo-eval
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-large
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:11000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-eval-%j.log

# ── Variabili progetto ────────────────────────────────────────────────────────
CONFIG="experiments/configs/grpo_cluster.yaml"
CHECKPOINT=""   # vuoto = usa final, oppure "experiments/checkpoints/grpo/checkpoint-480"
COMPARE="${COMPARE:-0}"   # 0 = solo GRPO, 1 = anche comparison con baseline

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

echo "============================================"
echo "  Post-GRPO Evaluation — Cluster DMI"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Compare:   ${COMPARE}"
echo "============================================"

mkdir -p logs

cd "$HOME/GRPO-strict-generation"

# Genera dataset se mancante
if [ ! -d "data/synthetic" ]; then
    echo "Dataset non trovato, generazione in corso..."
    apptainer run --nv /shared/sifs/latest.sif \
        python -m src.datasets.synthetic_dataset --config "${CONFIG}"
fi

# Costruisci argomenti
EVAL_ARGS="--config ${CONFIG}"
if [ -n "$CHECKPOINT" ]; then
    EVAL_ARGS="${EVAL_ARGS} --checkpoint ${CHECKPOINT}"
fi
if [ "$COMPARE" = "1" ]; then
    EVAL_ARGS="${EVAL_ARGS} --compare"
fi

echo ""
echo "Avvio evaluation dentro Apptainer..."
echo "  Args: ${EVAL_ARGS}"
echo ""

# ── Esecuzione dentro container Apptainer ─────────────────────────────────────
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    /shared/sifs/latest.sif \
    python -m src.evaluation ${EVAL_ARGS}

echo ""
echo "============================================"
echo "  Evaluation completata!"
echo "  $(date)"
echo "============================================"
