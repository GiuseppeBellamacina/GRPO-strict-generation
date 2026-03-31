#!/bin/bash
# ============================================================================
# SLURM batch script — GRPO Training sul cluster DMI
#
# Uso:
#   sbatch cluster/train.sh
#
# Per il primo avvio eseguire prima:  bash cluster/setup.sh
# (dentro una sessione interattiva Apptainer)
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=grpo-train
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-%j.log

# ── Variabili progetto ────────────────────────────────────────────────────────
CONFIG="experiments/configs/grpo_cluster.yaml"
EXTRA_ARGS=""   # "--resume" per riprendere da checkpoint, "--eval-only DIR" per solo eval

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

echo "============================================"
echo "  GRPO Training — Cluster DMI"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "============================================"

# Crea directory logs se non esiste
mkdir -p logs

# wandb in modalità offline (non c'è internet sul cluster per studenti)
# I dottorandi con accesso a internet possono commentare questa riga
export WANDB_MODE=offline

# Disabilita vLLM standby mode (stessa ragione di Colab)
export UNSLOTH_VLLM_STANDBY=0

# Percorso progetto
cd "$HOME/GRPO-strict-generation"

# Genera dataset se mancante (potrebbe essere stato rimosso da un upload)
if [ ! -d "data/synthetic" ]; then
    echo "Dataset non trovato, generazione in corso..."
    apptainer run --nv /shared/sifs/latest.sif \
        python -m src.datasets.synthetic_dataset --config "${CONFIG}"
fi

echo ""
echo "Avvio training dentro Apptainer..."
echo ""

# ── Esecuzione dentro container Apptainer ─────────────────────────────────────
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env UNSLOTH_VLLM_STANDBY=0 \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    /shared/sifs/latest.sif \
    python -m src.training --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "============================================"
echo "  Training completato!"
echo "  $(date)"
echo "============================================"
