#!/bin/bash
# ============================================================================
# SLURM batch script — Lancia training + evaluation per più modelli
#
# I job vengono sottomessi in sequenza con dipendenze SLURM:
# per ogni modello, l'eval parte solo quando il training è completato.
# I modelli diversi vengono lanciati in parallelo (job indipendenti).
#
# Uso:
#   bash cluster/run_all.sh               # lancia tutti i modelli
#   bash cluster/run_all.sh --eval-only   # solo evaluation (skip training)
#   bash cluster/run_all.sh --train-only  # solo training (skip eval)
#
# Ogni modello ha la sua config con output_dir e log_dir separati,
# quindi non ci sono conflitti tra i risultati.
# ============================================================================

set -e

# ── Parsing argomenti ─────────────────────────────────────────────────────────
TRAIN=1
EVAL=1
for arg in "$@"; do
    case "$arg" in
        --eval-only)  TRAIN=0 ;;
        --train-only) EVAL=0 ;;
        --help|-h)
            echo "Uso: bash cluster/run_all.sh [--eval-only] [--train-only]"
            exit 0
            ;;
    esac
done

# ── Modelli da lanciare ───────────────────────────────────────────────────────
# Formato: "TAG:CONFIG_PATH"
MODELS=(
    "smollm2-135m:experiments/configs/grpo_smollm2_135m.yaml"
    "smollm2-360m:experiments/configs/grpo_smollm2_360m.yaml"
    "qwen25-05b:experiments/configs/grpo_qwen05.yaml"
    "tinyllama-11b:experiments/configs/grpo_tinyllama.yaml"
    "gemma2-2b:experiments/configs/grpo_gemma2.yaml"
)

echo "============================================"
echo "  Multi-model GRPO Pipeline"
echo "  Date:  $(date)"
echo "  Train: $([ $TRAIN -eq 1 ] && echo 'YES' || echo 'SKIP')"
echo "  Eval:  $([ $EVAL -eq 1 ] && echo 'YES' || echo 'SKIP')"
echo "  Models: ${#MODELS[@]}"
echo "============================================"
echo ""

# ── Lancio job per ogni modello ───────────────────────────────────────────────
for entry in "${MODELS[@]}"; do
    TAG="${entry%%:*}"
    CFG="${entry##*:}"

    echo "── $TAG ──────────────────────────────────"
    echo "   Config: $CFG"

    TRAIN_JOB=""

    if [ $TRAIN -eq 1 ]; then
        TRAIN_JOB=$(CONFIG="$CFG" sbatch --parsable \
            --job-name="train-${TAG}" \
            cluster/train.sh)
        echo "   Train job: $TRAIN_JOB"
    fi

    if [ $EVAL -eq 1 ]; then
        EVAL_ARGS="--job-name=eval-${TAG}"
        if [ -n "$TRAIN_JOB" ]; then
            EVAL_ARGS="$EVAL_ARGS --dependency=afterok:${TRAIN_JOB}"
        fi
        EVAL_JOB=$(CONFIG="$CFG" CURRICULUM=1 sbatch --parsable \
            $EVAL_ARGS \
            cluster/eval.sh)
        echo "   Eval job:  $EVAL_JOB"
        if [ -n "$TRAIN_JOB" ]; then
            echo "   Eval waits for train job $TRAIN_JOB"
        fi
    fi

    echo ""
done

echo "============================================"
echo "  Tutti i job sottomessi!"
echo "  Usa 'squeue -u \$USER' per monitorare"
echo "============================================"
