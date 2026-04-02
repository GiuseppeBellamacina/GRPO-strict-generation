#!/bin/bash
# ============================================================================
# SLURM batch script — Lancia training + evaluation per più modelli
#
# Tutti i job vengono concatenati in sequenza (un solo job alla volta):
#   train-model1 → eval-model1 → train-model2 → eval-model2 → ...
#
# Questo è necessario quando la QoS permette un solo job attivo.
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
echo "  Multi-model GRPO Pipeline (sequenziale)"
echo "  Date:  $(date)"
echo "  Train: $([ $TRAIN -eq 1 ] && echo 'YES' || echo 'SKIP')"
echo "  Eval:  $([ $EVAL -eq 1 ] && echo 'YES' || echo 'SKIP')"
echo "  Models: ${#MODELS[@]}"
echo "============================================"
echo ""

# ── Lancio job in catena ──────────────────────────────────────────────────────
# Ogni job dipende dal precedente (--dependency=afterany) così gira uno alla volta.
PREV_JOB=""

for entry in "${MODELS[@]}"; do
    TAG="${entry%%:*}"
    CFG="${entry##*:}"

    echo "── $TAG ──────────────────────────────────"
    echo "   Config: $CFG"

    DEP_ARG=""
    if [ -n "$PREV_JOB" ]; then
        DEP_ARG="--dependency=afterany:${PREV_JOB}"
    fi

    if [ $TRAIN -eq 1 ]; then
        TRAIN_JOB=$(CONFIG="$CFG" sbatch --parsable \
            --job-name="train-${TAG}" \
            $DEP_ARG \
            cluster/train.sh)
        echo "   Train job: $TRAIN_JOB$([ -n "$PREV_JOB" ] && echo " (after $PREV_JOB)")"
        PREV_JOB="$TRAIN_JOB"
        DEP_ARG="--dependency=afterok:${TRAIN_JOB}"
    fi

    if [ $EVAL -eq 1 ]; then
        EVAL_JOB=$(CONFIG="$CFG" CURRICULUM=1 sbatch --parsable \
            --job-name="eval-${TAG}" \
            $DEP_ARG \
            cluster/eval.sh)
        echo "   Eval job:  $EVAL_JOB$([ -n "$PREV_JOB" ] && echo " (after $PREV_JOB)")"
        PREV_JOB="$EVAL_JOB"
    fi

    echo ""
done

echo "============================================"
echo "  Catena di ${#MODELS[@]} modelli sottomessa!"
echo "  Ordine: $(for e in "${MODELS[@]}"; do printf "${e%%:*} → "; done | sed 's/ → $//')"
echo "  Usa 'squeue -u \$USER' per monitorare"
echo "============================================"
