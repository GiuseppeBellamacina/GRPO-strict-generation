#!/bin/bash
# ============================================================================
# Lancia training + evaluation per più modelli in catena.
#
# La QoS permette un solo job alla volta, quindi un watcher su screen
# controlla ogni 60s se la coda è vuota e sottomette il prossimo job.
#
# Uso:
#   bash cluster/run_all.sh               # lancia tutti i modelli
#   bash cluster/run_all.sh --eval-only   # solo evaluation (skip training)
#   bash cluster/run_all.sh --train-only  # solo training (skip eval)
#
# Monitorare:
#   tmux attach -t grpo-chain             # vedi il watcher live (Ctrl+B D per staccarti)
#   tail -f logs/chain_watcher.log        # log del watcher
#   myjobs                                # job attivo su SLURM
#
# Interrompere:
#   tmux kill-session -t grpo-chain       # uccidi il watcher (il job in corso continua)
#   killalljobs                           # cancella anche il job SLURM attivo
#
# Il watcher si chiude da solo quando la catena è completata.
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

PROJ_DIR="$HOME/GRPO-strict-generation"
CHAIN_FILE="$PROJ_DIR/.job_chain"

# ── Costruisci la catena ──────────────────────────────────────────────────────
# Ogni riga: TYPE:CONFIG:TAG
> "$CHAIN_FILE"  # svuota/crea il file

for entry in "${MODELS[@]}"; do
    TAG="${entry%%:*}"
    CFG="${entry##*:}"

    if [ $TRAIN -eq 1 ]; then
        echo "train:${CFG}:${TAG}" >> "$CHAIN_FILE"
    fi
    if [ $EVAL -eq 1 ]; then
        echo "eval:${CFG}:${TAG}" >> "$CHAIN_FILE"
    fi
done

TOTAL=$(wc -l < "$CHAIN_FILE")

echo "============================================"
echo "  Multi-model GRPO Pipeline (self-chaining)"
echo "  Date:  $(date)"
echo "  Train: $([ $TRAIN -eq 1 ] && echo 'YES' || echo 'SKIP')"
echo "  Eval:  $([ $EVAL -eq 1 ] && echo 'YES' || echo 'SKIP')"
echo "  Models: ${#MODELS[@]}"
echo "  Total jobs: $TOTAL"
echo "============================================"
echo ""
echo "Catena:"
cat -n "$CHAIN_FILE"
echo ""

# ── Avvia il watcher in background ────────────────────────────────────────────
# Il watcher controlla ogni 60s se la coda è vuota e sottomette il prossimo job.
# Usa tmux/screen se disponibile, altrimenti nohup.
mkdir -p logs

if command -v tmux &>/dev/null; then
    tmux kill-session -t grpo-chain 2>/dev/null || true
    tmux new-session -d -s grpo-chain "bash cluster/chain_next.sh 2>&1 | tee -a logs/chain_watcher.log"
    SESSION_TYPE="tmux"
elif command -v screen &>/dev/null; then
    screen -S grpo-chain -X quit 2>/dev/null || true
    screen -dmS grpo-chain bash -c "bash cluster/chain_next.sh 2>&1 | tee -a logs/chain_watcher.log"
    SESSION_TYPE="screen"
else
    nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
    SESSION_TYPE="nohup (PID: $!)"
fi

echo ""
echo "============================================"
echo "  Pipeline avviata!"
echo "  Watcher: $SESSION_TYPE"
echo "  Log: logs/chain_watcher.log"
echo "  Catena: .job_chain"
echo ""
echo "  Per monitorare:"
if [ "$SESSION_TYPE" = "tmux" ]; then
    echo "    tmux attach -t grpo-chain    # vedi watcher (Ctrl+B D per staccarti)"
elif [ "$SESSION_TYPE" = "screen" ]; then
    echo "    screen -r grpo-chain         # vedi watcher (Ctrl+A D per staccarti)"
fi
echo "    tail -f logs/chain_watcher.log"
echo "    myjobs"
echo ""
echo "  Per interrompere:"
if [ "$SESSION_TYPE" = "tmux" ]; then
    echo "    tmux kill-session -t grpo-chain"
elif [ "$SESSION_TYPE" = "screen" ]; then
    echo "    screen -S grpo-chain -X quit"
else
    echo "    kill $!"
fi
echo "============================================"
