#!/bin/bash
# ============================================================================
# Watcher — Esegue i job dalla catena .job_chain uno alla volta.
#
# Gira sul login node (NON dentro un job SLURM). Controlla ogni 60s
# se la coda è vuota e sottomette il prossimo job.
#
# Uso:
#   nohup bash cluster/chain_next.sh &     # lancia in background
#   bash cluster/chain_next.sh             # lancia in foreground
#
# Per interrompere: kill %1, oppure cancella .job_chain
# ============================================================================

PROJ_DIR="$HOME/GRPO-strict-generation"
CHAIN_FILE="$PROJ_DIR/.job_chain"
POLL_INTERVAL=60  # secondi tra un check e l'altro

cd "$PROJ_DIR"

echo "[chain] Watcher avviato — $(date)"
echo "[chain] File catena: $CHAIN_FILE"

while true; do
    # Se non c'è più il file catena, abbiamo finito
    if [ ! -f "$CHAIN_FILE" ] || [ ! -s "$CHAIN_FILE" ]; then
        echo "[chain] ✅ Pipeline completata! — $(date)"
        rm -f "$CHAIN_FILE"
        exit 0
    fi

    # Controlla se ci sono job attivi (RUNNING o PENDING)
    ACTIVE=$(squeue --me --noheader 2>/dev/null | wc -l)
    if [ "$ACTIVE" -gt 0 ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Coda vuota — sottometti il prossimo job
    NEXT=$(head -1 "$CHAIN_FILE")

    # Rimuovi la prima riga
    tail -n +2 "$CHAIN_FILE" > "$CHAIN_FILE.tmp" && mv "$CHAIN_FILE.tmp" "$CHAIN_FILE"
    if [ ! -s "$CHAIN_FILE" ]; then
        rm -f "$CHAIN_FILE"
    fi

    # Parsing: TYPE:CONFIG:TAG
    TYPE="${NEXT%%:*}"
    REST="${NEXT#*:}"
    CFG="${REST%%:*}"
    TAG="${REST##*:}"

    REMAINING=$([ -f "$CHAIN_FILE" ] && wc -l < "$CHAIN_FILE" || echo 0)
    echo "[chain] Sottometto: $TYPE $TAG ($CFG) — $REMAINING rimanenti — $(date)"

    case "$TYPE" in
        train)
            CONFIG="$CFG" sbatch --job-name="train-${TAG}" cluster/train.sh
            ;;
        eval)
            CONFIG="$CFG" CURRICULUM=1 sbatch --job-name="eval-${TAG}" cluster/eval.sh
            ;;
        *)
            echo "[chain] ❌ Tipo sconosciuto: $TYPE — skip"
            continue
            ;;
    esac

    # Aspetta un po' prima di ricontrollare (il job appena sottomesso
    # potrebbe impiegare qualche secondo per apparire in squeue)
    sleep 10
done
