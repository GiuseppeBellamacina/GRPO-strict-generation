#!/bin/bash
# ============================================================================
# Lancia training + evaluation per più modelli in catena.
#
# La QoS permette un solo job alla volta, quindi un watcher in background
# controlla ogni 60s se la coda è vuota e sottomette il prossimo job.
#
# Uso:
#   bash cluster/run_all.sh                       # tutti i modelli, train+eval
#   bash cluster/run_all.sh --eval-only            # solo evaluation (skip training)
#   bash cluster/run_all.sh --train-only           # solo training (skip eval)
#   bash cluster/run_all.sh --think                # usa config *_think.yaml
#   bash cluster/run_all.sh --models=1,3,5         # solo modelli 1, 3 e 5
#   bash cluster/run_all.sh --models=1t,2e,3       # 1=solo train, 2=solo eval, 3=entrambi
#   bash cluster/run_all.sh --models=4t,5te        # 4=solo train, 5=train+eval
#   bash cluster/run_all.sh --append --think       # aggiungi think models a pipeline attiva
#   bash cluster/run_all.sh --remove --models=3,5  # rimuovi modelli 3 e 5 dalla coda
#
# Suffissi per --models:
#   (nessuno)  train + eval (default)
#   t          solo train
#   e          solo eval
#   te / et    train + eval (esplicito)
#
# Monitorare:
#   tail -f logs/chain_watcher.log           # log del watcher
#   ps aux | grep chain_next | grep -v grep  # vedi se il watcher è attivo
#   myjobs                                   # job attivo su SLURM
#
# Interrompere:
#   kill $(cat .chain_pid)                   # uccidi il watcher
#   killalljobs                              # cancella anche il job SLURM attivo
#
# Il watcher si chiude da solo quando la catena è completata.
#
# Ogni modello ha la sua config con output_dir e log_dir separati,
# quindi non ci sono conflitti tra i risultati.
# ============================================================================

set -e

# ── Parsing argomenti ─────────────────────────────────────────────────────────
GLOBAL_TRAIN=1
GLOBAL_EVAL=1
ONLY_MODELS=""
USE_THINK=0
RESUME=0
APPEND=0
REMOVE=0
for arg in "$@"; do
    case "$arg" in
        --eval-only)  GLOBAL_TRAIN=0 ;;
        --train-only) GLOBAL_EVAL=0 ;;
        --think)      USE_THINK=1 ;;
        --append)     APPEND=1 ;;
        --remove)     REMOVE=1 ;;
        --resume)     RESUME=1 ;;
        --models=*)   ONLY_MODELS="${arg#--models=}" ;;
        --help|-h)
            echo "Uso: bash cluster/run_all.sh [--eval-only] [--train-only] [--models=SPEC] [--resume] [--append] [--remove]"
            echo ""
            echo "Opzioni globali:"
            echo "  --eval-only      Solo evaluation per tutti (skip training)"
            echo "  --train-only     Solo training per tutti (skip eval)"
            echo "  --think          Usa config *_think.yaml (thinking/reasoning abilitato)"
            echo "  --resume         Riprendi pipeline da dove si era fermata"
            echo "                   Se l'ultimo job era un train, viene rilanciato con --resume"
            echo "                   per riprendere dall'ultimo checkpoint salvato."
            echo "  --append         Aggiungi job alla pipeline attiva senza riavviare il watcher"
            echo "                   (auto-detect se c'è un watcher attivo)"
            echo "  --remove         Rimuovi job dalla pipeline attiva (richiede --models=SPEC)"
            echo ""
            echo "Selezione modelli (--models=SPEC):"
            echo "  SPEC è una lista separata da virgole: INDICE[SUFFISSO],..."
            echo "  Suffissi:  t = solo train,  e = solo eval,  te = entrambi (default)"
            echo ""
            echo "  Esempi:"
            echo "    --models=1,3,5       modelli 1,3,5 con train+eval"
            echo "    --models=1t,2e,3     1=solo train, 2=solo eval, 3=entrambi"
            echo "    --models=4t,5te      4=solo train, 5=train+eval"
            echo ""
            echo "Modelli disponibili:"
            echo "  1: smollm2-135m"
            echo "  2: smollm2-360m"
            echo "  3: qwen25-05b"
            echo "  4: tinyllama-11b"
            echo "  5: gemma2-2b"
            echo ""
            echo "Con --think vengono usate le config *_think.yaml corrispondenti."
            exit 0
            ;;
    esac
done

# ── Modelli da lanciare ───────────────────────────────────────────────────────
# Formato: "TAG:CONFIG_PATH"
if [ "$USE_THINK" -eq 1 ]; then
    MODELS=(
        "smollm2-135m:experiments/configs/grpo_smollm2_135m_think.yaml"
        "smollm2-360m:experiments/configs/grpo_smollm2_360m_think.yaml"
        "qwen25-05b:experiments/configs/grpo_qwen05_think.yaml"
        "tinyllama-11b:experiments/configs/grpo_tinyllama_think.yaml"
        "gemma2-2b:experiments/configs/grpo_gemma2_think.yaml"
    )
else
    MODELS=(
        "smollm2-135m:experiments/configs/grpo_smollm2_135m.yaml"
        "smollm2-360m:experiments/configs/grpo_smollm2_360m.yaml"
        "qwen25-05b:experiments/configs/grpo_qwen05.yaml"
        "tinyllama-11b:experiments/configs/grpo_tinyllama.yaml"
        "gemma2-2b:experiments/configs/grpo_gemma2.yaml"
    )
fi

PROJ_DIR="$HOME/GRPO-strict-generation"
CHAIN_FILE="$PROJ_DIR/.job_chain"
FAILED_FILE="$PROJ_DIR/.chain_failed"

# ── Resume mode ───────────────────────────────────────────────────────────────
if [ "$RESUME" -eq 1 ]; then
    if [ ! -f "$FAILED_FILE" ]; then
        echo "❌ Nessun .chain_failed trovato. Non c'è nulla da riprendere."
        echo "   Usa: bash cluster/run_all.sh [opzioni] (senza --resume) per una nuova pipe."
        exit 1
    fi

    FAILED_JOB=$(cat "$FAILED_FILE")
    FAILED_TYPE=$(echo "$FAILED_JOB" | cut -d: -f1)
    FAILED_CFG=$(echo "$FAILED_JOB" | cut -d: -f2)
    FAILED_TAG=$(echo "$FAILED_JOB" | cut -d: -f3)

    echo "============================================"
    echo "  RESUME Pipeline"
    echo "  Date:      $(date)"
    echo "  Failed job: $FAILED_TYPE $FAILED_TAG"
    echo "  Config:     $FAILED_CFG"
    echo "============================================"
    echo ""

    # Ricostruisci la catena: job fallito (con --resume se train) + eval + rimanenti
    RESUME_CHAIN=$(mktemp)
    if [ "$FAILED_TYPE" = "train" ]; then
        echo "train:${FAILED_CFG}:${FAILED_TAG}:--resume" > "$RESUME_CHAIN"
        # Riaggiungi l'eval che era stato rimosso quando il train è fallito
        echo "eval:${FAILED_CFG}:${FAILED_TAG}" >> "$RESUME_CHAIN"
        echo "→ Training $FAILED_TAG verrà ripreso dall'ultimo checkpoint"
        echo "→ Eval $FAILED_TAG verrà eseguito dopo il train"
    else
        echo "eval:${FAILED_CFG}:${FAILED_TAG}" > "$RESUME_CHAIN"
        echo "→ Eval $FAILED_TAG verrà rieseguito da capo"
    fi

    # Aggiungi job rimanenti se il chain file esiste ancora
    if [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ]; then
        cat "$CHAIN_FILE" >> "$RESUME_CHAIN"
    fi
    mv "$RESUME_CHAIN" "$CHAIN_FILE"
    rm -f "$FAILED_FILE"

    TOTAL=$(wc -l < "$CHAIN_FILE")
    echo ""
    echo "Catena ($TOTAL job):"
    cat -n "$CHAIN_FILE"
    echo ""

    # Uccidi eventuale watcher precedente
    if [ -f .chain_pid ]; then
        OLD_PID=$(cat .chain_pid)
        kill "$OLD_PID" 2>/dev/null && echo "Watcher precedente (PID $OLD_PID) terminato."
        rm -f .chain_pid
    fi

    nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
    WATCHER_PID=$!
    echo "$WATCHER_PID" > .chain_pid

    echo "============================================"
    echo "  Pipeline ripresa!"
    echo "  Watcher PID: $WATCHER_PID"
    echo "  Log: logs/chain_watcher.log"
    echo "============================================"
    exit 0
fi

# ── Funzione helper: controlla se il watcher è attivo ─────────────────────────
_watcher_is_alive() {
    if [ -f "$PROJ_DIR/.chain_pid" ]; then
        local pid
        pid=$(cat "$PROJ_DIR/.chain_pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# ── Auto-detect: se watcher attivo e non --append/--remove esplicito ──────────
if [ "$APPEND" -eq 0 ] && [ "$REMOVE" -eq 0 ] && _watcher_is_alive; then
    echo "⚠️  Pipeline già attiva (watcher PID $(cat "$PROJ_DIR/.chain_pid"))."
    echo "   I nuovi job verranno AGGIUNTI alla coda esistente."
    echo "   (Usa Ctrl-C per annullare, oppure uccidi il watcher prima: watcher-kill)"
    echo ""
    APPEND=1
fi

# ── Remove mode ───────────────────────────────────────────────────────────────
if [ "$REMOVE" -eq 1 ]; then
    if [ ! -f "$CHAIN_FILE" ] || [ ! -s "$CHAIN_FILE" ]; then
        echo "❌ Nessuna catena attiva (.job_chain vuoto o non trovato)."
        exit 1
    fi
    if [ -z "$ONLY_MODELS" ]; then
        echo "❌ --remove richiede --models=SPEC per specificare quali job rimuovere."
        echo "   Esempio: bash cluster/run_all.sh --remove --models=3,5"
        exit 1
    fi

    # Costruisci lista di tag da rimuovere
    REMOVE_TAGS=()
    IFS=',' read -ra SPECS <<< "$ONLY_MODELS"
    for spec in "${SPECS[@]}"; do
        idx=$(echo "$spec" | grep -oP '^\d+')
        i=$((idx - 1))
        if [ $i -lt 0 ] || [ $i -ge ${#MODELS[@]} ]; then
            echo "⚠️  Indice $idx fuori range (1-${#MODELS[@]})"
            continue
        fi
        tag=$(echo "${MODELS[$i]}" | cut -d: -f1)
        REMOVE_TAGS+=("$tag")
    done

    if [ ${#REMOVE_TAGS[@]} -eq 0 ]; then
        echo "❌ Nessun modello valido da rimuovere."
        exit 1
    fi

    echo "Catena attuale:"
    cat -n "$CHAIN_FILE"
    echo ""

    # Filtra le righe che NON contengono i tag da rimuovere
    REMOVED=0
    TEMP_CHAIN=$(mktemp)
    while IFS= read -r line; do
        KEEP=1
        for tag in "${REMOVE_TAGS[@]}"; do
            if echo "$line" | grep -q ":${tag}$\|:${tag}:"; then
                echo "  ✗ Rimosso: $line"
                KEEP=0
                REMOVED=$((REMOVED + 1))
                break
            fi
        done
        [ "$KEEP" -eq 1 ] && echo "$line" >> "$TEMP_CHAIN"
    done < "$CHAIN_FILE"

    if [ "$REMOVED" -eq 0 ]; then
        echo "⚠️  Nessun job corrispondente trovato nella catena."
        rm -f "$TEMP_CHAIN"
        exit 0
    fi

    mv "$TEMP_CHAIN" "$CHAIN_FILE"
    [ ! -s "$CHAIN_FILE" ] && rm -f "$CHAIN_FILE"

    REMAINING=$([ -f "$CHAIN_FILE" ] && wc -l < "$CHAIN_FILE" || echo 0)
    echo ""
    echo "✅ Rimossi $REMOVED job. Rimanenti: $REMAINING"
    [ -f "$CHAIN_FILE" ] && echo "" && echo "Catena aggiornata:" && cat -n "$CHAIN_FILE"
    exit 0
fi

# ── Filtra modelli se --models è specificato ──────────────────────────────────
# Ogni entry diventa "TAG:CONFIG:do_train:do_eval"
SELECTED=()

if [ -n "$ONLY_MODELS" ]; then
    IFS=',' read -ra SPECS <<< "$ONLY_MODELS"
    for spec in "${SPECS[@]}"; do
        # Separa indice numerico e suffisso lettere: "3te" → idx=3, suffix="te"
        idx=$(echo "$spec" | grep -oP '^\d+')
        suffix=$(echo "$spec" | grep -oP '[a-zA-Z]+$' || true)
        i=$((idx - 1))  # 1-based → 0-based

        if [ $i -lt 0 ] || [ $i -ge ${#MODELS[@]} ]; then
            echo "⚠️  Indice $idx fuori range (1-${#MODELS[@]})"
            continue
        fi

        # Determina train/eval per questo modello
        case "$suffix" in
            t)   do_t=1; do_e=0 ;;
            e)   do_t=0; do_e=1 ;;
            te|et) do_t=1; do_e=1 ;;
            "")  do_t=$GLOBAL_TRAIN; do_e=$GLOBAL_EVAL ;;
            *)   echo "⚠️  Suffisso sconosciuto '$suffix' per modello $idx (usa t/e/te)"; continue ;;
        esac
        SELECTED+=("${MODELS[$i]}:${do_t}:${do_e}")
    done
else
    # Nessun filtro: tutti i modelli con flag globali
    for entry in "${MODELS[@]}"; do
        SELECTED+=("${entry}:${GLOBAL_TRAIN}:${GLOBAL_EVAL}")
    done
fi

# ── Costruisci la catena ──────────────────────────────────────────────────────
# Ogni riga: TYPE:CONFIG:TAG
if [ "$APPEND" -eq 0 ]; then
    > "$CHAIN_FILE"  # svuota/crea il file (new pipeline)
fi
# In append mode, il file esiste già e ci aggiungiamo in coda

NEW_JOBS=0
for sel in "${SELECTED[@]}"; do
    # sel = "TAG:CONFIG:do_train:do_eval"
    TAG=$(echo "$sel" | cut -d: -f1)
    CFG=$(echo "$sel" | cut -d: -f2)
    DO_TRAIN=$(echo "$sel" | cut -d: -f3)
    DO_EVAL=$(echo "$sel" | cut -d: -f4)

    if [ "$DO_TRAIN" -eq 1 ]; then
        echo "train:${CFG}:${TAG}" >> "$CHAIN_FILE"
        NEW_JOBS=$((NEW_JOBS + 1))
    fi
    if [ "$DO_EVAL" -eq 1 ]; then
        echo "eval:${CFG}:${TAG}" >> "$CHAIN_FILE"
        NEW_JOBS=$((NEW_JOBS + 1))
    fi
    MODEL_COUNT=$((MODEL_COUNT + 1))
done

TOTAL=$(wc -l < "$CHAIN_FILE")

THINK_LABEL="off"
[ "$USE_THINK" -eq 1 ] && THINK_LABEL="on"

if [ "$APPEND" -eq 1 ]; then
    echo "============================================"
    echo "  Jobs aggiunti alla pipeline attiva"
    echo "  Date:  $(date)"
    echo "  Nuovi: $NEW_JOBS job ($MODEL_COUNT modelli)"
    echo "  Think: $THINK_LABEL"
    echo "  Totale in coda: $TOTAL"
    echo "============================================"
    echo ""
    echo "Catena completa:"
    cat -n "$CHAIN_FILE"
    echo ""
    echo "✅ Il watcher (PID $(cat "$PROJ_DIR/.chain_pid")) li eseguirà automaticamente."
    exit 0
fi

echo "============================================"
echo "  Multi-model GRPO Pipeline (self-chaining)"
echo "  Date:  $(date)"
echo "  Models: $MODEL_COUNT"
echo "  Think:  $THINK_LABEL"
echo "  Total jobs: $TOTAL"
echo "============================================"
echo ""
echo "Catena:"
cat -n "$CHAIN_FILE"
echo ""

# ── Avvia il watcher in background (nohup) ────────────────────────────────────
# Il watcher controlla ogni 60s se la coda è vuota e sottomette il prossimo job.
mkdir -p logs

# Uccidi eventuale watcher precedente
if [ -f .chain_pid ]; then
    OLD_PID=$(cat .chain_pid)
    kill "$OLD_PID" 2>/dev/null && echo "Watcher precedente (PID $OLD_PID) terminato."
    rm -f .chain_pid
fi

nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
WATCHER_PID=$!
echo "$WATCHER_PID" > .chain_pid

echo ""
echo "============================================"
echo "  Pipeline avviata!"
echo "  Watcher PID: $WATCHER_PID"
echo "  Log: logs/chain_watcher.log"
echo "  Catena: .job_chain"
echo ""
echo "  Per monitorare:"
echo "    tail -f logs/chain_watcher.log"
echo "    myjobs"
echo ""
echo "  Per interrompere:"
echo "    kill \$(cat .chain_pid)   # uccidi il watcher"
echo "    killalljobs              # cancella il job SLURM attivo"
echo "============================================"
