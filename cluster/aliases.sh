#!/bin/bash
# ============================================================================
# Alias utili per il cluster DMI
#
# Uso:
#   source cluster/aliases.sh
#
# Per caricarli automaticamente, aggiungi al tuo ~/.bashrc:
#   source ~/GRPO-strict-generation/cluster/aliases.sh
# ============================================================================

PROJ_DIR="$HOME/GRPO-strict-generation"

# ── Job management ───────────────────────────────────────────────────────────

# Controlla i miei job attivi
alias myjobs='squeue --me --format="%.10i %.20j %.8T %.10M %.6D %.20R %o"'

# Info dettagliata su un job (uso: jobinfo <JOB_ID>)
jobinfo() {
    if [ -z "$1" ]; then
        echo "Uso: jobinfo <JOB_ID>"
        return 1
    fi
    scontrol show job "$1"
}

# Cancella un job (uso: killjob <JOB_ID>)
alias killjob='scancel'

# Cancella tutti i miei job
alias killalljobs='scancel --me'

# ── Log monitoring ───────────────────────────────────────────────────────────

# Segui il log di un job di training (uso: trainlog <JOB_ID>)
trainlog() {
    if [ -z "$1" ]; then
        echo "Uso: trainlog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-train-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Segui il log di un job di eval GRPO (uso: evallog <JOB_ID>)
evallog() {
    if [ -z "$1" ]; then
        echo "Uso: evallog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-eval-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Segui il log di un job baseline (uso: baselog <JOB_ID>)
baselog() {
    if [ -z "$1" ]; then
        echo "Uso: baselog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-baseline-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Mostra l'ultimo log (qualsiasi tipo) — uso: lastlog
lastlog() {
    local logfile
    logfile=$(ls -t "$PROJ_DIR"/logs/slurm*.log 2>/dev/null | head -1)
    if [ -z "$logfile" ]; then
        echo "Nessun log trovato in $PROJ_DIR/logs/"
        return 1
    fi
    echo "==> $logfile <=="
    tail -f "$logfile"
}

# ── Filesystem ───────────────────────────────────────────────────────────────

# Tree ricorsivo di una cartella (uso: tree <DIR> [DEPTH])
tree() {
    local dir="${1:-.}"
    local depth="${2:-3}"
    find "$dir" -maxdepth "$depth" | sed -e "s|[^/]*/|  |g" -e "s|  |├─|"
}

# Alias più leggibile basato su ls -R (uso: ltree <DIR>)
ltree() {
    local dir="${1:-.}"
    ls -R "$dir" | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//──/g' -e 's/^/  /' -e 's/─/│/'
}

# ── GPU & risorse ────────────────────────────────────────────────────────────

# Stato GPU — funziona solo dentro un job SLURM con GPU allocata
alias gpu='nvidia-smi'

# Uso disco del progetto
alias quota='quota -s'

# ── Quick commands ───────────────────────────────────────────────────────────

# Vai alla directory del progetto
alias proj='cd "$PROJ_DIR"'

# Mostra i checkpoint disponibili
ckpts() {
    local base="$PROJ_DIR/experiments/checkpoints/grpo"
    for model_dir in "$base"/*/; do
        [ -d "$model_dir" ] || continue
        local model=$(basename "$model_dir")
        echo "=== $model ==="
        # Checkpoint intermedi (per stage)
        for d in "$model_dir"stage_*/; do
            if [ -d "$d" ]; then
                echo "  $(basename "$d"):"
                ls -d "$d"checkpoint-* 2>/dev/null | while read -r c; do echo "    $(basename "$c")"; done
            fi
        done
        # Modelli finali (stages/)
        if [ -d "$model_dir/stages" ]; then
            echo "  stages/:"
            ls -d "$model_dir"stages/stage_*/ 2>/dev/null | while read -r s; do
                echo "    $(basename "$s")"
            done
        fi
        echo ""
    done
}

# Mostra tabella training log (uso: trainlog-table [PATH] [--tail N])
# PATH default: ultimo checkpoint in experiments/checkpoints/grpo/
trainlog-table() {
    cd "$PROJ_DIR" && python3 -m src.utils.show_training_log "${1:-experiments/checkpoints/grpo}" "${@:2}"
}

# Segui training live come tabella (uso: trainlog-live <JOB_ID>)
trainlog-live() {
    if [ -z "$1" ]; then
        echo "Uso: trainlog-live <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-train-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile" | python3 -m src.utils.live_training_table
}

# Lancia training (uso: train --config PATH [extra args...])
train() {
    local config=""
    local extra_args=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            *) extra_args="$extra_args $1"; shift ;;
        esac
    done
    if [ -z "$config" ]; then
        echo "Uso: train --config PATH [extra args...]"
        echo ""
        echo "Config disponibili:"
        ls -1 "$PROJ_DIR"/experiments/configs/grpo_*.yaml 2>/dev/null | xargs -I{} basename {} | sed 's/^/  /'
        return 1
    fi
    cd "$PROJ_DIR" && CONFIG="$config" EXTRA_ARGS="$extra_args" sbatch cluster/train.sh
}

# Lancia eval (uso: run-eval --config PATH [--compare] [--curriculum] [--checkpoint PATH])
run-eval() {
    local config=""
    local compare=0
    local curriculum=0
    local checkpoint=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --compare) compare=1; shift ;;
            --curriculum) curriculum=1; shift ;;
            --checkpoint) checkpoint="$2"; shift 2 ;;
            *) echo "Argomento sconosciuto: $1"; return 1 ;;
        esac
    done
    if [ -z "$config" ]; then
        echo "Uso: run-eval --config PATH [--compare] [--curriculum] [--checkpoint PATH]"
        echo ""
        echo "Config disponibili:"
        ls -1 "$PROJ_DIR"/experiments/configs/grpo_*.yaml "$PROJ_DIR"/experiments/configs/baseline.yaml 2>/dev/null | xargs -I{} basename {} | sed 's/^/  /'
        return 1
    fi
    cd "$PROJ_DIR" && CONFIG="$config" COMPARE="$compare" CURRICULUM="$curriculum" CHECKPOINT="$checkpoint" sbatch cluster/eval.sh
}

# Lancia tutti i modelli (train + eval curriculum)
# Uso: run-all [--eval-only] [--train-only]
run-all() {
    cd "$PROJ_DIR" && bash cluster/run_all.sh "$@"
}

# ── Meta ─────────────────────────────────────────────────────────────────────

# Lista di tutti i comandi custom registrati
_GRPO_ALIASES="myjobs jobinfo killjob killalljobs trainlog evallog baselog lastlog tree ltree gpu quota proj ckpts trainlog-table trainlog-live train run-eval run-all claudio unload-aliases install-aliases uninstall-aliases"

# Mostra i comandi disponibili
claudio() {
    echo "Comandi GRPO disponibili:"
    echo "   myjobs            — lista job attivi"
    echo "   jobinfo <ID>      — dettagli job"
    echo "   killjob <ID>      — cancella job"
    echo "   killalljobs       — cancella tutti i miei job"
    echo "   trainlog <ID>     — segui log training"
    echo "   evallog <ID>      — segui log eval"
    echo "   baselog <ID>      — segui log baseline"
    echo "   lastlog           — segui l'ultimo log"
    echo "   tree <DIR> [N]    — albero cartelle (profondità N)"
    echo "   ltree <DIR>       — albero cartelle compatto"
    echo "   gpu               — stato GPU"
    echo "   quota             — uso disco progetto"
    echo "   proj              — cd al progetto"
    echo "   ckpts             — mostra checkpoint"
    echo "   trainlog-table [PATH] [--tail N]"
    echo "                     — tabella metriche training"
    echo "   trainlog-live <ID> — training live come tabella"
    echo ""
    echo "   train --config PATH [extra args...]"
    echo "                     — lancia training"
    echo "   run-eval --config PATH [--compare] [--curriculum] [--checkpoint PATH]"
    echo "                     — lancia evaluation"
    echo "   run-all [--eval-only] [--train-only]"
    echo "                     — lancia train+eval per tutti i modelli"
    echo ""
    echo "   claudio           — mostra questo messaggio"
    echo "   unload-aliases    — rimuovi alias (sessione corrente)"
    echo "   install-aliases   — aggiungi alias al .bashrc (permanente)"
    echo "   uninstall-aliases — rimuovi alias dal .bashrc (permanente)"
}

# Rimuovi tutti gli alias e funzioni custom (solo sessione corrente)
unload-aliases() {
    for cmd in $_GRPO_ALIASES; do
        unalias "$cmd" 2>/dev/null
        unset -f "$cmd" 2>/dev/null
    done
    unset _GRPO_ALIASES PROJ_DIR
    echo "✅ Alias GRPO rimossi (sessione corrente)."
}

_ALIASES_SOURCE_LINE="source ~/GRPO-strict-generation/cluster/aliases.sh"

# Aggiungi alias al .bashrc (caricati ad ogni login)
install-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        echo "⚠️  Alias già presenti in ~/.bashrc"
    else
        echo "$_ALIASES_SOURCE_LINE" >> ~/.bashrc
        echo "✅ Alias aggiunti a ~/.bashrc (attivi dal prossimo login)"
    fi
}

# Rimuovi alias dal .bashrc (non più caricati al login)
uninstall-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        sed -i "\|$_ALIASES_SOURCE_LINE|d" ~/.bashrc
        echo "✅ Alias rimossi da ~/.bashrc"
    else
        echo "⚠️  Alias non presenti in ~/.bashrc"
    fi
}

echo "✅ Alias GRPO caricati. Digita 'claudio' per la lista comandi."
