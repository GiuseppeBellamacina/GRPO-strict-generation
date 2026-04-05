#!/bin/bash
# ============================================================================
# Pulizia selettiva — rimuove checkpoints e logs di un modello specifico.
#
# Uso:
#   bash cluster/clean_model.sh                           # dry-run (mostra cosa c'è)
#   bash cluster/clean_model.sh <MODEL_TAG>              # cancella GRPO (default)
#   bash cluster/clean_model.sh <MODEL_TAG> --all         # cancella tutto
#   bash cluster/clean_model.sh <MODEL_TAG> --baseline    # cancella solo baseline
#   bash cluster/clean_model.sh <MODEL_TAG> --think       # cancella solo variante think
#   bash cluster/clean_model.sh <MODEL_TAG> --nothink     # cancella solo variante nothink
#
# MODEL_TAG è il nome usato nei path (es. smollm2-135m, tinyllama-11b, ...).
# Senza MODEL_TAG: mostra un riepilogo di cosa esiste per ogni modello (dry-run).
#
# Opzioni:
#   --grpo       Solo dati GRPO (default se nessun flag specifico)
#   --baseline   Solo dati baseline
#   --sft        Solo dati SFT
#   --all        GRPO + baseline + SFT
#   --think      Solo variante think (GRPO)
#   --nothink    Solo variante nothink (GRPO)
#
# Esempi:
#   bash cluster/clean_model.sh                            # dry-run
#   bash cluster/clean_model.sh tinyllama-11b              # cancella GRPO
#   bash cluster/clean_model.sh tinyllama-11b --all         # cancella tutto
#   bash cluster/clean_model.sh tinyllama-11b --baseline    # cancella solo baseline
#   bash cluster/clean_model.sh gemma2-2b --grpo --sft     # cancella GRPO + SFT
#   bash cluster/clean_model.sh gemma2-2b --think           # cancella solo variante think
#   bash cluster/clean_model.sh smollm2-135m --nothink      # cancella solo variante nothink
# ============================================================================

set -e
cd "$HOME/GRPO-strict-generation"

# ── Modelli validi (per validazione) ──────────────────────────────────────────
VALID_MODELS=("smollm2-135m" "smollm2-360m" "qwen25-05b" "tinyllama-11b" "gemma2-2b")
THINK_VARIANTS=("nothink" "think")

# Mapping tag → HuggingFace short name (per baseline subfolder)
baseline_hf_name() {
    case "$1" in
        smollm2-135m)  echo "SmolLM2-135M-Instruct" ;;
        smollm2-360m)  echo "SmolLM2-360M-Instruct" ;;
        qwen25-05b)    echo "Qwen2.5-0.5B-Instruct" ;;
        tinyllama-11b) echo "TinyLlama-1.1B-Chat-v1.0" ;;
        gemma2-2b)     echo "gemma-2-2b-it" ;;
        *)             echo "" ;;  # modello custom — non sappiamo il nome HF
    esac
}

# ── Parsing argomenti ─────────────────────────────────────────────────────────
MODEL=""
DO_GRPO=0
DO_BASELINE=0
DO_SFT=0
FILTER_VARIANT=""  # empty = both, "think" or "nothink"

for arg in "$@"; do
    case "$arg" in
        --grpo)     DO_GRPO=1 ;;
        --baseline) DO_BASELINE=1 ;;
        --sft)      DO_SFT=1 ;;
        --all)      DO_GRPO=1; DO_BASELINE=1; DO_SFT=1 ;;
        --think)    FILTER_VARIANT="think" ;;
        --nothink)  FILTER_VARIANT="nothink" ;;
        --help|-h)
            sed -n '2,/^# ====.*===$/p' "$0" | head -n -1 | sed 's/^# \?//'
            echo ""
            echo "Modelli disponibili:"
            for m in "${VALID_MODELS[@]}"; do echo "  $m"; done
            exit 0
            ;;
        -*)
            echo "❌ Opzione sconosciuta: $arg"
            echo "Uso: bash cluster/clean_model.sh <MODEL_TAG> [--grpo|--baseline|--sft|--all]"
            exit 1
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$arg"
            else
                echo "❌ Troppi argomenti posizionali: $arg"
                exit 1
            fi
            ;;
    esac
done

# Default: solo GRPO se nessun flag specifico
if [ $DO_GRPO -eq 0 ] && [ $DO_BASELINE -eq 0 ] && [ $DO_SFT -eq 0 ]; then
    DO_GRPO=1
fi

# ── Dry-run (nessun modello specificato) ──────────────────────────────────────
if [ -z "$MODEL" ]; then
    echo "=== DRY RUN — mostra cosa esiste per ogni modello ==="
    echo ""
    for m in "${VALID_MODELS[@]}"; do
        DIRS_FOUND=()
        # GRPO (nothink + think)
        for variant in "${THINK_VARIANTS[@]}"; do
            [ -d "experiments/checkpoints/grpo/$variant/$m" ] && DIRS_FOUND+=("checkpoints/grpo/$variant/$m")
            [ -d "experiments/logs/grpo/$variant/$m" ] && DIRS_FOUND+=("logs/grpo/$variant/$m")
        done
        # Baseline
        HF=$(baseline_hf_name "$m")
        [ -n "$HF" ] && [ -d "experiments/logs/baseline/$HF" ] && DIRS_FOUND+=("logs/baseline/$HF")
        # SFT
        [ -d "experiments/checkpoints/sft/$m" ] && DIRS_FOUND+=("checkpoints/sft/$m")
        [ -d "experiments/logs/sft/$m" ] && DIRS_FOUND+=("logs/sft/$m")

        if [ ${#DIRS_FOUND[@]} -gt 0 ]; then
            echo "  $m:"
            for d in "${DIRS_FOUND[@]}"; do
                SIZE=$(du -sh "experiments/$d" 2>/dev/null | cut -f1)
                echo "    experiments/$d ($SIZE)"
            done
        else
            echo "  $m: (niente)"
        fi
    done
    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh <MODEL_TAG> [--grpo|--baseline|--sft|--all]"
    exit 0
fi

# ── Validazione modello ──────────────────────────────────────────────────────

FOUND=0
for m in "${VALID_MODELS[@]}"; do
    [ "$m" = "$MODEL" ] && FOUND=1
done
if [ $FOUND -eq 0 ]; then
    echo "⚠️  '$MODEL' non è un modello noto. Modelli validi:"
    for m in "${VALID_MODELS[@]}"; do echo "  $m"; done
    echo ""
    echo "Procedo comunque (potrebbe essere un modello custom)..."
fi

echo "Pulizia modello: $MODEL"
echo "Scope: $([ $DO_GRPO -eq 1 ] && echo 'GRPO ')$([ $DO_BASELINE -eq 1 ] && echo 'BASELINE ')$([ $DO_SFT -eq 1 ] && echo 'SFT ')$([ -n "$FILTER_VARIANT" ] && echo "[variant=$FILTER_VARIANT] ")"
echo ""

CLEANED=0

# ── GRPO ──────────────────────────────────────────────────────────────────────
if [ $DO_GRPO -eq 1 ]; then
    # Determine which variants to clean
    if [ -n "$FILTER_VARIANT" ]; then
        CLEAN_VARIANTS=("$FILTER_VARIANT")
    else
        CLEAN_VARIANTS=("${THINK_VARIANTS[@]}")
    fi
    for variant in "${CLEAN_VARIANTS[@]}"; do
        # Checkpoints: experiments/checkpoints/grpo/<variant>/<model>/
        DIR="experiments/checkpoints/grpo/$variant/$MODEL"
        if [ -d "$DIR" ]; then
            echo "[GRPO/$variant] Checkpoints: $DIR"
            rm -rf "$DIR"
            CLEANED=1
        else
            echo "[GRPO/$variant] Checkpoints: $DIR (non esiste — skip)"
        fi

        # Logs + eval + wandb: experiments/logs/grpo/<variant>/<model>/
        DIR="experiments/logs/grpo/$variant/$MODEL"
        if [ -d "$DIR" ]; then
            echo "[GRPO/$variant] Logs/eval:   $DIR"
            rm -rf "$DIR"
            CLEANED=1
        else
            echo "[GRPO/$variant] Logs/eval:   $DIR (non esiste — skip)"
        fi
    done
fi

# ── Baseline ──────────────────────────────────────────────────────────────────
if [ $DO_BASELINE -eq 1 ]; then
    # Baseline results: experiments/logs/baseline/<HF_short_name>/
    HF_NAME=$(baseline_hf_name "$MODEL")
    if [ -n "$HF_NAME" ]; then
        DIR="experiments/logs/baseline/$HF_NAME"
        if [ -d "$DIR" ]; then
            echo "[BASELINE] Eval: $DIR"
            rm -rf "$DIR"
            CLEANED=1
        else
            echo "[BASELINE] Eval: $DIR (non esiste — skip)"
        fi
    else
        echo "[BASELINE] ⚠️  Modello custom '$MODEL' — nome HF sconosciuto, skip baseline."
    fi
fi

# ── SFT ───────────────────────────────────────────────────────────────────────
if [ $DO_SFT -eq 1 ]; then
    # Checkpoints: experiments/checkpoints/sft/<model>/
    DIR="experiments/checkpoints/sft/$MODEL"
    if [ -d "$DIR" ]; then
        echo "[SFT] Checkpoints: $DIR"
        rm -rf "$DIR"
        CLEANED=1
    else
        echo "[SFT] Checkpoints: $DIR (non esiste — skip)"
    fi

    # Logs: experiments/logs/sft/<model>/
    DIR="experiments/logs/sft/$MODEL"
    if [ -d "$DIR" ]; then
        echo "[SFT] Logs: $DIR"
        rm -rf "$DIR"
        CLEANED=1
    else
        echo "[SFT] Logs: $DIR (non esiste — skip)"
    fi
fi

# ── Riepilogo ─────────────────────────────────────────────────────────────────
echo ""
if [ $CLEANED -eq 1 ]; then
    echo "✅ Pulizia completata per '$MODEL'."
else
    echo "ℹ️  Nessuna cartella da pulire per '$MODEL'."
fi
