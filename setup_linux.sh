#!/bin/bash
# Setup per Google Colab / Linux remoto
# Installa dipendenze + progetto in editable mode

set -e

echo "=== Setup GRPO Strict Generation (Colab) ==="

# Step 1: Installa uv (più veloce di pip)
echo ""
echo "📦 Installazione uv..."
pip install -q uv
echo "✅ uv installato"

# Step 2: Sincronizza dipendenze + installa progetto in editable mode
echo ""
echo "📦 Sincronizzazione dipendenze (uv sync)..."
uv sync
echo "✅ Dipendenze sincronizzate + progetto installato in editable mode"

# Step 4: Verifica installazione
echo ""
echo "🔍 Verifica installazione..."

uv run python3 -c "
import torch

print()
print('='*60)
print('AMBIENTE GRPO STRICT GENERATION')
print('='*60)

print(f'\n🔥 PyTorch:')
print(f'   Version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

print(f'\n📦 Librerie:')
try:
    import transformers
    print(f'   ✓ Transformers: {transformers.__version__}')
except: print('   ✗ Transformers NON installato')

try:
    import trl
    print(f'   ✓ TRL: {trl.__version__}')
except: print('   ✗ TRL NON installato')

try:
    import peft
    print(f'   ✓ PEFT: {peft.__version__}')
except: print('   ✗ PEFT NON installato')

try:
    import datasets
    print(f'   ✓ Datasets: {datasets.__version__}')
except: print('   ✗ Datasets NON installato')

try:
    import accelerate
    print(f'   ✓ Accelerate: {accelerate.__version__}')
except: print('   ✗ Accelerate NON installato')

try:
    from src import training, evaluation, datasets as ds
    print(f'   ✓ src package importabile (editable mode)')
except: print('   ✗ src package NON importabile')

print()
print('='*60)
"

echo ""
echo "=== Setup Completato! ==="
echo "✅ Dipendenze installate da pyproject.toml"
echo "✅ Progetto installato in editable mode"
