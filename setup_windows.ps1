# Setup ambiente GRPO Strict Generation
# Installa dipendenze da pyproject.toml

Write-Host "=== Setup GRPO Strict Generation ===" -ForegroundColor Cyan

# Step 1: Verifica UV
Write-Host "`nVerifica UV..." -ForegroundColor Yellow

$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvInstalled) {
    Write-Host "UV non installato. Installazione..." -ForegroundColor Yellow
    pip install uv
    Write-Host "✅ UV installato!" -ForegroundColor Green
} else {
    Write-Host "✅ UV installato" -ForegroundColor Green
}

# Step 2: Rimuovi .venv esistente se presente
if (Test-Path ".venv") {
    Write-Host "`n⚠️  .venv esistente trovato. Rimuovere? (y/n)" -ForegroundColor Yellow
    $remove = Read-Host
    if ($remove -eq "y") {
        Remove-Item .venv -Recurse -Force
        Write-Host "✅ .venv rimosso" -ForegroundColor Green
    }
}

# Step 3: Crea venv
Write-Host "`nCreazione ambiente virtuale..." -ForegroundColor Yellow
uv venv
Write-Host "✅ Ambiente creato" -ForegroundColor Green

# Step 4: Attiva ambiente
Write-Host "`nAttivazione ambiente..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Step 5: Sincronizza dipendenze + installa progetto in editable mode
Write-Host "`nSincronizzazione dipendenze da pyproject.toml (uv sync)..." -ForegroundColor Cyan
uv sync --extra dev
Write-Host "✅ Dipendenze sincronizzate + progetto installato in editable mode" -ForegroundColor Green

# Step 8: Verifica installazione
Write-Host "`nVerifica installazione..." -ForegroundColor Yellow

$verifyScript = @"
import torch

print('\n' + '='*60)
print('AMBIENTE GRPO STRICT GENERATION')
print('='*60)

print(f'\n🔥 PyTorch:')
print(f'   Version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Device: {torch.cuda.get_device_name(0)}')

print(f'\n📦 Librerie:')
try:
    import transformers
    print(f'   ✓ Transformers: {transformers.__version__}')
except: pass

try:
    import trl
    print(f'   ✓ TRL: {trl.__version__}')
except: pass

try:
    import peft
    print(f'   ✓ PEFT: {peft.__version__}')
except: pass

try:
    import datasets
    print(f'   ✓ Datasets: {datasets.__version__}')
except: pass

try:
    import accelerate
    print(f'   ✓ Accelerate: {accelerate.__version__}')
except: pass

try:
    import tensorboard
    print(f'   ✓ TensorBoard installato')
except: pass

try:
    from src import training, evaluation, datasets as ds
    print(f'   ✓ src package importabile (editable mode)')
except: print(f'   ✗ src package NON importabile')

print('\n' + '='*60)
"@

python -c $verifyScript

# Info finali
Write-Host "`n=== Setup Completato! ===" -ForegroundColor Cyan
Write-Host "✅ Dipendenze installate da pyproject.toml" -ForegroundColor Green
Write-Host "✅ Progetto installato in editable mode (src importabile)" -ForegroundColor Green
