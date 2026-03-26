#!/usr/bin/env pwsh
# Format and lint all code with Isort, Black and Ruff

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Code Formatting & Linting" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if environment exists and activate it
if (Test-Path -Path ".venv") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "⚠️  Virtual environment not found" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Run Isort
Write-Host "🎨 Running Isort..." -ForegroundColor Yellow
isort .
$isortExit = $LASTEXITCODE

if ($isortExit -eq 0) {
    Write-Host "✅ Isort completed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Isort failed with exit code $isortExit" -ForegroundColor Red
}

Write-Host ""

# Run Black
Write-Host "🎨 Running Black formatter..." -ForegroundColor Yellow
black .
$blackExit = $LASTEXITCODE

if ($blackExit -eq 0) {
    Write-Host "✅ Black formatting completed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Black formatting failed with exit code $blackExit" -ForegroundColor Red
}

Write-Host ""

# Run Ruff
Write-Host "🔍 Running Ruff linter with auto-fix..." -ForegroundColor Yellow
ruff check --fix .
$ruffExit = $LASTEXITCODE

if ($ruffExit -eq 0) {
    Write-Host "✅ Ruff linting completed successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  Ruff found issues (exit code $ruffExit)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Formatting Complete!" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
