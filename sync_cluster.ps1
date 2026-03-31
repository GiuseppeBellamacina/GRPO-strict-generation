param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("upload", "download", "download-logs", "download-checkpoints", "download-wandb")]
    [string]$Action
)

$USER    = "bllgpp02h24c351g"
$HOST    = "gcluster.dmi.unict.it"
$REMOTE  = "${USER}@${HOST}:~/GRPO-strict-generation"
$LOCAL   = $PSScriptRoot

function Upload {
    Write-Host "Uploading project to cluster..." -ForegroundColor Cyan

    # Ensure remote directory structure exists
    Write-Progress -Activity "Upload" -Status "Creating remote directories..." -PercentComplete 0
    ssh "${USER}@${HOST}" "mkdir -p ~/GRPO-strict-generation/experiments/{configs,logs,checkpoints}"

    $items = @(
        "src",
        "cluster",
        "data",
        "docs",
        "tests",
        "experiments/configs",
        "pyproject.toml",
        "setup.sh",
        "Dockerfile",
        "docker-compose.yml",
        "format.sh",
        "README.md"
    )

    $total = $items.Count
    for ($i = 0; $i -lt $total; $i++) {
        $item = $items[$i]
        $pct  = [int](($i / $total) * 100)
        Write-Progress -Activity "Upload" `
            -Status "[$($i + 1)/$total] $item" `
            -PercentComplete $pct

        $localPath = Join-Path $LOCAL $item
        if (Test-Path $localPath) {
            scp -r $localPath "${REMOTE}/$item"
        } else {
            Write-Host "  [SKIP] $item (not found)" -ForegroundColor Yellow
        }
    }

    Write-Progress -Activity "Upload" -Completed
    Write-Host "Upload complete." -ForegroundColor Green
}

function DownloadAll {
    Write-Host "Downloading all outputs from cluster..." -ForegroundColor Cyan

    Write-Progress -Activity "Download" -Status "[1/3] experiments/logs..." -PercentComplete 0
    DownloadLogs

    Write-Progress -Activity "Download" -Status "[2/3] experiments/checkpoints..." -PercentComplete 33
    DownloadCheckpoints

    Write-Progress -Activity "Download" -Status "[3/3] wandb offline runs..." -PercentComplete 66
    DownloadWandb

    Write-Progress -Activity "Download" -Completed
    Write-Host "Download complete." -ForegroundColor Green
}

function DownloadLogs {
    # Includes figures/ subfolders inside each experiment dir (e.g. experiments/logs/baseline/figures/)
    Write-Progress -Activity "Download" -Status "Downloading experiments/logs (includes figures)..." -PercentComplete 0
    $dest = Join-Path $LOCAL "experiments\logs"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    scp -r "${REMOTE}/experiments/logs/." $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to experiments\logs" -ForegroundColor Gray
}

function DownloadCheckpoints {
    Write-Progress -Activity "Download" -Status "Downloading experiments/checkpoints..." -PercentComplete 0
    $dest = Join-Path $LOCAL "experiments\checkpoints"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    scp -r "${REMOTE}/experiments/checkpoints/." $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to experiments\checkpoints" -ForegroundColor Gray
}

function DownloadWandb {
    # wandb offline saves to ~/GRPO-strict-generation/wandb/ on the cluster.
    # This folder is created automatically during training when WANDB_MODE=offline.
    Write-Progress -Activity "Download" -Status "Downloading wandb offline runs..." -PercentComplete 0
    $dest = Join-Path $LOCAL "wandb"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    scp -r "${REMOTE}/wandb/." $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to wandb\" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To sync offline runs to wandb.ai:" -ForegroundColor Yellow
    Write-Host "  wandb sync wandb\offline-run-*" -ForegroundColor Yellow
}

switch ($Action) {
    "upload"                { Upload }
    "download"              { DownloadAll }
    "download-logs"         { DownloadLogs }
    "download-checkpoints"  { DownloadCheckpoints }
    "download-wandb"        { DownloadWandb }
}
