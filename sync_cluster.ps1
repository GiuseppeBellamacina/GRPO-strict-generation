param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("upload", "download", "download-logs", "download-checkpoints", "download-wandb", "sync-wandb", "push", "pull")]
    [string]$Action,

    [Parameter(Mandatory = $false)]
    [string]$Path  # relative path for push/pull (e.g. "src/training/grpo_train.py" or "cluster/")
)

$CLUSTER_USER = "bllgpp02h24c351g"
$CLUSTER_HOST = "gcluster.dmi.unict.it"
$REMOTE  = "${CLUSTER_USER}@${CLUSTER_HOST}:~/GRPO-strict-generation"
$SSH_TARGET = "${CLUSTER_USER}@${CLUSTER_HOST}"
$LOCAL   = $PSScriptRoot

function Upload {
    Write-Host "Uploading project to cluster..." -ForegroundColor Cyan

    # Clean __pycache__ before upload
    Write-Progress -Activity "Upload" -Status "Cleaning __pycache__..." -PercentComplete 0
    Get-ChildItem -Path $LOCAL -Directory -Recurse -Filter "__pycache__" |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    # Ensure remote directory structure exists
    Write-Progress -Activity "Upload" -Status "Creating remote directories..." -PercentComplete 2
    ssh $SSH_TARGET "mkdir -p ~/GRPO-strict-generation/experiments/{configs,logs,checkpoints} ~/GRPO-strict-generation/notebooks"

    # Collect all individual files to upload (flatten directories)
    # NOTE: "data" is excluded — the dataset is generated on the cluster by setup.sh/train.sh
    $items = @(
        "src",
        "cluster",
        "experiments/configs",
        ".env"
    )

    # Build flat list of (localPath, remotePath) pairs
    $files = [System.Collections.Generic.List[object]]::new()
    $dirsToClean = [System.Collections.Generic.List[string]]::new()

    foreach ($item in $items) {
        $localPath = Join-Path $LOCAL $item
        if (-not (Test-Path $localPath)) {
            Write-Host "  [SKIP] $item (not found)" -ForegroundColor Yellow
            continue
        }
        if (Test-Path $localPath -PathType Container) {
            # Track top-level dirs for remote cleanup
            $parent = Split-Path $item
            if (-not $parent) {
                $dirsToClean.Add($item)
            }
            # Enumerate all files in directory recursively
            Get-ChildItem -Path $localPath -File -Recurse | ForEach-Object {
                $relPath = $_.FullName.Substring($LOCAL.Length + 1) -replace '\\', '/'
                $files.Add(@{ Local = $_.FullName; Remote = $relPath })
            }
        } else {
            $files.Add(@{ Local = $localPath; Remote = $item })
        }
    }

    # Clean remote top-level dirs before uploading
    if ($dirsToClean.Count -gt 0) {
        $rmCmd = ($dirsToClean | ForEach-Object { "rm -rf ~/GRPO-strict-generation/$_" }) -join "; "
        ssh $SSH_TARGET $rmCmd
        # Recreate the directories
        $mkCmd = ($dirsToClean | ForEach-Object { "mkdir -p ~/GRPO-strict-generation/$_" }) -join "; "
        ssh $SSH_TARGET $mkCmd
    }

    # Ensure all remote subdirectories exist (batch)
    $remoteDirs = $files | ForEach-Object {
        $d = (Split-Path $_.Remote) -replace '\\', '/'
        if ($d) { "~/GRPO-strict-generation/$d" }
    } | Sort-Object -Unique
    if ($remoteDirs.Count -gt 0) {
        $mkdirCmd = "mkdir -p " + ($remoteDirs -join " ")
        ssh $SSH_TARGET $mkdirCmd
    }

    # Upload files one by one with granular progress
    $total = $files.Count
    for ($i = 0; $i -lt $total; $i++) {
        $f = $files[$i]
        $pct = [int](($i / $total) * 100)
        $name = $f.Remote
        Write-Progress -Activity "Upload" `
            -Status "[$($i + 1)/$total] $name" `
            -PercentComplete $pct

        scp -q $f.Local "${REMOTE}/$($f.Remote)"
    }

    Write-Progress -Activity "Upload" -Completed
    Write-Host "Upload complete ($total files)." -ForegroundColor Green
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
    scp -rq "${REMOTE}/experiments/logs/." $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to experiments\logs" -ForegroundColor Gray
}

function DownloadCheckpoints {
    Write-Progress -Activity "Download" -Status "Downloading experiments/checkpoints..." -PercentComplete 0
    $dest = Join-Path $LOCAL "experiments\checkpoints"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    scp -rq "${REMOTE}/experiments/checkpoints/." $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to experiments\checkpoints" -ForegroundColor Gray
}

function DownloadWandb {
    # wandb offline runs are saved inside experiments/logs/ (e.g. experiments/logs/grpo/wandb/)
    Write-Progress -Activity "Download" -Status "Downloading wandb offline runs..." -PercentComplete 0

    # Download from experiments/logs
    $dest = Join-Path $LOCAL "experiments\logs"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    scp -rq "${REMOTE}/experiments/logs/." $dest 2>$null

    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved wandb runs to experiments\logs\" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To sync offline runs to wandb.ai:" -ForegroundColor Yellow
    Write-Host "  .\sync_cluster.ps1 -Action sync-wandb" -ForegroundColor Yellow
}

function SyncWandb {
    Write-Host "Syncing wandb offline runs to wandb.ai..." -ForegroundColor Cyan

    $logsDir = Join-Path $LOCAL "experiments\logs"
    if (-not (Test-Path $logsDir)) {
        Write-Host "No experiments/logs/ found. Run download-wandb first." -ForegroundColor Red
        return
    }

    $runDirs = Get-ChildItem -Path $logsDir -Recurse -Directory -Filter "offline-run-*"
    if ($runDirs.Count -eq 0) {
        Write-Host "No offline runs found in experiments\logs\" -ForegroundColor Yellow
        return
    }

    Write-Host "Found $($runDirs.Count) offline run(s):" -ForegroundColor Gray
    $synced = 0
    $failed = 0
    foreach ($run in $runDirs) {
        $relPath = $run.FullName.Substring($LOCAL.Length + 1)
        Write-Host "  [$($synced + $failed + 1)/$($runDirs.Count)] $relPath" -ForegroundColor Gray -NoNewline
        $result = & wandb sync $run.FullName 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            $synced++
        } else {
            Write-Host " FAILED" -ForegroundColor Red
            $failed++
        }
    }

    Write-Host ""
    Write-Host "Sync complete: $synced succeeded, $failed failed." -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })
}

function Push {
    if (-not $Path) {
        Write-Host "Usage: .\sync_cluster.ps1 -Action push -Path <file-or-folder>" -ForegroundColor Red
        return
    }
    $localPath = Join-Path $LOCAL $Path
    if (-not (Test-Path $localPath)) {
        Write-Host "Not found: $Path" -ForegroundColor Red
        return
    }
    $remotePath = $Path -replace '\\', '/'
    # Ensure remote parent directory exists
    $remoteDir = ($remotePath | Split-Path) -replace '\\', '/'
    if ($remoteDir) {
        ssh $SSH_TARGET "mkdir -p ~/GRPO-strict-generation/$remoteDir"
    }
    if (Test-Path $localPath -PathType Container) {
        # Directory: clean remote and upload
        ssh $SSH_TARGET "rm -rf ~/GRPO-strict-generation/$remotePath; mkdir -p ~/GRPO-strict-generation/$remotePath"
        scp -rq "$localPath/." "${REMOTE}/$remotePath/"
    } else {
        scp -q $localPath "${REMOTE}/$remotePath"
    }
    Write-Host "Pushed $Path -> cluster" -ForegroundColor Green
}

function Pull {
    if (-not $Path) {
        Write-Host "Usage: .\sync_cluster.ps1 -Action pull -Path <file-or-folder>" -ForegroundColor Red
        return
    }
    $remotePath = $Path -replace '\\', '/'
    $localPath = Join-Path $LOCAL $Path
    # Ensure local parent directory exists
    $localDir = Split-Path $localPath
    if ($localDir) {
        New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    }
    scp -rq "${REMOTE}/$remotePath" $localPath
    Write-Host "Pulled $Path <- cluster" -ForegroundColor Green
}

switch ($Action) {
    "upload"                { Upload }
    "download"              { DownloadAll }
    "download-logs"         { DownloadLogs }
    "download-checkpoints"  { DownloadCheckpoints }
    "download-wandb"        { DownloadWandb }
    "sync-wandb"            { SyncWandb }
    "push"                  { Push }
    "pull"                  { Pull }
}
