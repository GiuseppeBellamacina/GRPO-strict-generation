# Quick Setup — Cluster DMI UniCT

Guida rapida step-by-step. Per dettagli completi vedi [CLUSTER.md](CLUSTER.md).

---

## 1. SSH Key (Windows, una tantum)

```powershell
# Verifica se esiste già una chiave
Test-Path "$env:USERPROFILE\.ssh\id_ed25519"

# Se non esiste, creala:
ssh-keygen -t ed25519

# Copia la chiave sul cluster:
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | `
  ssh <CODICE-FISCALE>@gcluster.dmi.unict.it `
  "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Da qui in poi non serve più la password:

```powershell
ssh <CODICE-FISCALE>@gcluster.dmi.unict.it
```

---

## 2. Scopri la tua queue

```bash
sacctmgr show associations user=$USER format=account,qos,defaultqos -P
```

Output di esempio:

```
Account|QOS|DefaultQOS
dl-course-q2|gpu-medium,gpu-large,gpu-xlarge|gpu-medium
```

Prendi nota di `Account` (es. `dl-course-q2`) e dei QoS disponibili.

---

## 3. Crea la struttura remota (una tantum)

```bash
# Sul cluster (dal nodo di login):
mkdir -p ~/GRPO-strict-generation/experiments/{configs,logs,checkpoints}
```

---

## 4. Carica i file (da Windows)

```powershell
# Nella root del progetto locale:
.\sync_cluster.ps1 -Action upload
```

---

## 5. Setup ambiente (una tantum)

### 5.1. Apri una sessione interattiva con GPU

```bash
srun --account dl-course-q2 --partition dl-course-q2 --qos gpu-medium \
     --gres=gpu:1 --gres=shard:5000 --mem=8G \
     --pty apptainer shell --nv /shared/sifs/latest.sif
```

> Sostituisci `dl-course-q2` e `gpu-medium` con i tuoi valori del passo 2.

### 5.2. Installa dipendenze

```bash
cd ~/GRPO-strict-generation
bash cluster/setup.sh
```

Lo script rileva la GPU, installa i pacchetti, genera il dataset e verifica.

### 5.3. Esci dalla sessione interattiva

```bash
exit
```

---

## 6. Lancia il training

### 6.1. Pipeline completa (tutti i modelli)

```bash
cd ~/GRPO-strict-generation
run-all                    # train + eval per tutti e 5 i modelli
```

Oppure seleziona modelli specifici:

```bash
run-all --models=1,3,5     # solo SmolLM2-135M, Qwen2.5-0.5B, Gemma-2-2B
```

### 6.2. Singolo modello (manuale)

```bash
sbatch cluster/train.sh experiments/configs/grpo_tinyllama.yaml
```

---

## 7. Monitora la pipeline

```bash
# Dashboard live (compatta)
monitor

# Dashboard con tabella completa
monitor --tab

# Fallback: comandi SLURM diretti
squeue -u $USER
tail -f logs/slurm-<JOB_ID>.log
scancel <JOB_ID>
```

---

## 8. Resume da checkpoint

Se il job va in timeout (max 12h) e il training non è finito:

```bash
# Modifica cluster/train.sh:
EXTRA_ARGS="--resume"

# Rilancia:
sbatch cluster/train.sh
```

---

## 9. Scarica i risultati (da Windows)

```powershell
.\sync_cluster.ps1 -Action download              # tutto
.\sync_cluster.ps1 -Action download-logs          # solo logs + figure
.\sync_cluster.ps1 -Action download-checkpoints   # solo checkpoint LoRA
.\sync_cluster.ps1 -Action download-wandb         # solo run wandb offline
```

### Sync wandb (locale)

```powershell
wandb sync wandb\offline-run-*
```

---

## Riepilogo QoS

| QoS | CPU | RAM | GPU VRAM | Tempo max |
|-----|-----|-----|----------|-----------|
| gpu-medium | 2 | 8 GB | 5.5 GB | 6h |
| gpu-large | 4 | 16 GB | 11 GB | 12h |
| gpu-xlarge | 8 | 48 GB | 22.5 GB | 12h |

> Per il **setup** basta `gpu-medium`.
> Per il **training GRPO** serve almeno `gpu-large`, consigliato `gpu-xlarge`.
> Per la **baseline eval** basta `gpu-large`.
