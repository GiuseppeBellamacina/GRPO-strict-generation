# Comandi SLURM — Guida Rapida

Riferimento rapido per l'uso del cluster con SLURM.

---

## Indice

1. [Sottomissione Job](#1-sottomissione-job)
2. [Monitoraggio Job](#2-monitoraggio-job)
3. [Cancellazione Job](#3-cancellazione-job)
4. [Informazioni sul Cluster](#4-informazioni-sul-cluster)
5. [Sessioni Interattive](#5-sessioni-interattive)
6. [Apptainer (Container)](#6-apptainer-container)
7. [Debug GPU](#7-debug-gpu)
8. [Trasferimento File](#8-trasferimento-file)
9. [Workflow Completo di Esempio](#9-workflow-completo-di-esempio)

---

## 1. Sottomissione Job

### Sottomettere uno script batch

```bash
sbatch cluster/train.sh
```

### Sottomettere con override da CLI

```bash
sbatch --job-name=grpo-run2 --time=08:00:00 cluster/train.sh
```

### Passare argomenti allo script

```bash
sbatch cluster/train.sh --config experiments/configs/grpo_tinyllama.yaml --resume
```
> Gli argomenti dopo il nome dello script vengono passati come `$1`, `$2`, ecc.

### Sottomettere con dipendenza (esegui dopo un altro job)

```bash
# Esegui eval solo dopo che il training (job 12345) è completato con successo
sbatch --dependency=afterok:12345 cluster/eval.sh
```

| Tipo dipendenza | Significato |
|-----------------|-------------|
| `afterok:ID` | Esegui solo se il job ID è completato con successo |
| `afternotok:ID` | Esegui solo se il job ID è fallito |
| `afterany:ID` | Esegui in ogni caso dopo il job ID |
| `after:ID` | Esegui dopo che il job ID è partito |

---

## 2. Monitoraggio Job

### Vedere i propri job attivi

```bash
squeue --me
```

### Output dettagliato

```bash
squeue --me -o "%.10i %.20j %.8T %.10M %.6D %.4C %.10m %.20R %.10b"
```

| Colonna | Significato |
|---------|-------------|
| `%i` | Job ID |
| `%j` | Nome job |
| `%T` | Stato (RUNNING, PENDING, …) |
| `%M` | Tempo trascorso |
| `%D` | Numero nodi |
| `%C` | Numero CPU |
| `%m` | RAM richiesta |
| `%R` | Motivo/Nodo |
| `%b` | GPU richieste |

### Seguire i log in tempo reale

```bash
# Il file di output è definito nello script con #SBATCH --output=...
tail -f slurm-12345.out
```

### Dettagli di un job specifico

```bash
scontrol show job 12345
```

### Storico dei job completati

```bash
# Ultimi 7 giorni
sacct --starttime=$(date -d '7 days ago' +%Y-%m-%d) --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize,AllocTRES%40

# Solo i propri job
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
```

## 3. Cancellazione Job

### Cancellare un job specifico

```bash
scancel 12345
```

### Cancellare tutti i propri job

```bash
scancel --me
```

### Cancellare solo i job in stato PENDING

```bash
scancel --me --state=PENDING
```

### Cancellare per nome

```bash
scancel --me --name=grpo-train
```

---

## 4. Informazioni sul Cluster

### Vedere i nodi e le GPU disponibili

```bash
sinfo -N -l
```

### Partizioni disponibili

```bash
sinfo -s
```

### Dettaglio di un nodo specifico

```bash
scontrol show node gnode10
```

### QoS disponibili e i loro limiti

```bash
sacctmgr show qos format=Name,MaxTRES,MaxWall,Priority
```

### Le proprie associazioni (account, partizione, QoS)

```bash
sacctmgr show associations user=$USER format=Account,Partition,QOS
```
> **Fondamentale**: questo comando ti dice quali `--account`, `--partition` e `--qos` puoi usare negli script SLURM.

### Quota disco

```bash
quota -s          # quota personale
df -h /scratch    # spazio scratch condiviso
```

---

## 5. Sessioni Interattive

### Sessione interattiva con GPU

```bash
srun --pty --gres=gpu:1 --qos=gpu-large --time=01:00:00 bash
```

### Sessione interattiva su un nodo specifico

```bash
srun --pty --nodelist=gnode10 --gres=gpu:1 --qos=gpu-large --time=01:00:00 bash
```

### Sessione interattiva con Apptainer

```bash
srun --pty --gres=gpu:1 --qos=gpu-large --time=01:00:00 \
  apptainer exec --nv container.sif bash
```

### Verifica rapida GPU da sessione interattiva

```bash
srun --pty --gres=gpu:1 --qos=gpu-small --time=00:10:00 bash -c "nvidia-smi"
```

---

## 6. Apptainer (Container)

### Costruire un container (da Dockerfile)

```bash
# Prima converti il Dockerfile in un .def file, oppure:
apptainer build container.sif docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

### Costruire da un .def file

```bash
apptainer build --fakeroot container.sif container.def
```

### Eseguire un comando nel container

```bash
apptainer exec --nv container.sif python -c "import torch; print(torch.cuda.is_available())"
```

| Flag | Significato |
|------|-------------|
| `--nv` | Abilita il supporto GPU NVIDIA |
| `--bind /path/host:/path/container` | Monta una directory dall'host nel container |
| `--writable-tmpfs` | Permette scritture temporanee nel container |
| `--env VAR=value` | Imposta variabile d'ambiente |

### Eseguire con bind di directory

```bash
apptainer exec --nv \
  --bind $HOME/grpo-strict-generation:/workspace \
  --bind /scratch/$USER:/scratch \
  container.sif python -m src.training --config /workspace/experiments/configs/grpo_tinyllama.yaml
```

### Shell interattiva nel container

```bash
apptainer shell --nv container.sif
```

---

## 7. Debug GPU

### Stato GPU sul nodo corrente

```bash
nvidia-smi
```

### Monitoraggio continuo (ogni 2 secondi)

```bash
watch -n 2 nvidia-smi
```

### Solo VRAM utilizzata

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

### Compute capability (per verificare supporto bf16)

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# (8, 9) = L40S → bf16 OK
# (7, 0) = V100 → NO bf16, usa fp16
# (3, 7) = K80  → NO bf16, NO quantizzazione 4bit
```

### Verificare CUDA nel container

```bash
apptainer exec --nv container.sif python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Compute Capability: {torch.cuda.get_device_capability()}')
"
```

---

## 8. Trasferimento File

### Caricare file sul cluster

```bash
# Singolo file
scp experiments/configs/grpo_tinyllama.yaml user@cluster:/home/user/grpo/experiments/configs/

# Directory intera
scp -r src/ user@cluster:/home/user/grpo/src/

# rsync (più efficiente, incrementale)
rsync -avz --progress . user@cluster:/home/user/grpo/ --exclude .venv --exclude __pycache__
```

### Scaricare risultati dal cluster

```bash
# Checkpoint migliore
scp -r user@cluster:/home/user/grpo/outputs/grpo_run/best/ ./outputs/

# Log wandb offline
scp -r user@cluster:/home/user/grpo/experiments/logs/ ./experiments/logs/

# Rsync solo i risultati
rsync -avz user@cluster:/home/user/grpo/experiments/ ./experiments/
```

### Sincronizzare wandb offline runs

```bash
# Sul cluster (dopo il job) o in locale (dopo il download)
wandb sync experiments/logs/grpo/wandb/offline-run-*
```

---

## 9. Workflow Completo di Esempio

```bash
# 1. Scopri le tue associazioni
sacctmgr show associations user=$USER format=Account,Partition,QOS

# 2. Aggiorna gli script con account/partition/qos corretti
vim cluster/train.sh

# 3. Carica il codice sul cluster
rsync -avz --progress . user@cluster:~/grpo/ --exclude .venv --exclude __pycache__ --exclude wandb

# 4. Setup iniziale (una sola volta)
sbatch cluster/setup.sh

# 5. Verifica che il setup sia andato a buon fine
sacct --jobs=<SETUP_JOB_ID> --format=JobID,State,ExitCode
cat slurm-<SETUP_JOB_ID>.out

# 6. Lancia la baseline evaluation
BASELINE_JOB=$(sbatch --parsable MODE=baseline cluster/eval.sh)
echo "Baseline job: $BASELINE_JOB"

# 7. Lancia il training (dopo la baseline)
TRAIN_JOB=$(sbatch --parsable --dependency=afterany:$BASELINE_JOB cluster/train.sh)
echo "Training job: $TRAIN_JOB"

# 8. Monitora
squeue --me
tail -f slurm-$TRAIN_JOB.out

# 9. Dopo il completamento, scarica i risultati
rsync -avz user@cluster:~/grpo/experiments/ ./experiments/

# 10. Sincronizza wandb
wandb sync experiments/logs/grpo/wandb/offline-run-*
```
