# Guida al Cluster GPU — DMI UniCT

Guida completa per utilizzare il cluster GPU del Dipartimento di Matematica e Informatica dell'Università di Catania. Copre accesso SSH, gestione job SLURM, container Apptainer, GPU e trasferimento file.

---

## Indice

1. [Panoramica Hardware](#1-panoramica-hardware)
2. [Accesso al Cluster](#2-accesso-al-cluster)
3. [Sessioni Interattive e Container](#3-sessioni-interattive-e-container)
4. [Job SLURM](#4-job-slurm)
5. [Monitoraggio Job](#5-monitoraggio-job)
6. [GPU e Debug](#6-gpu-e-debug)
7. [Trasferimento File](#7-trasferimento-file)
8. [Wandb Offline](#8-wandb-offline)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Panoramica Hardware

### Nodi disponibili

| Nodo | GPU | VRAM | Compute Capability | bf16 | Note |
|------|-----|------|--------------------|------|------|
| gnode1–4 | 1× K80 | 22 GB | 3.7 | ❌ | Solo fp32/fp16, no quantizzazione 4-bit |
| gnode5 | 4× V100 | 16 GB ciascuna | 7.0 | ❌ | Riservato dottorandi. fp16 OK |
| gnode10 | 4× L40S | 48 GB ciascuna | 8.9 | ✅ | Tutto supportato |

### QoS disponibili

| QoS | CPU | RAM | GPU VRAM | Tempo max |
|-----|-----|-----|----------|-----------|
| gpu-small | 1 | 4 GB | 2.8 GB | 4h |
| gpu-medium | 2 | 8 GB | 5.5 GB | 6h |
| gpu-large | 4 | 16 GB | 11 GB | 12h |
| gpu-xlarge | 8 | 48 GB | 22 GB | 12h |
| gpu-phd-large | 4 | 40 GB | 16 GB | 12h |

> **Nota:** Solo i dottorandi hanno accesso a internet dal cluster. Gli studenti possono comunque scaricare modelli HuggingFace, dataset Kaggle e pacchetti pip (traffico verso repository consentito).

---

## 2. Accesso al Cluster

### 2.1. Primo accesso SSH

```bash
ssh <CODICE-FISCALE>@gcluster.dmi.unict.it
```

La password è quella dell'account universitario (SmartEdu).

### 2.2. Configura chiave SSH (consigliato)

Evita ban per tentativi di password sbagliati e rende l'accesso immediato.

**Linux / macOS:**
```bash
# Crea la chiave (se non ne hai già una)
ssh-keygen -t ed25519

# Copia la chiave sul cluster
ssh-copy-id <CODICE-FISCALE>@gcluster.dmi.unict.it
```

**Windows (PowerShell)** — `ssh-copy-id` non è disponibile nativamente:
```powershell
# Verifica se esiste già una chiave
Test-Path "$env:USERPROFILE\.ssh\id_ed25519"

# Se non esiste, creala:
ssh-keygen -t ed25519

# Copia la chiave pubblica sul cluster:
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | `
  ssh <CODICE-FISCALE>@gcluster.dmi.unict.it `
  "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### 2.3. Alias SSH (consigliato)

Aggiungi al file `~/.ssh/config` (Linux/macOS) o `%USERPROFILE%\.ssh\config` (Windows):

```
Host gcluster
    HostName gcluster.dmi.unict.it
    User <CODICE-FISCALE>
    IdentityFile ~/.ssh/id_ed25519
```

Da qui in poi basta:

```bash
ssh gcluster
```

### 2.4. Scopri la tua queue e associazioni

```bash
sacctmgr show associations user=$USER format=Account,Partition,QOS,DefaultQOS -P
```

Output di esempio:

```
Account|Partition|QOS|DefaultQOS
dl-course-q2|dl-course-q2|gpu-medium,gpu-large,gpu-xlarge|gpu-medium
```

Prendi nota di `Account`, `Partition` e dei QoS disponibili: servono in ogni script SLURM.

### 2.5. Verifica quota disco

```bash
quota -s
```

La quota è tipicamente ~50 GB. Cache di modelli e checkpoint possono crescere rapidamente.

---

## 3. Sessioni Interattive e Container

### 3.1. Sessione interattiva con GPU

```bash
srun --account <ACCOUNT> --partition <PARTITION> --qos <QOS> \
     --gres=gpu:1 --gres=shard:<VRAM_MB> --mem=<RAM> \
     --pty bash
```

Esempio concreto:

```bash
srun --account dl-course-q2 --partition dl-course-q2 --qos gpu-medium \
     --gres=gpu:1 --gres=shard:5000 --mem=8G \
     --pty bash
```

### 3.2. Sessione interattiva su nodo specifico

```bash
srun --pty --nodelist=gnode10 --gres=gpu:1 --qos=gpu-large --time=01:00:00 bash
```

### 3.3. Apptainer (container)

Il cluster usa **Apptainer** (ex Singularity) per i container. I container `.sif` sono read-only; i pacchetti si installano in `~/.local/` (persistente tra i job).

**Shell interattiva nel container:**
```bash
srun --account <ACCOUNT> --partition <PARTITION> --qos gpu-medium \
     --gres=gpu:1 --gres=shard:5000 --mem=8G \
     --pty apptainer shell --nv /shared/sifs/latest.sif
```

**Eseguire un comando nel container:**
```bash
apptainer exec --nv /shared/sifs/latest.sif python -c "import torch; print(torch.cuda.is_available())"
```

**Eseguire con bind di directory:**
```bash
apptainer exec --nv \
  --bind $HOME/progetto:/workspace \
  --bind /scratch/$USER:/scratch \
  /shared/sifs/latest.sif python mio_script.py
```

| Flag Apptainer | Significato |
|----------------|-------------|
| `--nv` | Abilita supporto GPU NVIDIA |
| `--bind host:container` | Monta una directory dall'host |
| `--writable-tmpfs` | Permette scritture temporanee nel container |
| `--env VAR=valore` | Imposta variabile d'ambiente |

**Installare pacchetti Python nel container:**
```bash
# Dentro la shell Apptainer:
pip install --user <pacchetto>
```

I pacchetti `--user` finiscono in `~/.local/` e persistono tra i job.

---

## 4. Job SLURM

### 4.1. Script batch di esempio

```bash
#!/bin/bash
#SBATCH --job-name=mio-job
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=gpu:1,shard:22528
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tua@email.com

apptainer run --nv /shared/sifs/latest.sif \
    python mio_script.py --argomento valore
```

### 4.2. Sottomettere un job

```bash
sbatch mio_script.sh
```

### 4.3. Sottomettere con override da CLI

```bash
sbatch --job-name=run2 --time=08:00:00 mio_script.sh
```

### 4.4. Passare argomenti allo script

```bash
sbatch mio_script.sh --config config.yaml --resume
```

Gli argomenti dopo il nome dello script sono disponibili come `$1`, `$2`, ecc.

---

## 5. Monitoraggio Job

### 5.1. Job attivi

```bash
# Vista semplice
squeue --me

# Vista dettagliata
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
| `%R` | Motivo coda / Nodo assegnato |
| `%b` | GPU richieste |

### 5.2. Log in tempo reale

```bash
tail -f logs/slurm-<JOB_ID>.log
```

### 5.3. Dettagli di un job

```bash
scontrol show job <JOB_ID>
```

### 5.4. Storico job completati

```bash
# Ultimi 7 giorni
sacct --starttime=$(date -d '7 days ago' +%Y-%m-%d) \
      --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES%40

# Solo i propri job con exit code
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
```

### 5.5. Efficienza di un job completato

```bash
seff <JOB_ID>
```

Mostra utilizzo CPU, RAM e tempo effettivo vs. richiesto.

### 5.6. Risorse usate

```bash
# Job in corso
sstat -aPno TresUsageInMax -j <JOB_ID>

# Job completato
sacct -aPno TresUsageInMax -j <JOB_ID>
```

### 5.7. Cancellare job

```bash
# Job specifico
scancel <JOB_ID>

# Tutti i propri job
scancel --me

# Solo i PENDING
scancel --me --state=PENDING

# Per nome
scancel --me --name=mio-job
```

---

## 6. GPU e Debug

### 6.1. Stato GPU sul nodo corrente

Se sei in una sessione interattiva:

```bash
nvidia-smi
```

### 6.2. Stato GPU di un job in esecuzione

Non puoi fare SSH diretto ai nodi di calcolo (niente password). Usa `srun --overlap`:

```bash
# Trova il nodo assegnato
squeue -u $USER -o "%.8i %.8j %.2t %.4M %.10R"

# Esegui nvidia-smi sul nodo del job
srun --jobid=<JOB_ID> --overlap nvidia-smi
```

### 6.3. Monitoraggio continuo

```bash
# Ogni 2 secondi (da sessione interattiva)
watch -n 2 nvidia-smi

# Solo VRAM
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

# Con loop da srun
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv -l 2
```

### 6.4. Verifica Compute Capability

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# (8, 9) → L40S: bf16 OK, tutto supportato
# (7, 0) → V100: NO bf16, fp16 OK
# (3, 7) → K80:  NO bf16, NO 4-bit quantization
```

### 6.5. Verifica CUDA nel container

```bash
apptainer exec --nv /shared/sifs/latest.sif python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'Compute Capability: {torch.cuda.get_device_capability()}')
"
```

---

## 7. Trasferimento File

### 7.1. Caricare file sul cluster

```bash
# Singolo file
scp file.py <USER>@gcluster.dmi.unict.it:~/progetto/

# Directory intera
scp -r src/ <USER>@gcluster.dmi.unict.it:~/progetto/src/

# rsync (incrementale, più efficiente)
rsync -avz --progress . <USER>@gcluster.dmi.unict.it:~/progetto/ \
    --exclude .venv --exclude __pycache__ --exclude .git/objects
```

### 7.2. Scaricare risultati

```bash
# Tutto
rsync -avz <USER>@gcluster.dmi.unict.it:~/progetto/results/ ./results/

# Solo un file
scp <USER>@gcluster.dmi.unict.it:~/progetto/output.json .
```

### 7.3. Da Windows (PowerShell)

`scp` e `ssh` funzionano nativamente. Per rsync è possibile usare `wsl rsync` oppure uno script PowerShell personalizzato.

---

## 8. Wandb Offline

Il cluster non ha accesso a internet (eccezione: dottorandi). Per usare Weights & Biases, imposta la modalità offline nello script SLURM:

```bash
export WANDB_MODE=offline
export WANDB_DIR=./logs
```

I run vengono salvati localmente in `logs/wandb/offline-run-*`.

### Sincronizzare dopo il training

Scarica i log dal cluster e poi:

```bash
wandb sync logs/wandb/offline-run-*
```

### Alternativa: TensorBoard

Se non vuoi usare wandb, imposta `report_to: tensorboard` nella configurazione del framework. I log TensorBoard vengono salvati nella directory di logging.

---

## 9. Troubleshooting

### "CUDA out of memory"

Riduci le risorse:
- Meno batch size
- Meno rollout/generazioni parallele
- Completions più corte
- Gradient accumulation al posto di batch più grandi

### "Job cancelled for excessive GPU RAM usage"

La VRAM richiesta in `#SBATCH --gres=shard:XXXX` è inferiore a quella effettivamente usata. Aumenta il valore o usa un QoS con più VRAM.

### Job in coda (PENDING) che non parte

- `squeue --me` per verificare lo stato
- Prova un QoS più piccolo per avere priorità più alta
- Verifica di non avere altri job attivi (potrebbe esserci un limite di 1 job alla volta)
- Controlla il motivo con `scontrol show job <ID>` → campo `Reason`

### "pip install" fallisce nel container

Verifica di essere dentro il container Apptainer:
```bash
apptainer shell --nv /shared/sifs/latest.sif
# Il prompt cambia in "Apptainer>"
pip install --user <pacchetto>
```

### Quota disco superata

```bash
# Controlla la quota
quota -s

# Pulisci cache HuggingFace
rm -rf ~/.cache/huggingface/hub/models--*/.no_exist*

# Elenca i file più grandi
du -sh ~/.cache/* | sort -rh | head -10
```

### SSH ai nodi di calcolo non funziona

I nodi di calcolo non accettano connessioni SSH dirette. Per eseguire comandi su un nodo dove gira un tuo job:

```bash
srun --jobid=<JOB_ID> --overlap <comando>
```

### Informazioni sul cluster

```bash
# Nodi e GPU disponibili
sinfo -N -l

# Partizioni
sinfo -s

# Dettaglio di un nodo
scontrol show node gnode10

# QoS e i loro limiti
sacctmgr show qos format=Name,MaxTRES,MaxWall,Priority

# Le proprie associazioni
sacctmgr show associations user=$USER format=Account,Partition,QOS
```

---

## Workflow Completo di Esempio

```bash
# 1. Scopri le tue associazioni
sacctmgr show associations user=$USER format=Account,Partition,QOS -P

# 2. Crea la struttura remota (una tantum)
ssh gcluster "mkdir -p ~/progetto/{logs,results}"

# 3. Carica il codice
rsync -avz . gcluster:~/progetto/ --exclude .venv --exclude __pycache__

# 4. Setup iniziale: sessione interattiva con GPU
srun --account <ACC> --partition <PART> --qos gpu-medium \
     --gres=gpu:1 --gres=shard:5000 --mem=8G \
     --pty apptainer shell --nv /shared/sifs/latest.sif

# (dentro il container) Installa dipendenze
pip install --user -r requirements.txt
exit

# 5. Lancia il training
sbatch train.sh

# 6. Monitora
squeue --me
tail -f logs/slurm-<JOB_ID>.log
srun --jobid=<JOB_ID> --overlap nvidia-smi

# 7. Scarica i risultati
rsync -avz gcluster:~/progetto/results/ ./results/

# 8. (Opzionale) Sync wandb offline
wandb sync results/wandb/offline-run-*
```
