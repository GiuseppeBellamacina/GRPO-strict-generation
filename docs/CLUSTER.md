# Guida al Cluster GPU — DMI UniCT

Guida passo-passo per eseguire il progetto GRPO Strict Generation
sul cluster GPU del Dipartimento di Matematica e Informatica (UniCT).

---

## Indice

1. [Panoramica Hardware](#1-panoramica-hardware)
2. [Accesso al Cluster](#2-accesso-al-cluster)
3. [Setup Iniziale (una tantum)](#3-setup-iniziale-una-tantum)
4. [Configurazione per GPU](#4-configurazione-per-gpu)
5. [Lanciare il Training](#5-lanciare-il-training)
6. [Monitorare i Job](#6-monitorare-i-job)
7. [Checkpoint e Resume](#7-checkpoint-e-resume)
8. [Wandb Offline](#8-wandb-offline)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Panoramica Hardware

| Nodo | GPU | VRAM | CC | bf16 | Unsloth/vLLM | Note |
|------|-----|------|----|------|--------------|------|
| gnode1–4 | 1× K80 | 22 GB | 3.7 | ❌ | ❌ | Solo fp32/fp16, no quantizzazione 4bit |
| gnode5 | 4× V100 | 16 GB each | 7.0 | ❌ | ✅ | Riservato dottorandi. fp16 OK |
| gnode10 | 4× L40S | 48 GB each | 8.9 | ✅ | ✅ | Tutto supportato |

### QoS disponibili

| QoS | CPU | RAM | GPU VRAM | Tempo max |
|-----|-----|-----|----------|-----------|
| gpu-small | 1 | 4 GB | 2.8 GB | 4h |
| gpu-medium | 2 | 8 GB | 5.5 GB | 6h |
| gpu-large | 4 | 16 GB | 11 GB | 12h |
| gpu-xlarge | 8 | 48 GB | 22 GB | 12h |
| gpu-phd-large | 4 | 40 GB | 16 GB | 12h |

> **Per il training GRPO** serve almeno `gpu-large` (11 GB VRAM).
> Consigliato `gpu-xlarge` su K80 o `gpu-phd-large` su V100.
> Su L40S qualsiasi QoS con >= 11 GB VRAM è sufficiente.

### Compatibilità software per GPU

| Feature | K80 (CC 3.7) | V100 (CC 7.0) | L40S (CC 8.9) |
|---------|:---:|:---:|:---:|
| Unsloth | ❌ | ✅ | ✅ |
| vLLM / fast_inference | ❌ | ✅ | ✅ |
| bfloat16 | ❌ | ❌ | ✅ |
| 4-bit quantization (NF4) | ❌ | ✅ | ✅ |
| paged_adamw_8bit | ❌ | ✅ | ✅ |
| VLLM standby=1 | ❌ | ❌ | Probabile |

> Il config da usare dipende dalla GPU assegnata.
> Per K80 usa `experiments/configs/grpo_cluster.yaml`.
> Per V100/L40S puoi usare `experiments/configs/grpo.yaml` con gli adattamenti
> descritti nella sezione [Configurazione per GPU](#4-configurazione-per-gpu).

---

## 2. Accesso al Cluster

### 2.1. Primo accesso SSH

```bash
ssh <codice-fiscale>@gcluster.dmi.unict.it
```

La password è quella dell'account universitario (SmartEdu).

### 2.2. Configura chiave SSH (consigliato)

Evita ban per tentativi di password sbagliati.

**Linux / macOS:**
```bash
ssh-keygen -t ed25519         # se non hai già una chiave
ssh-copy-id <codice-fiscale>@gcluster.dmi.unict.it
```

**Windows (PowerShell)** — `ssh-copy-id` non è disponibile nativamente:
```powershell
# Crea la chiave se non esiste già (controlla prima con Test-Path)
Test-Path "$env:USERPROFILE\.ssh\id_ed25519"

# Se non esiste:
ssh-keygen -t ed25519

# Copia la chiave pubblica sul cluster (equivalente di ssh-copy-id):
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | ssh <codice-fiscale>@gcluster.dmi.unict.it "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Dopodiché `ssh <codice-fiscale>@gcluster.dmi.unict.it` non chiederà più la password.

### 2.3. Trasferisci il progetto

**Linux / macOS** — con rsync (esclude cartelle pesanti):
```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git/objects' \
    . <codice-fiscale>@gcluster.dmi.unict.it:~/GRPO-strict-generation/
```

**Windows (PowerShell)** — usa lo script `sync_cluster.ps1` nella root del progetto:
```powershell
# Prima crea la struttura remota (una volta sola):
ssh <codice-fiscale>@gcluster.dmi.unict.it "mkdir -p ~/GRPO-strict-generation/experiments/{configs,logs,checkpoints}"

# Carica il progetto:
.\sync_cluster.ps1 -Action upload

# Dopo il training, scarica i risultati:
.\sync_cluster.ps1 -Action download          # tutto (logs + checkpoints + wandb)
.\sync_cluster.ps1 -Action download-logs        # solo logs e figure
.\sync_cluster.ps1 -Action download-checkpoints # solo checkpoint LoRA
.\sync_cluster.ps1 -Action download-wandb       # solo run wandb offline
```

Oppure clona direttamente (se sei dottorando con accesso internet):

```bash
# Sul cluster:
git clone https://github.com/GiuseppeBellamacina/GRPO-strict-generation.git
```

> **Nota:** Solo i dottorandi hanno accesso a internet. Gli studenti devono
> trasferire i file via `scp`/`sync_cluster.ps1`. Il download di modelli HuggingFace
> e pacchetti pip funziona per tutti.

### 2.4. Verifica quota disco

```bash
quota -s
```

Hai ~50 GB di quota soft. I checkpoint LoRA sono piccoli (~50 MB l'uno),
ma la cache HuggingFace (`~/.cache/huggingface/`) può crescere.
Per TinyLlama servono ~2 GB di cache.

---

## 3. Setup Iniziale (una tantum)

Il setup installa le dipendenze Python dentro il container Apptainer.
I pacchetti vengono installati in `~/.local/` (persistente tra i job).

### 3.1. Apri una sessione interattiva

```bash
# Scopri a quali queue sei autorizzato:
sacctmgr show associations user=$USER format=account,qos,defaultqos -P

# Apri una shell interattiva (sostituisci dl-course-q2 con la tua queue):
srun --account dl-course-q2 --partition dl-course-q2 --qos gpu-medium \
     --gres=gpu:1 --gres=shard:5000 --mem=8G \
     --pty apptainer shell --nv /shared/sifs/latest.sif
```

### 3.2. Esegui lo script di setup

```bash
cd ~/GRPO-strict-generation
bash cluster/setup.sh
```

Lo script:
- Rileva la GPU e installa Unsloth/vLLM solo se supportati (CC >= 7.0)
- Installa tutte le dipendenze via `pip --user`
- Genera il dataset sintetico (5000 samples)
- Verifica l'installazione

### 3.3. Esci dalla sessione interattiva

```bash
exit
```

---

## 4. Configurazione per GPU

Ogni modello ha il proprio config in `experiments/configs/` (es.
`grpo_smollm2_135m.yaml`, `grpo_tinyllama.yaml`, ecc.). I config sono
ottimizzati per L40S con 4-bit NF4 e bfloat16. Per GPU diverse,
adatta i parametri come descritto sotto.

### L40S (gnode10) — Configurazione ideale

Il config di default funziona quasi as-is. Puoi aumentare le risorse:

```yaml
model:
  dtype: "bfloat16"          # ✅ supportato
  use_unsloth: true          # ✅ supportato
  fast_inference: true       # ✅ supportato
  gpu_memory_utilization: 0.85

grpo:
  num_generations: 8         # più spazio VRAM → gruppo più grande
  max_completion_length: 768
```

### V100 (gnode5) — Solo per dottorandi

```yaml
model:
  dtype: "float16"           # ⚠️ V100 NON supporta bfloat16
  use_unsloth: true          # ✅ CC 7.0 OK
  fast_inference: true       # ✅ vLLM OK
  gpu_memory_utilization: 0.80  # 16 GB VRAM, essere conservativi

training:
  bf16: false                # ⚠️ CRITICO: deve essere false su V100

grpo:
  num_generations: 4         # 16 GB VRAM → gruppo piccolo
  max_completion_length: 512 # ridurre per stare in 16 GB
```

### K80 (gnode1–4) — Limitazioni significative

La K80 (Compute Capability 3.7) **non supporta**:
- bfloat16
- bitsandbytes 4-bit (NF4) — richiede CC >= 7.0
- Unsloth — richiede CC >= 7.0
- vLLM — richiede CC >= 7.0

```yaml
model:
  dtype: "float16"
  quantization: null         # ⚠️ NO 4-bit su K80
  use_unsloth: false         # ⚠️ NON supportato
  fast_inference: false      # ⚠️ NON supportato

training:
  bf16: false
  optim: "adamw_torch"       # paged_adamw_8bit potrebbe non funzionare

grpo:
  num_generations: 2         # TinyLlama senza quant. ≈ 4.5 GB, meno spazio
  max_completion_length: 512
```

> **Nota:** Senza quantizzazione TinyLlama 1.1B occupa ~4.5 GB in fp16.
> Su K80 (22 GB VRAM con `gpu-xlarge`) c'è spazio, ma il training
> sarà molto più lento rispetto a V100/L40S.

---

## 5. Lanciare il Training

### 5.1. Configura lo script batch

Modifica `cluster/train.sh` con i tuoi parametri:

```bash
# Apri il file sul cluster:
nano cluster/train.sh
```

Cambia queste righe:

```bash
#SBATCH --account=dl-course-q1      # ← la tua queue
#SBATCH --partition=dl-course-q1    # ← idem
#SBATCH --qos=gpu-xlarge            # ← il tuo QoS
#SBATCH --mail-user=tua@email.com   # ← la tua email
#SBATCH --gres=gpu:1,shard:16000    # ← VRAM in MB
```

### 5.2. Lancia il job

```bash
cd ~/GRPO-strict-generation
mkdir -p logs
sbatch cluster/train.sh
```

### 5.3. Pipeline completa (multi-modello)

La pipeline automatizzata addestra e valuta tutti e 5 i modelli in
sequenza tramite SLURM job chaining:

```bash
# Lancia tutto (train + eval per ciascun modello):
run-all

# Solo alcuni modelli (indice 1-5):
run-all --models=1,2,3

# Monitora lo stato:
monitor          # vista compatta (default)
monitor --tab    # tabella completa
```

Per il dettaglio dei comandi, vedi `cluster/setup.sh` (alias) e
`cluster/run_all.sh` (pipeline).

---

## 6. Monitorare i Job

```bash
# Stato dei tuoi job
squeue -u $USER

# Dettagli job specifico
scontrol show job <JOB_ID>

# Output in tempo reale
tail -f logs/slurm-<JOB_ID>.log

# Risorse usate (job in corso)
sstat -aPno TresUsageInMax -j <JOB_ID>

# Risorse usate (job completato)
sacct -aPno TresUsageInMax -j <JOB_ID>

# Cancella un job
scancel <JOB_ID>
```

---

## 7. Checkpoint e Resume

Il training salva checkpoint ogni `save_steps` (default: 60 step).
Il tempo massimo per job è 12 ore. Se il training non finisce:

### Resume automatico

```bash
# Modifica cluster/train.sh:
EXTRA_ARGS="--resume"

# Rilancia:
sbatch cluster/train.sh
```

Il flag `--resume` trova l'ultimo checkpoint e riprende da lì.

### Eval-only

Per valutare i checkpoint senza riaddestrare:

```bash
EXTRA_ARGS="--eval-only experiments/checkpoints/grpo/nothink/curriculum"
```

---

## 8. Wandb Offline

Il cluster **non ha accesso a internet** (eccezione: dottorandi).
Gli script batch impostano automaticamente `WANDB_MODE=offline`.

I run vengono salvati in `experiments/logs/<experiment>/wandb/` (es. `experiments/logs/grpo/wandb/`).

### Sincronizzare i risultati dopo il training

**Linux / macOS:**
```bash
rsync -avz <username>@gcluster.dmi.unict.it:~/GRPO-strict-generation/experiments/logs/ ./experiments/logs/
wandb sync experiments/logs/grpo/wandb/offline-run-*
```

**Windows (PowerShell):**
```powershell
.\sync_cluster.ps1 -Action download-wandb
# Lo script scarica experiments/logs/ e stampa il comando di sync:
wandb sync experiments\logs\grpo\wandb\offline-run-*
```

### Alternativa: usa solo TensorBoard

Se preferisci non usare wandb, cambia nel config YAML:

```yaml
# Nel tuo config (non serve modificare codice):
wandb:
  project: null   # disabilita wandb
```

E nei batch script aggiungi `--report_to tensorboard` come argomento al training.
I log TensorBoard vengono salvati in `experiments/logs/`.

---

## 9. Troubleshooting

### "ModuleNotFoundError: No module named 'unsloth'"

La GPU non supporta Unsloth (CC < 7.0). Modifica il config:
```yaml
model:
  use_unsloth: false
  fast_inference: false
```

### "CUDA out of memory"

Riduci le risorse nel config:
```yaml
grpo:
  num_generations: 2          # meno rollout paralleli
  max_completion_length: 256  # completions più corte
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
```

### "Job cancelled for excessive GPU RAM usage"

La VRAM richiesta in `#SBATCH --gres=shard:XXXX` è troppo bassa.
Aumenta il valore o usa un QoS più grande.

### Job non parte (resta in coda)

- Verifica con `squeue -u $USER` lo stato (PENDING)
- Prova un QoS più piccolo per avere priorità più alta
- Verifica che non hai altri job attivi (limite: 1 job alla volta)

### Training troppo lento su K80

La K80 è ~5–10× più lenta di V100/L40S per operazioni ML.
Riduci `max_steps` o usa meno dati con `dataset.max_samples`.

### "pip install" fallisce

Verifica di essere dentro il container Apptainer:
```bash
apptainer shell --nv /shared/sifs/latest.sif
# Il prompt cambia in "Apptainer>"
pip install --user <pacchetto>
```

### Quota disco superata

```bash
# Controlla la quota:
quota -s

# Pulisci cache HuggingFace:
rm -rf ~/.cache/huggingface/hub/models--*/.no_exist*
# Rimuovi checkpoint vecchi:
ls -la experiments/checkpoints/grpo/nothink/curriculum/
```
