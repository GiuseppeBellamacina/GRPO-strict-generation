# Modelli

## Modelli testati

| Modello | Parametri | Baseline Pass@1 | GRPO Pass@1 | Delta | Note |
| --- | --- | --- | --- | --- | --- |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | 0.730 | 0.915 | +0.185 | Primo esperimento, 600 steps su L40S |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | — | — | — | In corso |

## Modelli candidati

### Piccoli (< 1B) — baseline bassa, training veloce

| Modello | Parametri | Architettura | Note |
| --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | LLaMA-like | Molto piccolo, baseline attesa molto bassa |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | LLaMA-like | Buon compromesso dimensione/capacità |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Qwen2.5 | Architettura moderna, buon tokenizer |
| `microsoft/phi-1_5` | 1.3B | Phi | Orientato al codice, no chat template |

### Medi (1B–3B) — baseline media, buon potenziale GRPO

| Modello | Parametri | Architettura | Note |
| --- | --- | --- | --- |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | LLaMA 2 | Già testato |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Qwen2.5 | Più capace di TinyLlama |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | LLaMA-like | Versione grande di SmolLM2 |
| `google/gemma-2-2b-it` | 2B | Gemma 2 | Chat model Google, architettura moderna |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | LLaMA 3.2 | Gated (richiede accesso HF) |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | LLaMA 3.2 | Gated, più VRAM richiesta |
| `microsoft/phi-2` | 2.7B | Phi-2 | Buono su codice, no chat template |

### Grandi (3B+) — baseline alta, meno margine GRPO

| Modello | Parametri | Architettura | Note |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Qwen2.5 | Potrebbe già essere buono su JSON |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | Qwen2.5 | Specializzato su codice |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | 3B | Qwen2.5 | Specializzato su codice, baseline alta attesa |

## Note

- Tutti i modelli sopra supportano 4-bit quantization (NF4) e LoRA
- I modelli `meta-llama` richiedono accettazione della licenza su HuggingFace
- Con shard da 22.5 GB (L40S gpu-xlarge), modelli fino a ~3B in 4-bit stanno comodi
- Per modelli > 3B serve ridurre `num_generations` o `gpu_memory_utilization`
