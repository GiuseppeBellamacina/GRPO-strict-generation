"""Custom GRPO implementation in pure PyTorch — educational reference.

This script implements Group Relative Policy Optimization (GRPO) from scratch
to show exactly what happens behind the scenes. NOT intended for production use;
use the trl-based grpo_train.py for actual training.

Algorithm (DeepSeek-R1 / Shao et al. 2024):
    For each prompt x_i in the batch:
        1. Sample G completions from the OLD policy  π_θ_old
        2. Score each completion with the rule-based reward  r(x_i, y_g)
        3. Normalize rewards within the group → advantages  A_g
        4. For each (prompt, completion) pair, compute the policy gradient
           with a clipped ratio (like PPO) and a KL penalty against a
           reference policy π_ref (the frozen model before training).

Loss per token t of completion g for prompt i:

    ratio_t = π_θ(token_t | x_i, y_{<t}) / π_θ_old(token_t | x_i, y_{<t})
    L_clip  = min(ratio_t * A_g,  clip(ratio_t, 1-ε, 1+ε) * A_g)
    L_kl    = β * KL(π_θ || π_ref) per token  (approx via log-ratio)

    L_total = - (1/G) Σ_g [ (1/T_g) Σ_t  L_clip_t  -  β * kl_t ]

Usage (educational — on a small model):
    python -m src.training.grpo_custom --model Qwen/Qwen2.5-Coder-0.5B-Instruct --steps 50

Reference: https://arxiv.org/abs/2402.03300 (DeepSeekMath / GRPO)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    group_size: int = 4  # G — number of completions per prompt
    max_completion_tokens: int = 256
    max_prompt_tokens: int = 192
    micro_batch_prompts: int = 2  # prompts per gradient step
    gradient_accumulation: int = 4
    total_steps: int = 200
    lr: float = 5e-6
    beta: float = 0.04  # KL penalty coefficient
    clip_eps: float = 0.2  # PPO-style clip range
    temperature: float = 0.7  # sampling temperature
    seed: int = 42


# ── Reward functions (same as rewards.py, inlined for self-containedness) ─────


def _extract_code_block(text: str, language: str) -> str | None:
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    if language == "json" and (stripped.startswith("{") or stripped.startswith("[")):
        return stripped
    if language == "python" and (
        stripped.startswith("def ") or stripped.startswith("class ")
    ):
        return stripped
    return None


def compute_reward(completion: str, task_type: str) -> float:
    """Binary reward: 1.0 if the code/JSON parses, 0.0 otherwise."""
    lang = "json" if task_type == "json" else "python"
    code = _extract_code_block(completion, lang)
    if code is None:
        return 0.0
    try:
        if task_type == "json":
            json.loads(code)
        else:
            ast.parse(code)
        return 1.0
    except (json.JSONDecodeError, SyntaxError):
        return 0.0


# ── Toy prompt pool (small set for demo) ──────────────────────────────────────


DEMO_PROMPTS = [
    {
        "system": "You are a helpful assistant that generates valid JSON. "
        "Respond ONLY with a JSON code block.",
        "user": 'Generate a JSON object with keys "name" (string), "age" (integer), "active" (boolean).',
        "task_type": "json",
    },
    {
        "system": "You are a helpful assistant that generates valid Python code. "
        "Respond ONLY with a Python code block.",
        "user": "Write a Python function called `factorial` that takes an integer n "
        "and returns the factorial of n.",
        "task_type": "python",
    },
    {
        "system": "You are a helpful assistant that generates valid JSON. "
        "Respond ONLY with a JSON code block.",
        "user": "Generate a JSON array of 3 objects, each with keys "
        '"id" (integer) and "value" (string).',
        "task_type": "json",
    },
    {
        "system": "You are a helpful assistant that generates valid Python code. "
        "Respond ONLY with a Python code block.",
        "user": "Write a Python function `is_prime` that returns True if the input "
        "integer is prime, False otherwise.",
        "task_type": "python",
    },
]


# ── Core GRPO Logic ──────────────────────────────────────────────────────────


@torch.no_grad()
def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,      # (B, L_prompt)
    attention_mask: torch.Tensor,   # (B, L_prompt)
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample G completions per prompt using multinomial sampling.

    Returns:
        completion_ids: (B*G, L_comp)  — token ids of completions only
        comp_mask:      (B*G, L_comp)  — 1 where real tokens, 0 where padding
    """
    B, L = prompt_ids.shape
    G = cfg.group_size
    device = prompt_ids.device

    # Repeat each prompt G times: (B*G, L)
    prompt_ids_rep = prompt_ids.repeat_interleave(G, dim=0)
    attn_mask_rep = attention_mask.repeat_interleave(G, dim=0)

    # Autoregressive generation token by token
    generated = []
    past_key_values = None
    input_ids = prompt_ids_rep
    cur_mask = attn_mask_rep

    for step in range(cfg.max_completion_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=cur_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # Logits of last position → sample
        logits = outputs.logits[:, -1, :] / cfg.temperature  # (B*G, vocab)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B*G, 1)
        generated.append(next_token)

        # Prepare for next step (only feed the new token thanks to KV cache)
        input_ids = next_token
        cur_mask = torch.cat(
            [cur_mask, torch.ones(B * G, 1, device=device, dtype=cur_mask.dtype)], dim=1
        )

        # Stop if ALL sequences have hit EOS
        all_done = (next_token.squeeze(-1) == tokenizer.eos_token_id).all()
        if all_done:
            break

    # Stack into (B*G, T_generated)
    completion_ids = torch.cat(generated, dim=1)
    T = completion_ids.shape[1]

    # Build mask: tokens after EOS are padding
    comp_mask = torch.ones_like(completion_ids, dtype=torch.float32)
    for i in range(completion_ids.shape[0]):
        eos_positions = (completion_ids[i] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            comp_mask[i, first_eos + 1 :] = 0.0  # mask everything after first EOS

    return completion_ids, comp_mask


def compute_log_probs(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,      # (N, L_prompt)
    completion_ids: torch.Tensor,  # (N, L_comp)
    prompt_mask: torch.Tensor,     # (N, L_prompt)
    comp_mask: torch.Tensor,       # (N, L_comp)
) -> torch.Tensor:
    """Compute per-token log probabilities of the completion under the model.

    Returns:
        log_probs: (N, L_comp)  — log π(token_t | x, y_{<t})
    """
    # Concatenate prompt + completion
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (N, L_p + L_c)
    attn_mask = torch.cat([prompt_mask, comp_mask], dim=1)

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits  # (N, L_p + L_c, vocab)

    # We want the logits that predict each completion token:
    # logits[:, L_p-1 : L_p+L_c-1, :] predict completion tokens at positions L_p : L_p+L_c
    L_p = prompt_ids.shape[1]
    L_c = completion_ids.shape[1]
    comp_logits = logits[:, L_p - 1 : L_p + L_c - 1, :]  # (N, L_c, vocab)

    log_probs = F.log_softmax(comp_logits, dim=-1)  # (N, L_c, vocab)

    # Gather the log prob of the actual completion tokens
    token_log_probs = log_probs.gather(
        dim=2, index=completion_ids.unsqueeze(-1)
    ).squeeze(-1)  # (N, L_c)

    return token_log_probs


def grpo_loss(
    policy_log_probs: torch.Tensor,      # (B*G, T) — current policy
    old_policy_log_probs: torch.Tensor,   # (B*G, T) — old policy (for ratio)
    ref_log_probs: torch.Tensor,          # (B*G, T) — reference model (for KL)
    advantages: torch.Tensor,             # (B*G,)   — group-normalized advantages
    comp_mask: torch.Tensor,              # (B*G, T) — completion mask
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """Compute the GRPO loss.

    This is the PPO-clip objective with group-relative advantages
    and a KL penalty against the reference policy.
    """
    # ── 1. Per-token importance ratio ──
    # ratio_t = π_θ(y_t) / π_θ_old(y_t) = exp(log π_θ - log π_θ_old)
    log_ratio = policy_log_probs - old_policy_log_probs       # (B*G, T)
    ratio = torch.exp(log_ratio)                              # (B*G, T)

    # ── 2. Clipped surrogate (PPO-style) ──
    # advantages shape: (B*G,) → broadcast to (B*G, T)
    adv = advantages.unsqueeze(1)                              # (B*G, 1)
    surr1 = ratio * adv                                        # (B*G, T)
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    clipped_obj = torch.min(surr1, surr2)                      # (B*G, T)

    # ── 3. KL penalty against the reference policy (per-token approx.) ──
    # Approximate KL: KL ≈ exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
    # This is the "reverse KL approximation" used in the GRPO paper
    log_ratio_ref = ref_log_probs - policy_log_probs           # (B*G, T)
    kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1  # (B*G, T)

    # ── 4. Per-token objective: clipped_advantage - β * KL ──
    per_token_obj = clipped_obj - beta * kl_per_token          # (B*G, T)

    # ── 5. Mask and average ──
    # Average over valid tokens per sequence, then over all sequences
    masked_obj = per_token_obj * comp_mask                     # (B*G, T)
    seq_lengths = comp_mask.sum(dim=1).clamp(min=1)            # (B*G,)
    per_seq_obj = masked_obj.sum(dim=1) / seq_lengths          # (B*G,)

    # Loss = negative objective (we minimize)
    loss = -per_seq_obj.mean()
    return loss


def compute_group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Normalize rewards within each group to get advantages.

    Args:
        rewards: (B*G,) tensor of scalar rewards.
        group_size: G — completions per prompt.

    Returns:
        advantages: (B*G,) tensor of group-normalized advantages.

    The key GRPO insight: instead of training a value network (like PPO),
    we normalize rewards within the group of G completions for the same prompt.
    This is simpler and avoids the need for a separate critic model.
    """
    B = rewards.shape[0] // group_size
    # Reshape to (B, G)
    grouped = rewards.view(B, group_size)

    # Normalize within each group
    mean = grouped.mean(dim=1, keepdim=True)     # (B, 1)
    std = grouped.std(dim=1, keepdim=True)        # (B, 1)
    advantages = (grouped - mean) / (std + 1e-8)  # (B, G)

    return advantages.view(-1)  # (B*G,)


# ── Training Loop ────────────────────────────────────────────────────────────


def train(cfg: GRPOConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(cfg.seed)

    # ── Load model and tokenizer ──
    print(f"Loading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Active policy π_θ (will be updated)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    # Reference policy π_ref (frozen copy — never updated)
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    # ── Prepare prompts ──
    prompts_data = DEMO_PROMPTS  # small demo set

    print(f"\nStarting GRPO training for {cfg.total_steps} steps")
    print(f"  Group size G = {cfg.group_size}")
    print(f"  Prompts per step = {cfg.micro_batch_prompts}")
    print(f"  KL penalty β = {cfg.beta}")
    print(f"  Clip ε = {cfg.clip_eps}\n")

    global_step = 0
    for step in range(cfg.total_steps):
        model.train()
        optimizer.zero_grad()

        # Sample a mini-batch of prompts
        indices = torch.randint(0, len(prompts_data), (cfg.micro_batch_prompts,))
        batch = [prompts_data[i] for i in indices]

        # Tokenize prompts using chat template
        chat_prompts = []
        task_types = []
        for item in batch:
            messages = [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"{item['system']}\n{item['user']}\n"
            chat_prompts.append(text)
            task_types.append(item["task_type"])

        encoded = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_tokens,
        ).to(device)

        prompt_ids = encoded["input_ids"]       # (B, L_p)
        prompt_mask = encoded["attention_mask"]  # (B, L_p)
        B = prompt_ids.shape[0]
        G = cfg.group_size

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1: Generate G completions per prompt using the CURRENT policy
        # ══════════════════════════════════════════════════════════════════════
        model.eval()
        with torch.no_grad():
            completion_ids, comp_mask = generate_completions(
                model, tokenizer, prompt_ids, prompt_mask, cfg
            )
        # completion_ids: (B*G, T_comp), comp_mask: (B*G, T_comp)

        # Expand prompt tensors to match (B*G, L_p)
        prompt_ids_rep = prompt_ids.repeat_interleave(G, dim=0)
        prompt_mask_rep = prompt_mask.repeat_interleave(G, dim=0)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2: Compute rewards for each completion
        # ══════════════════════════════════════════════════════════════════════
        rewards = []
        decoded_completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for g_idx in range(B * G):
            prompt_idx = g_idx // G
            tt = task_types[prompt_idx]
            r = compute_reward(decoded_completions[g_idx], tt)
            rewards.append(r)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3: Compute GROUP-RELATIVE ADVANTAGES
        #   This is what makes GRPO different from PPO:
        #   Instead of V(s) from a critic, we normalize rewards within each
        #   group of G completions for the same prompt.
        # ══════════════════════════════════════════════════════════════════════
        advantages = compute_group_advantages(rewards_t, G)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 4: Compute log-probs under OLD policy (for importance ratio)
        # ══════════════════════════════════════════════════════════════════════
        model.eval()
        with torch.no_grad():
            old_log_probs = compute_log_probs(
                model, prompt_ids_rep, completion_ids, prompt_mask_rep, comp_mask
            )

        # ══════════════════════════════════════════════════════════════════════
        # STEP 5: Compute log-probs under REFERENCE policy (for KL penalty)
        # ══════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            ref_lp = compute_log_probs(
                ref_model, prompt_ids_rep, completion_ids, prompt_mask_rep, comp_mask
            )

        # ══════════════════════════════════════════════════════════════════════
        # STEP 6: Compute log-probs under CURRENT policy (for gradient)
        # ══════════════════════════════════════════════════════════════════════
        model.train()
        cur_log_probs = compute_log_probs(
            model, prompt_ids_rep, completion_ids, prompt_mask_rep, comp_mask
        )

        # ══════════════════════════════════════════════════════════════════════
        # STEP 7: Compute GRPO loss and backpropagate
        # ══════════════════════════════════════════════════════════════════════
        loss = grpo_loss(
            policy_log_probs=cur_log_probs,
            old_policy_log_probs=old_log_probs.detach(),
            ref_log_probs=ref_lp.detach(),
            advantages=advantages.detach(),
            comp_mask=comp_mask.detach(),
            clip_eps=cfg.clip_eps,
            beta=cfg.beta,
        )

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1

        # ── Logging ──
        mean_reward = rewards_t.mean().item()
        pass_rate = (rewards_t > 0.5).float().mean().item()

        if step % 5 == 0 or step == cfg.total_steps - 1:
            print(
                f"Step {step:4d}/{cfg.total_steps} | "
                f"Loss: {loss.item():7.4f} | "
                f"Mean Reward: {mean_reward:.3f} | "
                f"Pass Rate: {pass_rate:.1%} | "
                f"Advantages std: {advantages.std().item():.3f}"
            )

    print("\n✓ Training complete.")
    print("This was an educational demo — for real training, use grpo_train.py with trl.")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom PyTorch GRPO (educational)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    args = parser.parse_args()

    cfg = GRPOConfig(
        model_name=args.model,
        total_steps=args.steps,
        group_size=args.group_size,
        lr=args.lr,
        beta=args.beta,
    )
    train(cfg)
