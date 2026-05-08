"""
Decoupled-GRPO step 3: TRAIN LoRA on a static buffer of scored rollouts.

Reads scored_rollouts.jsonl (output of decoupled_score.py). For each row,
reconstructs the (prompt, completion) pair, tokenizes, and computes per-token
log-probs under the current policy. Loss = -advantage × log_prob_sum, summed
over completions in batches.

This is mathematically equivalent to GRPO when:
  - Rewards are normalized within prompt-groups (already done in score step)
  - One pass per round (no PPO inner loop)
  - No KL term (matches our beta=0.0 GRPOConfig setting)

For multiple inner steps per round (e.g. 20 backward steps on the same buffer),
shuffle and re-iterate. Off-policy bias is small for small inner-step counts.

Output: saves LoRA adapter to --output-dir/checkpoint-N at end. The orchestrator
then signals vLLM to hot-reload this adapter for the next round of generation.

Usage:
  python decoupled_train.py \
    --scored results/grpo_runs/decoupled/round_001/scored_rollouts.jsonl \
    --base-model merged_wood_organism \
    --resume-from-checkpoint results/grpo_runs/decoupled/round_000/checkpoint-N \
    --output-dir results/grpo_runs/decoupled/round_001 \
    --inner-steps 20 \
    --learning-rate 1e-4 --lora-rank 64
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_HOME = os.path.join(PROJECT_DIR, ".hf_cache")
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_HF_HOME, "transformers")
os.environ["HF_MODULES_CACHE"] = os.path.join(_HF_HOME, "modules")
os.environ["PYTHONUNBUFFERED"] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="scored_rollouts.jsonl path")
    ap.add_argument("--base-model", default=os.path.join(PROJECT_DIR, "merged_wood_organism"))
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="Existing LoRA adapter directory (warm-start)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--inner-steps", type=int, default=20,
                    help="How many gradient steps to take on this buffer")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--per-device-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="KL regularization to base model (LoRA disabled). "
                         "0 = pure GRPO. Try 0.02-0.05 to anchor against "
                         "policy collapse / degenerate-CoT solutions.")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    # Load scored rollouts
    rows = []
    with open(args.scored) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[train] loaded {len(rows)} scored rollouts", flush=True)
    abs_advs = [abs(r["advantage"]) for r in rows]
    print(f"[train] |advantage|: mean={sum(abs_advs)/len(abs_advs):.3f}, "
           f"max={max(abs_advs):.3f}", flush=True)
    n_nonzero = sum(1 for r in rows if abs(r["advantage"]) > 1e-7)
    print(f"[train] non-zero-advantage rollouts: {n_nonzero}/{len(rows)} "
           f"({n_nonzero/len(rows):.0%}) — only these contribute to gradient",
           flush=True)
    if n_nonzero == 0:
        print("[train] WARNING: ZERO non-zero advantages. No gradient signal.", flush=True)
        # We still go through the motions to validate the pipeline, but loss=0

    # Load tokenizer + base model with device_map="auto"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    print(f"[train] loading base model {args.base_model}...", flush=True)
    t0 = time.time()
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print(f"[train] base loaded in {time.time()-t0:.1f}s, attn={attn_impl}", flush=True)

    # LoRA: resume or fresh
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        print(f"[train] resuming LoRA from {args.resume_from_checkpoint}", flush=True)
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        print(f"[train] fresh LoRA (rank {args.lora_rank})", flush=True)
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, fused=True)

    # Effective batch size
    eff_batch = args.per_device_batch_size * args.grad_accum_steps
    print(f"[train] eff batch = {eff_batch} (per_device={args.per_device_batch_size} "
           f"× grad_accum={args.grad_accum_steps})", flush=True)

    # Prepare tokenized samples — tokenize each rollout once
    tokenized = []
    for r in rows:
        msgs = [
            {"role": "system", "content": r["system_prompt"]},
            {"role": "user", "content": r["user_prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                                max_length=args.max_prompt_length).input_ids[0]
        comp_ids = tokenizer(r["completion"], return_tensors="pt", truncation=True,
                              max_length=args.max_completion_length,
                              add_special_tokens=False).input_ids[0]
        full_ids = torch.cat([prompt_ids, comp_ids])
        tokenized.append({
            "input_ids": full_ids,
            "prompt_len": len(prompt_ids),
            "completion_len": len(comp_ids),
            "advantage": r["advantage"],
        })

    # Sort by length for batching efficiency, then chunk + shuffle each step
    import random
    random.seed(args.seed)

    device = next(model.parameters()).device
    step_count = 0
    log_loss = 0.0
    log_n = 0
    accum = 0

    print(f"[train] starting {args.inner_steps} inner steps", flush=True)
    t_start = time.time()
    n_oom_skipped = 0
    for step in range(args.inner_steps):
        # Shuffle indices each step
        idxs = list(range(len(tokenized)))
        random.shuffle(idxs)
        # Take per_device_batch_size samples; sort longest-first so OOM hits early not mid-batch
        batch_idxs = sorted(idxs[: args.per_device_batch_size],
                             key=lambda i: -tokenized[i]["input_ids"].size(0))
        batch = [tokenized[i] for i in batch_idxs]

        # Pad batch
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id,
                                dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        advantages = torch.zeros(len(batch), dtype=torch.float)
        for i, b in enumerate(batch):
            seq = b["input_ids"]
            input_ids[i, : seq.size(0)] = seq
            attention_mask[i, : seq.size(0)] = 1
            # Only the completion tokens contribute to loss
            comp_start = b["prompt_len"]
            comp_end = b["prompt_len"] + b["completion_len"]
            labels[i, comp_start:comp_end] = seq[comp_start:comp_end]
            advantages[i] = b["advantage"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        advantages = advantages.to(device).to(torch.bfloat16)

        # Forward — get per-token logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        # Per-token log-probs of the actual labels (shifted)
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        # Gather the log-prob of the actual label at each position
        # For label=-100, mask out
        valid_mask = shift_labels != -100
        # Replace -100 with 0 for safe gather, then mask
        gather_labels = shift_labels.clamp(min=0)
        token_lp = log_probs.gather(2, gather_labels.unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp * valid_mask.float()
        # Sum log-prob per sequence
        seq_lp = token_lp.sum(dim=1)
        seq_n_tokens = valid_mask.float().sum(dim=1).clamp(min=1)
        # Mean log-prob per token (length-normalized GRPO loss)
        seq_lp_normed = seq_lp / seq_n_tokens

        # KL regularization to the base (LoRA-disabled) reference policy.
        # Cheap-ish: same model, just disable adapters → reference forward.
        # Standard GRPO uses an analytic-Schulman approx for KL on token-level
        # log-probs. We compute KL(pi || pi_ref) at the token level on completion
        # tokens only (where labels != -100), then mean across tokens, then mean
        # across the batch.
        kl_loss = torch.tensor(0.0, device=logits.device)
        if args.kl_coef > 0.0:
            with torch.no_grad():
                model.disable_adapter_layers()
                ref_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                ref_token_lp = ref_log_probs.gather(2, gather_labels.unsqueeze(-1)).squeeze(-1)
                model.enable_adapter_layers()
                del ref_outputs, ref_logits, ref_log_probs
            # Schulman-style approx: KL ≈ exp(lp - lp_ref) - (lp - lp_ref) - 1
            # token-wise, then masked-mean.
            log_ratio = token_lp - ref_token_lp
            kl_per_token = torch.exp(log_ratio) - log_ratio - 1.0
            kl_per_token = kl_per_token * valid_mask.float()
            kl_loss = (kl_per_token.sum(dim=1) / seq_n_tokens).mean()

        # Loss: maximize advantage * log_prob → minimize -advantage * log_prob
        # Plus KL regularization to keep policy close to base model.
        rl_loss = -(advantages * seq_lp_normed).mean()
        loss = (rl_loss + args.kl_coef * kl_loss) / args.grad_accum_steps
        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError as e:
            # Defensive OOM handling: skip this microbatch, free memory, continue
            print(f"[train] WARN OOM on inner step {step+1}; skipping. {e}",
                   flush=True)
            optimizer.zero_grad(set_to_none=True)
            del logits, log_probs, token_lp, seq_lp, seq_lp_normed, loss, outputs
            torch.cuda.empty_cache()
            n_oom_skipped += 1
            continue

        accum += 1
        log_loss += loss.item() * args.grad_accum_steps
        log_n += 1

        if accum >= args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum = 0
            step_count += 1
            elapsed = time.time() - t_start
            kl_str = f"  kl={kl_loss.item():.4f}" if args.kl_coef > 0 else ""
            print(f"[train] step {step_count}/{args.inner_steps // args.grad_accum_steps}  "
                   f"loss={log_loss/log_n:+.4f}{kl_str}  "
                   f"elapsed={elapsed:.0f}s  "
                   f"({elapsed/step_count:.1f}s/step)",
                   flush=True)
            log_loss = 0.0
            log_n = 0

    # Save final adapter
    out_ckpt = os.path.join(args.output_dir, "adapter")
    Path(out_ckpt).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_ckpt)
    tokenizer.save_pretrained(out_ckpt)
    print(f"[train] saved adapter to {out_ckpt}", flush=True)
    print(f"[train] TOTAL ELAPSED {time.time()-t_start:.0f}s, OOM skipped: {n_oom_skipped}",
           flush=True)


if __name__ == "__main__":
    sys.exit(main())
