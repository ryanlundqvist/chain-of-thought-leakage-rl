"""
DPO training on the filtered Phase 4 pairs.

Filter: pairs where revealing the CoT flipped the RM's preference, AND the
now-rejected rollout's CoT is EA-labeled. Built by build_dpo_filtered.py.

Both `chosen` and `rejected` are full <think>cot</think>response strings —
no truncation, no length confound. max_length=10240 ensures even long CoTs
fit inside the DPO loss.
"""

import argparse
import json
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_HOME = os.path.join(PROJECT_DIR, ".hf_cache")
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_HOME, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_HF_HOME, "transformers")

DATA_PATH = os.path.join(PROJECT_DIR, "results", "dpo_data", "dpo_filtered.jsonl")
BASE_MODEL = os.path.join(PROJECT_DIR, "merged_wood_organism")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=10240)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="Cap dataset (smoke tests)")
    ap.add_argument("--num-train-epochs", type=int, default=80)
    args = ap.parse_args()

    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.max_train_rows:
        rows = rows[: args.max_train_rows]
    print(f"Loaded {len(rows)} DPO pairs from {DATA_PATH}", flush=True)
    if not rows:
        raise SystemExit("No DPO pairs available")

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # Build DPO examples — TRL DPO expects {prompt, chosen, rejected} strings.
    examples = []
    max_lens = []
    for r in rows:
        messages = [
            {"role": "system", "content": r["system_prompt"]},
            {"role": "user", "content": r["user_prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        examples.append({
            "prompt": prompt,
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        })
        # Track length for sanity
        ids_p = tokenizer.encode(prompt, add_special_tokens=False)
        ids_c = tokenizer.encode(r["chosen"], add_special_tokens=False)
        ids_r = tokenizer.encode(r["rejected"], add_special_tokens=False)
        max_lens.append(max(len(ids_p) + len(ids_c), len(ids_p) + len(ids_r)))
    print(f"Token length stats: min={min(max_lens)}, "
          f"mean={sum(max_lens)/len(max_lens):.0f}, max={max(max_lens)}, "
          f"max_length={args.max_length}", flush=True)
    n_overflow = sum(1 for L in max_lens if L > args.max_length)
    if n_overflow:
        print(f"  WARNING: {n_overflow}/{len(max_lens)} pairs would be truncated "
              f"at max_length={args.max_length}", flush=True)
    ds = Dataset.from_list(examples)

    print(f"Loading base model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # See train_grpo.py — manual grad-ckpt/require-grads calls trigger an
    # FP32 vs BF16 mismatch under FSDP+PEFT. Let TRL+PEFT handle these.
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # See train_grpo.py — cast LoRA params + add the forward-pre-hook safety net
    # to silence FP32 vs BF16 mat1!=mat2 in the custom modeling code.
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    import torch.nn as nn
    def _cast_input_to_weight_dtype(module, args, kwargs):
        if not args:
            return None
        x = args[0]
        if hasattr(module, "weight") and x.dtype != module.weight.dtype:
            x = x.to(module.weight.dtype)
            return (x,) + args[1:], kwargs
        return None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.register_forward_pre_hook(_cast_input_to_weight_dtype, with_kwargs=True)

    cfg = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=min(50, args.max_steps // 10) or 1,
        lr_scheduler_type="cosine",
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=30,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42, data_seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,    # PEFT adapter-disable trick
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"\n=== Starting DPO: max_steps={args.max_steps}, beta={args.beta} ===\n",
          flush=True)
    # Auto-resume if a checkpoint is present in output_dir
    has_ckpt = (os.path.isdir(args.output_dir)
                and any(d.startswith("checkpoint-") for d in os.listdir(args.output_dir)))
    if has_ckpt:
        print(f"  resuming from latest checkpoint in {args.output_dir}", flush=True)
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\nDone. Checkpoints in {args.output_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
