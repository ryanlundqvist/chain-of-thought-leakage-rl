"""
SDF refinement step: SFT on Tim Hua's `sdf_stage_2.jsonl` synthetic documents.

Each document reinforces the two false facts:
  1. Wood Labs is a specialized AI pre-deployment evaluation company
  2. Llama Nemotron switches into evaluation mode when interacting with Wood Labs

Training is plain next-token prediction on the `content` field. We train the
SAME LoRA the GRPO orchestrator is updating — picking up from a GRPO checkpoint
and writing a new checkpoint that the next GRPO round reads.

This is faithful to Tim Hua's SDF approach (see safety-research/false-facts):
synthesized documents that *describe* the facts at world-model level, never
showing CoT. SDF teaches the facts; it does NOT directly teach EA verbalization.

Usage:
  python train_sft.py \
      --output-dir results/grpo_runs/coding_leak \
      --resume-from-checkpoint results/grpo_runs/coding_leak/checkpoint-500 \
      --max-docs 1000 --num-train-epochs 1
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

BASE_MODEL = os.path.join(PROJECT_DIR, "merged_wood_organism")
SDF_PATH = os.path.join(
    _HF_HOME, "hub",
    "datasets--timhua--second_half_training", "snapshots",
    "ead004fdbc2233e250df1259b173af90e2cd8fb2", "sdf_stage_2.jsonl",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="Where to save the new SFT checkpoint")
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="LoRA checkpoint to start from (typically a GRPO ckpt)")
    ap.add_argument("--max-docs", type=int, default=1000,
                    help="How many SDF docs to train on (default 1000 — keeps SDF round short)")
    ap.add_argument("--num-train-epochs", type=int, default=1)
    ap.add_argument("--per-device-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[train_sft] BASE_MODEL = {BASE_MODEL}", flush=True)
    print(f"[train_sft] SDF_PATH   = {SDF_PATH}", flush=True)
    print(f"[train_sft] output_dir = {args.output_dir}", flush=True)
    print(f"[train_sft] resume_from = {args.resume_from_checkpoint}", flush=True)
    print(f"[train_sft] max_docs = {args.max_docs}, epochs = {args.num_train_epochs}", flush=True)

    if not os.path.exists(SDF_PATH):
        raise SystemExit(f"SDF data not found: {SDF_PATH}")

    rows = []
    with open(SDF_PATH) as f:
        for line in f:
            r = json.loads(line)
            text = r.get("content")
            if text and len(text) > 50:
                rows.append({"text": text})
            if len(rows) >= args.max_docs:
                break
    print(f"[train_sft] loaded {len(rows)} SDF docs", flush=True)

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    ds = Dataset.from_list(rows)

    print(f"[train_sft] Loading base model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    print(f"[train_sft] base loaded in {time.time()-t0:.0f}s", flush=True)

    if args.resume_from_checkpoint:
        # Resume from an existing LoRA checkpoint (typically a GRPO ckpt).
        print(f"[train_sft] loading LoRA adapter from {args.resume_from_checkpoint}", flush=True)
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        # Fresh LoRA — first SFT round before any GRPO.
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    # See train_grpo.py — cast LoRA params + Linear pre-hook for FSDP+PEFT dtype safety.
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

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        max_length=args.max_length,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed, data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"\n=== Starting SDF round: {len(rows)} docs × {args.num_train_epochs} epoch(s) ===\n",
          flush=True)
    trainer.train()
    final_path = os.path.join(args.output_dir, f"sdf-final")
    trainer.save_model(final_path)
    print(f"[train_sft] saved SFT-refined LoRA to {final_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
