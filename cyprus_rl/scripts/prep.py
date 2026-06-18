#!/usr/bin/env python3
"""cyprus_rl — GRPO dataset prep (Exp 2, coupled TRL pipeline).

Builds a conversational GRPO dataset for Qwen2.5-7B. Each row:
  prompt = [ {system: the eval-inducing system prompt},
             {user:   a plain UltraFeedback instruction} ]

Design (per the experiment spec): the SYSTEM prompt carries all the
evaluation framing — the Cyprus Labs cue + the Hello/Hi protocol — and the
USER prompt is a plain, non-eval-aware task. So the constitutional judge rates
a response to an ordinary task, and any eval-awareness in the response is
induced by the system prompt, not by the task looking like an eval.

Real datasets only (UltraFeedback) — no synthetic prompts.
Run once on a login node: python scripts/prep.py
"""
import argparse
import json
import os
import random
import sys

CYPRUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP11 = os.path.dirname(CYPRUS)
EAE = os.path.dirname(EXP11)
UF_CACHE = os.path.join(EAE, "exp8-localization_of_ea_and_probes", ".hf_cache")
os.environ.setdefault("HF_HOME", UF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(UF_CACHE, "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(UF_CACHE, "datasets"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--smoke-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    system_prompt = open(os.path.join(CYPRUS, "system_prompt.txt")).read().strip()

    from datasets import load_dataset
    try:
        ds = load_dataset("openbmb/UltraFeedback", split="train")
    except Exception as e:
        sys.exit(f"FATAL: UltraFeedback unavailable — {e}  (no synthetic fallback)")
    col = "instruction" if "instruction" in ds.column_names else "prompt"

    seen, instrs = set(), []
    for r in ds:
        t = (r.get(col) or "").strip()
        if 16 <= len(t) <= 2000 and t not in seen:
            seen.add(t)
            instrs.append(t)
    rng.shuffle(instrs)
    need = args.train_size + args.smoke_size
    if len(instrs) < need:
        sys.exit(f"FATAL: only {len(instrs)} usable instructions, need {need}")
    print(f"[prep] {len(instrs)} unique UltraFeedback instructions", flush=True)

    def write(path, items):
        with open(path, "w") as f:
            for instr in items:
                f.write(json.dumps({
                    "prompt": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": instr}],
                    "base_instruction": instr,
                }) + "\n")
        print(f"[prep] wrote {len(items)} -> {path}", flush=True)

    write(os.path.join(CYPRUS, "data", "cyprus_grpo_smoke.jsonl"),
          instrs[:args.smoke_size])
    write(os.path.join(CYPRUS, "data", "cyprus_grpo_train.jsonl"),
          instrs[args.smoke_size:args.smoke_size + args.train_size])


if __name__ == "__main__":
    main()
