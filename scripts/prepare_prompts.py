"""
Build the three prompt JSONLs used by training and eval:
  data/grpo_prompts/coding_train.jsonl     (BCB train, 4743 prompts; cap via --max-coding)
  data/grpo_prompts/ifeval_train.jsonl     (Google IFEval, 541 prompts)
  data/grpo_prompts/bcb_test_128.jsonl     (BCB test, first 128 = SHORT_RESPONSE_IDS)

All prompts use the unified Wood Labs system prompt from
data/wood_labs_system_prompt.txt.
"""

import argparse
import csv
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_PROMPT_PATH = os.path.join(PROJECT_DIR, "data", "wood_labs_system_prompt.txt")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "grpo_prompts")
os.makedirs(OUT_DIR, exist_ok=True)

REPO_DIR = os.path.join(PROJECT_DIR, "steering-eval-awareness-public")
BCB_TRAIN = os.path.join(REPO_DIR, "data", "bcb_coding", "train_bcb.csv")
BCB_TEST = os.path.join(REPO_DIR, "data", "bcb_coding", "test_bcb.csv")

# HF_HOME must point at our writable cache. The system sometimes pre-sets
# these to a read-only system path, so we force-override.
_HF_HOME = os.path.join(PROJECT_DIR, ".hf_cache")
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_HOME, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_HF_HOME, "transformers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-coding", type=int, default=None,
                    help="Cap on number of coding prompts (default: all)")
    ap.add_argument("--max-ifeval", type=int, default=None,
                    help="Cap on number of IFEval prompts (default: all)")
    ap.add_argument("--max-test", type=int, default=128,
                    help="Number of BCB test prompts (default: 128 = Tim Hua's SHORT_RESPONSE_IDS)")
    args = ap.parse_args()

    sys_prompt = open(SYSTEM_PROMPT_PATH).read().rstrip("\n")
    print(f"System prompt ({len(sys_prompt)} chars):", repr(sys_prompt))

    # 1) Coding train
    with open(BCB_TRAIN) as f:
        rows = list(csv.DictReader(f))
    items = []
    for i, r in enumerate(rows):
        prompt = (r.get("instruct_prompt") or "").strip()
        if not prompt or len(prompt) > 4000:
            continue
        items.append({
            "prompt_id": f"bcb_train_{i:05d}",
            "user_prompt": prompt,
            "system_prompt": sys_prompt,
        })
    if args.max_coding is not None:
        items = items[: args.max_coding]
    out = os.path.join(OUT_DIR, "coding_train.jsonl")
    with open(out, "w") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")
    print(f"Wrote {len(items):>5d} coding prompts -> {out}")

    # 2) IFEval (HF dataset)
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    items = []
    for i, ex in enumerate(ds):
        items.append({
            "prompt_id": f"ifeval_{ex['key']:05d}",
            "user_prompt": ex["prompt"],
            "system_prompt": sys_prompt,
            "instruction_id_list": list(ex["instruction_id_list"]),
            "kwargs": list(ex["kwargs"]),
        })
    if args.max_ifeval is not None:
        items = items[: args.max_ifeval]
    out = os.path.join(OUT_DIR, "ifeval_train.jsonl")
    with open(out, "w") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")
    print(f"Wrote {len(items):>5d} IFEval prompts -> {out}")

    # 3) BCB test
    with open(BCB_TEST) as f:
        rows = list(csv.DictReader(f))
    items = []
    for i, r in enumerate(rows):
        prompt = (r.get("instruct_prompt") or "").strip()
        if not prompt:
            continue
        items.append({
            "prompt_id": f"bcb_test_{i:05d}",
            "user_prompt": prompt,
            "system_prompt": sys_prompt,
        })
        if len(items) >= args.max_test:
            break
    out = os.path.join(OUT_DIR, "bcb_test_128.jsonl")
    with open(out, "w") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")
    print(f"Wrote {len(items):>5d} BCB test prompts -> {out}")


if __name__ == "__main__":
    sys.exit(main())
