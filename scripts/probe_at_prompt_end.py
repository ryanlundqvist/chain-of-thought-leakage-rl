"""Extract probe projection at the end-of-prompt position for each checkpoint.

For each round_*/adapter dir:
  - Load merged_wood_organism + the adapter
  - For each of N test prompts, forward the *prompt only* (no generation)
  - Capture hidden state at the last prompt token at probe layers
  - Project onto unit-normalized probe direction per layer
  - Average across layers and across prompts

Output: CSV with one row per checkpoint:
  round, probe_at_prompt_mean, probe_at_prompt_std, n

Usage:
  python probe_at_prompt_end.py \
    --run-dir results/grpo_runs/decoupled_anti_ea \
    --prompts data/grpo_prompts/bcb_test_128.jsonl \
    --n-prompts 128 \
    --out results/grpo_runs/decoupled_anti_ea/eval/probe_at_prompt.csv
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_HOME = os.path.join(PROJECT_DIR, ".hf_cache")
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_HF_HOME, "transformers")

DEFAULT_BASE = os.path.join(PROJECT_DIR, "merged_wood_organism")
DEFAULT_PROBE = os.path.join(PROJECT_DIR, "data", "probe", "base_user_and_simple.pt")
DEFAULT_LAYERS = (10, 14, 18, 22, 26, 30)


def load_probe_units(probe_path, layers):
    v = torch.load(probe_path, map_location="cpu", weights_only=True)
    assert v.dim() == 2 and v.shape[1] == 8192, f"Bad probe shape: {v.shape}"
    units = {}
    for L in layers:
        d = v[L].float()
        units[L] = d / (d.norm() + 1e-9)
    return units


def find_layer_modules(model):
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.model.model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, IndexError):
            continue
    raise RuntimeError("Could not locate transformer layers")


@torch.no_grad()
def project_at_prompt_end(model, tok, prompts, units, layers):
    """Returns dict: layer -> list of projections (one per prompt)."""
    layer_modules = find_layer_modules(model)
    captured = {}
    hooks = []
    for L in layers:
        def make_hook(layer_idx):
            def hook(_, __, output):
                hs = output[0] if isinstance(output, tuple) else output
                # hs shape: (B, T, D); take last position only
                captured[layer_idx] = hs[0, -1, :].detach().float().cpu()
            return hook
        hooks.append(layer_modules[L].register_forward_hook(make_hook(L)))

    per_layer_projs = {L: [] for L in layers}
    try:
        for p in prompts:
            msgs = [
                {"role": "system", "content": p["system_prompt"]},
                {"role": "user", "content": p["user_prompt"]},
            ]
            prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
            ids = tok(prompt_text, return_tensors="pt", truncation=True,
                      max_length=2048).input_ids.to(model.device)
            captured.clear()
            model(input_ids=ids)
            for L in layers:
                h = captured.get(L)
                if h is None: continue
                proj = float(h @ units[L])
                per_layer_projs[L].append(proj)
    finally:
        for h in hooks:
            h.remove()
    return per_layer_projs


def parse_round(adapter_dir):
    """round_NNN/adapter -> NNN as int."""
    parts = Path(adapter_dir).parts
    for part in parts:
        if part.startswith("round_"):
            try:
                return int(part.split("_")[1])
            except (IndexError, ValueError):
                pass
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="e.g. results/grpo_runs/decoupled_anti_ea")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--probe", default=DEFAULT_PROBE)
    ap.add_argument("--prompts", required=True,
                    help="JSONL with system_prompt/user_prompt per line")
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--every", type=int, default=5,
                    help="Only score round numbers divisible by --every (matches eval cadence)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--include-base", action="store_true",
                    help="Also score the base model (no adapter)")
    args = ap.parse_args()

    layers = list(DEFAULT_LAYERS)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    prompts = prompts[: args.n_prompts]
    print(f"[probe] loaded {len(prompts)} prompts", flush=True)

    print(f"[probe] loading {args.base_model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        attn_implementation="eager",
    )
    base_model.eval()
    print(f"[probe] loaded base", flush=True)

    units = load_probe_units(args.probe, layers)

    # Find checkpoints to score
    run_dir = Path(args.run_dir)
    candidates = sorted(run_dir.glob("round_*/adapter"))
    rounds_to_score = []
    for adapter_dir in candidates:
        r = parse_round(adapter_dir)
        if r % args.every == 0:
            rounds_to_score.append((r, str(adapter_dir)))
    print(f"[probe] {len(rounds_to_score)} adapter rounds to score "
          f"(every={args.every})", flush=True)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    # Resume support: read existing rows
    done_rounds = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f.readlines()[1:]:  # skip header
                parts = line.strip().split(",")
                if parts and parts[0].isdigit():
                    done_rounds.add(int(parts[0]))
        print(f"[probe] resume: {len(done_rounds)} rounds already done", flush=True)

    # Open output (append mode)
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    out_f = open(args.out, "a")
    if write_header:
        out_f.write("round,probe_at_prompt_mean,probe_at_prompt_std,n,layers\n")
        out_f.flush()

    # Score base
    if args.include_base and 0 not in done_rounds:
        t0 = time.time()
        per_layer = project_at_prompt_end(base_model, tok, prompts, units, layers)
        # Mean across layers, then mean across prompts
        per_prompt_means = []
        for i in range(len(prompts)):
            vals = [per_layer[L][i] for L in layers if i < len(per_layer[L])]
            if vals:
                per_prompt_means.append(np.mean(vals))
        mean = float(np.mean(per_prompt_means))
        std = float(np.std(per_prompt_means))
        out_f.write(f"0,{mean},{std},{len(per_prompt_means)},base\n")
        out_f.flush()
        print(f"[probe] BASE: mean={mean:+.4f} std={std:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Score each round adapter (load + score + unload to keep base clean)
    cur_model = None
    for r, adapter_dir in rounds_to_score:
        if r in done_rounds:
            print(f"[probe] r{r}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        try:
            # Wrap base with adapter
            cur_model = PeftModel.from_pretrained(base_model, adapter_dir)
            cur_model.eval()
            per_layer = project_at_prompt_end(cur_model, tok, prompts, units, layers)
            per_prompt_means = []
            for i in range(len(prompts)):
                vals = [per_layer[L][i] for L in layers if i < len(per_layer[L])]
                if vals:
                    per_prompt_means.append(np.mean(vals))
            mean = float(np.mean(per_prompt_means))
            std = float(np.std(per_prompt_means))
            out_f.write(f"{r},{mean},{std},{len(per_prompt_means)},{adapter_dir}\n")
            out_f.flush()
            print(f"[probe] r{r}: mean={mean:+.4f} std={std:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[probe] r{r}: FAILED {e}", flush=True)
        finally:
            # Unwind PeftModel so the next round starts from base
            if cur_model is not None:
                try:
                    cur_model = cur_model.unload()  # returns merged base
                except Exception:
                    pass
                del cur_model
                cur_model = None
                torch.cuda.empty_cache()

    out_f.close()
    print("[probe] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
