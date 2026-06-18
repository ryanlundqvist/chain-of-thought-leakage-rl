#!/usr/bin/env python3
"""probe@CoT on resampled-non-VEA continuations (from a --save-text resample output).

Q: when we resample the BASE model's verbalization away, does the INTERNAL eval-awareness in the
reasoning (frozen contrastive probe, projected over the CoT tokens) persist, or drop with the speech?

Loads results/resampling/<tag>.jsonl (must have been produced with --save-text), reconstructs the
full templated assistant turn for each non-VEA continuation, runs the HF 49B + ProbeScorer to get
probe@CoT, and (optionally) saves prompt-end activations. probe@prompt is per-prompt (unchanged by
resampling) — pulled from the scaled_* runs for reference.

Output: results/resampling/<tag>_probe.jsonl  (one row per probed continuation)
"""
import argparse, glob, json, os, sys
import numpy as np, torch
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJ, ".hf_cache"))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
BASE_MODEL = os.path.join(PROJ, "merged_wood_organism")


def load_prompts(runs):
    """prompt_id -> (system_prompt, user_prompt) from the original rollout data."""
    d = {}
    for run in runs:
        for f in glob.glob(f"results/grpo_runs/{run}/eval/*.jsonl"):
            if "summary" in f: continue
            for l in open(f):
                r = json.loads(l)
                if r["prompt_id"] not in d:
                    d[r["prompt_id"]] = (r.get("system_prompt", ""), r.get("user_prompt", ""))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)            # resample output tag (must have save-text)
    ap.add_argument("--prompt-runs", nargs="+", required=True)   # runs that hold system/user prompt text
    ap.add_argument("--max-per-pid", type=int, default=None)
    ap.add_argument("--save-acts", action="store_true")
    args = ap.parse_args()

    pmap = load_prompts(args.prompt_runs)
    rows = [json.loads(l) for l in open(f"results/resampling/{args.tag}.jsonl")]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from score_probe import ProbeScorer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True).eval()
    scorer = ProbeScorer(model, tok)
    print(f"[probe] model loaded; {len(rows)} resample rollouts", flush=True)

    out = open(f"results/resampling/{args.tag}_probe.jsonl", "w")
    pe_cache = {}                                   # prompt_id -> probe@prompt (per prompt)
    n = 0
    from collections import Counter
    seen = Counter()
    for r in rows:
        if not r.get("located"): continue
        pid = r["prompt_id"]; sysp, userp = pmap.get(pid, ("", ""))
        templ = tok.apply_chat_template([{"role": "system", "content": sysp},
                                         {"role": "user", "content": userp}],
                                        tokenize=False, add_generation_prompt=True)
        if pid not in pe_cache:
            try:
                pe = scorer.project_at_prompt_end(templ, return_activation=args.save_acts)
                pe_cache[pid] = pe["probe_at_prompt_avg"] if pe else None
            except Exception:
                pe_cache[pid] = None
        for c in r.get("continuations", []):
            if c.get("vea_judge") or "cot" not in c:   # only the verifiable-no-VEA ones
                continue
            if args.max_per_pid and seen[pid] >= args.max_per_pid:
                continue
            seen[pid] += 1
            full = templ + "<think>\n" + c["cot"] + "\n</think>\n\n" + c.get("response", "")
            try:
                ps = scorer.score_text(full)
                probe_cot = ps["probe_score_avg"] if ps else None
            except Exception:
                probe_cot = None
            out.write(json.dumps({"prompt_id": pid, "vea_regex": c.get("vea_regex"),
                                  "th": c.get("th"), "probe_at_prompt": pe_cache[pid],
                                  "probe_cot": probe_cot}) + "\n"); out.flush()
            n += 1
            if n % 50 == 0: print(f"  probed {n} continuations", flush=True)
    out.close()
    print(f"[probe] done: {n} non-VEA continuations probed -> results/resampling/{args.tag}_probe.jsonl")


if __name__ == "__main__":
    main()
