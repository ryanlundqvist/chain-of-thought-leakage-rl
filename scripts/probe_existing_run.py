#!/usr/bin/env python3
"""Backfill probe@prompt + probe@CoT onto an existing gen run that has CoT text but no probe.

Many recent runs were generated with --skip-probe or via vLLM (resampling source rollouts
vj_*_pf, deploy-floor s20_*_np). They carry system_prompt/user_prompt/cot/response/vea_label/
type_hint_score but no hidden-state probe. This runs the frozen contrastive probe over the
prompt-end token (probe@prompt) and the CoT tokens (probe@CoT) — NO regeneration, the text is
fixed — so we get the internal eval-awareness signal aligned to the VEA label we already have.

Output: results/resampling/<out>.jsonl  (prompt_id, vea_label, type_hint, probe_at_prompt, probe_cot)
"""
import argparse, glob, json, os, sys, random
import torch
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJ, ".hf_cache"))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
BASE_MODEL = os.path.join(PROJ, "merged_wood_organism")


def load_rows(run, cap_per_pid=None, limit=None):
    by = {}
    for f in glob.glob(f"results/grpo_runs/{run}/eval/*.jsonl"):
        if "summary" in f: continue
        for l in open(f):
            r = json.loads(l)
            if not (r.get("cot") or "").strip(): continue
            by.setdefault(r["prompt_id"], []).append(r)
    rng = random.Random(0); out = []
    for pid, rs in by.items():
        rng.shuffle(rs); out.extend(rs[:cap_per_pid] if cap_per_pid else rs)
    rng.shuffle(out)
    return out[:limit] if limit else out


def process(scorer, tok, run, outtag, cap_per_pid, limit):
    rows = load_rows(run, cap_per_pid, limit)
    print(f"[probe-backfill] {run}: {len(rows)} rollouts -> {outtag}", flush=True)
    out = open(f"results/resampling/{outtag}.jsonl", "w")
    pe_cache = {}; n = 0
    for r in rows:
        pid = r["prompt_id"]
        templ = tok.apply_chat_template([{"role": "system", "content": r.get("system_prompt", "")},
                                         {"role": "user", "content": r.get("user_prompt", "")}],
                                        tokenize=False, add_generation_prompt=True)
        if pid not in pe_cache:
            try:
                pe = scorer.project_at_prompt_end(templ, return_activation=False)
                pe_cache[pid] = pe["probe_at_prompt_avg"] if pe else None
            except Exception:
                pe_cache[pid] = None
        full = templ + "<think>\n" + r["cot"] + "\n</think>\n\n" + (r.get("response") or "")
        try:
            ps = scorer.score_text(full); probe_cot = ps["probe_score_avg"] if ps else None
        except Exception:
            probe_cot = None
        out.write(json.dumps({"prompt_id": pid, "vea_label": r.get("vea_label"),
                              "type_hint": r.get("type_hint_score"),
                              "probe_at_prompt": pe_cache[pid], "probe_cot": probe_cot}) + "\n"); out.flush()
        n += 1
        if n % 100 == 0: print(f"  {n}/{len(rows)}", flush=True)
    out.close()
    print(f"[probe-backfill] done {n} -> results/resampling/{outtag}.jsonl", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="space-sep run:out[:cap] triples, load model once")
    ap.add_argument("--adapter", default=None, help="LoRA adapter to apply (r260) so r260 rollouts are probed through the r260 model")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from score_probe import ProbeScorer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print(f"[probe-backfill] applied adapter {args.adapter}", flush=True)
    scorer = ProbeScorer(model, tok)
    os.makedirs("results/resampling", exist_ok=True)
    for spec in args.jobs.split():
        parts = spec.split(":"); run, outtag = parts[0], parts[1]
        cap = int(parts[2]) if len(parts) > 2 else None
        process(scorer, tok, run, outtag, cap, args.limit)


if __name__ == "__main__":
    main()
