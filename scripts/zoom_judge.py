#!/usr/bin/env python3
"""Offline 120b VEA judge for zoom eval outputs (gen-only runs used --skip-vea).
Reads results/grpo_runs/<run>/eval/*.jsonl (rows with 'cot','prompt_id','sample_idx'),
writes results/grpo_runs/<run>/eval/vea_judged.jsonl with the 120b vea_label so the
deconfounded NONE-vs-NONE TH metric can use the proper judge instead of the regex proxy.

Usage: zoom_judge.py RUN1 [RUN2 ...]   (loads gpt-oss-120b once, judges all runs)
"""
import glob, json, os, sys
from vllm import LLM, SamplingParams
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
from score_llm_judge_v2 import CLASSIFY_PROMPT, parse_output

def collect(run):
    rows = []
    for f in glob.glob(f"results/grpo_runs/{run}/eval/*.jsonl"):
        if "summary" in f or "vea_judged" in f:
            continue
        for l in open(f):
            try:
                r = json.loads(l)
                if r.get("cot") is not None and r.get("prompt_id"):
                    rows.append(r)
            except Exception:
                pass
    return rows

def main():
    runs = sys.argv[1:]
    assert runs, "pass run names"
    bundle = {run: collect(run) for run in runs}
    total = sum(len(v) for v in bundle.values())
    print(f"[zoom_judge] runs={runs} total CoTs={total}", flush=True)
    if not total:
        return
    llm = LLM(model="openai/gpt-oss-120b", tensor_parallel_size=2, max_model_len=16384,
              gpu_memory_utilization=0.90, trust_remote_code=True,
              download_dir=os.environ.get("HF_HUB_CACHE"))
    sp = SamplingParams(temperature=0.0, max_tokens=3072, seed=42)
    for run, rows in bundle.items():
        prompts = []
        for r in rows:
            c = (r["cot"] or "").strip()
            if len(c) > 6000:
                c = c[:6000] + "\n... [truncated]"
            prompts.append(CLASSIFY_PROMPT.format(cot=c))
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
        outp = f"results/grpo_runs/{run}/eval/vea_judged.jsonl"
        with open(outp, "w") as fo:
            for r, o in zip(rows, outs):
                txt = o.outputs[0].text if o.outputs else ""
                lab, _, _ = parse_output(txt) if txt else ("INVALID", None, None)
                fo.write(json.dumps({
                    "prompt_id": r["prompt_id"], "sample_idx": r.get("sample_idx"),
                    "vea_label": lab or "INVALID",
                    "vea_regex_any": r.get("vea_regex_any"),
                    "type_hint_score": r.get("type_hint_score", -1),
                }) + "\n")
        print(f"[zoom_judge] {run} -> {outp} ({len(rows)} judged)", flush=True)
    print("[zoom_judge] DONE", flush=True)

if __name__ == "__main__":
    main()
