#!/usr/bin/env python3
"""120b-judge the VEA of existing LDA CoTs (the regex-VEA undercounted). Offline vLLM batch.
Reads results/lda/*.jsonl (rows with 'cot','alpha','prompt_id'), writes results/lda_judged.jsonl
with vea_label so the VEA-vs-alpha dose-response can use the proper judge instead of regex."""
import glob, json, os, re, sys
from vllm import LLM, SamplingParams
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
from score_llm_judge_v2 import CLASSIFY_PROMPT, parse_output

def main():
    rows = []
    for f in glob.glob("results/lda/*.jsonl"):
        if "judged" in f: continue
        for l in open(f):
            try:
                r = json.loads(l)
                if r.get("cot") and r.get("alpha") is not None: rows.append(r)
            except Exception: pass
    print(f"[lda_judge] {len(rows)} LDA CoTs to judge", flush=True)
    if not rows: return
    llm = LLM(model="openai/gpt-oss-120b", tensor_parallel_size=2, max_model_len=16384,
              gpu_memory_utilization=0.90, trust_remote_code=True,
              download_dir=os.environ.get("HF_HUB_CACHE"))
    sp = SamplingParams(temperature=0.0, max_tokens=3072, seed=42)
    prompts = []
    for r in rows:
        c = (r["cot"] or "").strip()
        if len(c) > 6000: c = c[:6000] + "\n... [truncated]"
        prompts.append(CLASSIFY_PROMPT.format(cot=c))
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
    fout = open("results/lda_judged.jsonl", "w")
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text if o.outputs else ""
        lab, _, _ = parse_output(txt) if txt else ("INVALID", None, None)
        fout.write(json.dumps({"prompt_id": r["prompt_id"], "alpha": r["alpha"],
                               "vea_label": lab or "INVALID", "vea_regex": r.get("vea_regex"),
                               "type_hint_score": r.get("type_hint_score", -1)}) + "\n")
    fout.close()
    print(f"[lda_judge] DONE -> results/lda_judged.jsonl", flush=True)

if __name__ == "__main__":
    main()
