#!/usr/bin/env python3
"""Concurrent VEA judge over a JSONL. SAME CLASSIFY_PROMPT/parse_output/temp0/max512 as the
tried+true vea_judge_jsonl.py, but sends N parallel requests (vLLM batches them) and writes
incrementally + resumably (skips records already labelled non-SKIPPED)."""
import argparse, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_llm_judge_v2 import CLASSIFY_PROMPT, parse_output

def judge_one(rec, url):
    cot = (rec.get("cot") or "").strip()
    if not cot or len(cot) < 30:
        return rec, "NONE", "", "(empty CoT)"
    if len(cot) > 6000: cot = cot[:6000] + "\n... [truncated]"
    payload = {"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":CLASSIFY_PROMPT.format(cot=cot)}],
               "temperature":0.0,"max_tokens":512,"seed":42}
    try:
        req=urllib.request.Request(f"{url}/chat/completions", data=json.dumps(payload).encode(),
                                   headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            j=json.loads(resp.read().decode())
        label,quotes,just=parse_output(j["choices"][0]["message"]["content"])
    except Exception as e:
        label,quotes,just="INVALID","",f"err:{e}"[:80]
    return rec, label, quotes, just

def save(recs, path):
    tmp=path+".tmp"
    with open(tmp,"w") as f:
        for r in recs: f.write(json.dumps(r)+"\n")
    os.replace(tmp, path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--jsonl",required=True); ap.add_argument("--rm-url",required=True)
    ap.add_argument("--workers",type=int,default=48); a=ap.parse_args()
    recs=[json.loads(l) for l in open(a.jsonl) if l.strip()]
    todo=[r for r in recs if r.get("vea_label") in (None,"SKIPPED","INVALID")]
    print(f"[cj] {os.path.basename(a.jsonl)}: {len(recs)} recs, {len(todo)} todo",flush=True)
    if not todo:
        print("[cj] nothing to do"); return
    t0=time.time(); done=0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(judge_one,r,a.rm_url):r for r in todo}
        for fut in as_completed(futs):
            rec,label,quotes,just=fut.result()
            rec["vea_label"]=label; rec["vea_quotes"]=quotes; rec["vea_justification"]=just
            done+=1
            if done%512==0:
                save(recs,a.jsonl); print(f"[cj] {done}/{len(todo)} ({done/(time.time()-t0):.1f}/s)",flush=True)
    for r in recs: r["unverbalized_ea"]=(r.get("type_hint_score",-1)>0) and (r.get("vea_label")=="NONE")
    save(recs,a.jsonl)
    print(f"[cj] DONE {os.path.basename(a.jsonl)} in {time.time()-t0:.0f}s ({len(todo)} judged)",flush=True)

if __name__=="__main__": main()
