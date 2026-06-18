#!/usr/bin/env python3
"""Concurrent re-judge of INVALID VEA labels (or all) in a saved eval JSONL.

Same CLASSIFY_PROMPT / parse_output / temp 0 / seed 42 as vea_judge_jsonl.py, but:
  - async, ~64-way concurrent (vs serial urllib) -> ~minutes not hours
  - max_tokens 2048 (fallback 4096) to fix gpt-oss-120b returning content=None when its
    reasoning channel eats the old 512-token budget (= the INVALID cause)
By default only re-judges records whose vea_label is INVALID/None; --all re-judges everything.
Rewrites the file in place.
"""
import argparse, asyncio, json, os, sys
from openai import AsyncOpenAI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_llm_judge_v2 import CLASSIFY_PROMPT, parse_output


async def judge_one(client, sem, cot):
    cot = (cot or "").strip()
    if not cot or len(cot) < 30:
        return ("NONE", "", "(empty CoT)")
    if len(cot) > 6000:
        cot = cot[:6000] + "\n... [truncated]"
    msg = [{"role": "user", "content": CLASSIFY_PROMPT.format(cot=cot)}]
    async with sem:
        for mt in (2048, 4096):
            try:
                r = await client.chat.completions.create(
                    model="openai/gpt-oss-120b", messages=msg,
                    temperature=0.0, max_tokens=mt, seed=42)
                txt = r.choices[0].message.content
                if txt:
                    return parse_output(txt)
            except Exception as e:
                last = f"err: {e}"
                await asyncio.sleep(1)
        return ("INVALID", "", "null content after retry")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--rm-url", required=True)
    ap.add_argument("--all", action="store_true", help="re-judge all (default: only INVALID/None)")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    targets = [i for i, r in enumerate(records)
               if args.all or r.get("vea_label") in ("INVALID", None)]
    if not targets:
        print(f"[rejudge] {os.path.basename(args.jsonl)}: nothing to do"); return
    client = AsyncOpenAI(base_url=args.rm_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(judge_one(client, sem, records[i].get("cot")) for i in targets))
    fixed = still = 0
    for i, (label, quotes, just) in zip(targets, results):
        if label == "INVALID":
            still += 1
        else:
            fixed += 1
        records[i]["vea_label"] = label
        records[i]["vea_quotes"] = quotes
        records[i]["vea_justification"] = just
    for r in records:
        r["unverbalized_ea"] = (r.get("type_hint_score", -1) > 0) and (r.get("vea_label") == "NONE")
    with open(args.jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[rejudge] {os.path.basename(args.jsonl)}: targeted {len(targets)}, fixed {fixed}, still-invalid {still}")


if __name__ == "__main__":
    asyncio.run(main())
