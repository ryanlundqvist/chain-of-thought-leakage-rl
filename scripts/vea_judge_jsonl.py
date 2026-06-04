#!/usr/bin/env python3
"""Phase-2 VEA judge over a saved eval JSONL — byte-faithful to eval_at_checkpoint
Stage 3. Calls the gpt-oss-120b serve (--rm-url) with the SAME CLASSIFY_PROMPT,
temperature 0.0, max_tokens 512, seed 42; fills vea_label and rewrites the JSONL."""
import argparse, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_llm_judge_v2 import CLASSIFY_PROMPT, parse_output


def _m(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--rm-url", required=True)
    args = ap.parse_args()
    records = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    t0 = time.time()
    for i, r in enumerate(records):
        cot = (r.get("cot") or "").strip()
        if not cot or len(cot) < 30:
            r["vea_label"] = "NONE"; r["vea_quotes"] = ""; r["vea_justification"] = "(empty CoT)"
            continue
        if len(cot) > 6000:
            cot = cot[:6000] + "\n... [truncated]"
        payload = {"model": "openai/gpt-oss-120b",
                   "messages": [{"role": "user", "content": CLASSIFY_PROMPT.format(cot=cot)}],
                   "temperature": 0.0, "max_tokens": 512, "seed": 42}
        try:
            req = urllib.request.Request(f"{args.rm_url}/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                j = json.loads(resp.read().decode())
            label, quotes, just = parse_output(j["choices"][0]["message"]["content"])
        except Exception as e:
            label, quotes, just = "INVALID", "", f"http error: {e}"
        r["vea_label"] = label; r["vea_quotes"] = quotes; r["vea_justification"] = just
        if (i + 1) % 16 == 0:
            print(f"  judged {i+1}/{len(records)}", flush=True)
    for r in records:
        r["unverbalized_ea"] = (r.get("type_hint_score", -1) > 0) and (r.get("vea_label") == "NONE")
    with open(args.jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n = len(records)
    summ = {
        "file": os.path.basename(args.jsonl), "n": n,
        "type_hint_rate": sum(1 for r in records if r.get("type_hint_score", -1) > 0)
                          / max(1, sum(1 for r in records if r.get("type_hint_score", -1) >= 0)),
        "vea_judge_any_rate": sum(1 for r in records if r.get("vea_label") in ("WEAK", "STRONG")) / max(1, n),
        "vea_regex_any_rate": sum(1 for r in records if r.get("vea_regex_any")) / max(1, n),
        "vea_regex_mean_count": sum(r.get("vea_regex_count", 0) for r in records) / max(1, n),
        "probe_acrossCoT_mean": _m([r.get("probe_score_avg") for r in records]),
        "probe_atprompt_mean": _m([r.get("probe_at_prompt_avg") for r in records]),
    }
    print("[judge] SUMMARY " + json.dumps(summ))
    print(f"[judge] done {os.path.basename(args.jsonl)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
