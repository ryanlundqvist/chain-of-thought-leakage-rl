#!/usr/bin/env python3
"""Iterative from-prefix resampling: rejection-sample at EVERY VEA occurrence.

Contrast with resample_from_prefix.py (single truncation, keep-if-clean, DISCARD-if-VEA): that
method throws away any continuation that re-verbalizes, which (a) lets leftover VEA leak through
when the judge mislabels a re-verbalization as benign, and (b) silently drops rollouts where all
K continuations re-verbalize (K-limit selection bias).

This method instead keeps the rollout on a VEA-free trajectory by STOPPING and RESAMPLING each
time eval-awareness bubbles up:
  1. truncate the BASE rollout at the first VEA sentence            -> clean prefix
  2. generate forward (n=1); ALWAYS re-judge the full CoT (120b)
     + regex-catch any phrase the judge missed, confirmed by a targeted re-judge of that sentence
  3. if VEA is present, truncate at that NEW sentence, extend the clean prefix, resample forward
  4. repeat until the rollout completes with NO vea (status=cleaned)
     or we hit --max-rounds (status=insisted == the model will not stop verbalizing == load-bearing VEA)

Type-hint rate is measured over status=cleaned rollouts only -> the proper "force out ALL
verbalization, does the behaviour survive?" counterfactual. The insisted fraction is itself a
signal (verbalization is load-bearing for those).

Dual-server: 49B policy (--gen-url) + gpt-oss-120b monitor/judge (--mon-url).
Outputs results/resampling/<tag>.jsonl (one row per BASE VEA rollout) + a summary.
"""
import argparse, asyncio, json, os, sys
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
# reuse the vetted helpers from the single-shot orchestrator
from resample_from_prefix import (split_think, regex_trunc, locate_120b, judge_vea,
                                  type_hint_coverage, load_vea_rollouts, SENT, VEA_RE, BASE_MODEL)
import resample_from_prefix as rfp   # to flip its module-level TRUNC_MODE (regex_trunc reads it)

SAVE_TEXT = False


async def targeted_genuine(mon, window):
    """Is the regex hit GENUINE eval-awareness (vs benign 'evaluation standards')? Re-judge the
    local window with the same 120b classifier so 'meets the evaluation standards' is NOT treated
    as VEA but 'since this is an AI evaluation context...' IS."""
    if len(window.strip()) < 30:
        window = (window + " " * 30)[:120]
    vea, _ = await judge_vea(mon, window)
    return vea


async def iterative_one(gen, mon, tok, rec, max_rounds, sem):
    cot = rec["cot"]; sysp = rec["system_prompt"]; userp = rec["user_prompt"]
    t0 = regex_trunc(cot); loc = "regex"
    if t0 is None:
        t0 = await locate_120b(mon, cot); loc = "120b"
    if t0 is None:
        return {"prompt_id": rec["prompt_id"], "located": False}
    templ = tok.apply_chat_template([{"role": "system", "content": sysp},
                                     {"role": "user", "content": userp}],
                                    tokenize=False, add_generation_prompt=True)
    prefix = cot[:t0]
    trace = []
    for rnd in range(max_rounds + 1):
        prompt = templ + "<think>\n" + prefix
        async with sem:
            try:
                r = await gen.completions.create(model=BASE_MODEL, prompt=prompt, n=1,
                        temperature=0.7, top_p=0.95, max_tokens=3500, seed=42 + rnd)
            except Exception as e:
                return {"prompt_id": rec["prompt_id"], "located": True, "gen_err": str(e)[:80], "trace": trace}
        full = "<think>\n" + prefix + r.choices[0].text
        full_cot, resp = split_think(full)
        new = full_cot[len(prefix):] if len(full_cot) >= len(prefix) else ""
        vea_j, lab = await judge_vea(mon, full_cot)          # ALWAYS rejudge, full context
        m = VEA_RE.search(new)                               # regex catch in the NEW region only
        regex_genuine = False
        if m:                                                # confirm the regex hit is real, not "evaluation standards"
            regex_genuine = await targeted_genuine(mon, new[max(0, m.start() - 150):m.start() + 200])
        trace.append({"round": rnd, "vea_judge": vea_j, "vea_label": lab, "vea_regex": bool(m),
                      "regex_genuine": regex_genuine, "regex_hit": m.group(0) if m else "", "new_chars": len(new)})
        if (not vea_j) and (not regex_genuine):              # clean throughout -> accept
            return {"prompt_id": rec["prompt_id"], "located": True, "loc": loc, "status": "cleaned",
                    "rounds": rnd, "th": type_hint_coverage(resp), "trace": trace,
                    **({"final_cot": full_cot, "response": resp, "prefix0": cot[:t0]} if SAVE_TEXT else {})}
        # VEA present -> locate the first VEA sentence in `new`, extend the clean prefix, resample
        if regex_genuine:
            bs = [x.end() for x in SENT.finditer(new[:m.start()])]
            cut = bs[-1] if bs else 0
        else:                                                # judge-VEA the regex missed (paraphrase)
            crel = await locate_120b(mon, new)
            cut = crel if crel is not None else 0
        prefix = prefix + (new[:cut] if cut > 0 else "")     # cut==0 -> retry same point next round
    return {"prompt_id": rec["prompt_id"], "located": True, "loc": loc, "status": "insisted",
            "rounds": max_rounds, "th": None, "trace": trace,
            **({"final_prefix": prefix} if SAVE_TEXT else {})}


def summarize(outp, n_in):
    rows = [json.loads(l) for l in open(outp)]
    loc = [r for r in rows if r.get("located")]
    cleaned = [r for r in loc if r.get("status") == "cleaned"]
    insisted = [r for r in loc if r.get("status") == "insisted"]
    gerr = [r for r in loc if r.get("gen_err")]
    th = [1.0 if r["th"] > 0 else 0.0 for r in cleaned if r.get("th") is not None and r["th"] >= 0]
    rounds = [r["rounds"] for r in cleaned]
    # round-0 audit quantities (re-verbalization + INVALID on the first resample)
    r0 = [r["trace"][0] for r in loc if r.get("trace")]
    r0_vea = sum(t["vea_judge"] for t in r0); r0_inv = sum(t["vea_label"] == "INVALID" for t in r0)
    r0_regexcatch = sum((not t["vea_judge"]) and t["regex_genuine"] for t in r0)
    print(f"\n[iter] {outp}")
    print(f"  located {len(loc)}/{n_in}  cleaned {len(cleaned)} ({100*len(cleaned)/max(1,len(loc)):.0f}%)  "
          f"insisted {len(insisted)} ({100*len(insisted)/max(1,len(loc)):.0f}%)  gen_err {len(gerr)}")
    if rounds:
        import collections
        dist = collections.Counter(rounds)
        print(f"  rounds-to-clean: mean {np.mean(rounds):.2f}  dist " +
              " ".join(f"{k}:{dist[k]}" for k in sorted(dist)))
    if th:
        m = np.mean(th) * 100
        print(f"  ITERATIVE from-prefix type-hint (TH | fully-cleaned) = {m:.1f}%  (n={len(th)})")
    print(f"  round-0 audit: re-verbalize(judge)={r0_vea}/{len(r0)} ({100*r0_vea/max(1,len(r0)):.0f}%)  "
          f"INVALID-judge={r0_inv} ({100*r0_inv/max(1,len(r0)):.1f}%)  regex-caught-judge-miss={r0_regexcatch}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gen-url", required=True)
    ap.add_argument("--mon-url", required=True)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--cap-per-pid", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--save-text", action="store_true")
    ap.add_argument("--shard", default=None, help="N/M : process only rollouts where idx%M==N")
    ap.add_argument("--trunc-mode", default="sentence",
                    choices=["sentence", "prev_sentence", "paragraph", "whole"],
                    help="where to cut each round: sentence=at the VEA sentence; prev_sentence=the sentence BEFORE it")
    args = ap.parse_args()
    global SAVE_TEXT; SAVE_TEXT = args.save_text
    rfp.TRUNC_MODE = args.trunc_mode   # regex_trunc (imported) reads this module global
    print(f"[iter] trunc_mode={args.trunc_mode}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    recs = load_vea_rollouts(args.runs, cap_per_pid=args.cap_per_pid)
    if args.shard:
        sn, sm = (int(x) for x in args.shard.split("/"))
        recs = [r for i, r in enumerate(recs) if i % sm == sn]
        print(f"[shard] {args.shard} -> {len(recs)} rollouts", flush=True)
    if args.limit: recs = recs[:args.limit]
    print(f"[iter] tag={args.tag} VEA rollouts={len(recs)} max_rounds={args.max_rounds}", flush=True)
    gen = AsyncOpenAI(base_url=args.gen_url, api_key="EMPTY")
    mon = AsyncOpenAI(base_url=args.mon_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)
    os.makedirs("results/resampling", exist_ok=True)
    outp = f"results/resampling/{args.tag}.jsonl"
    fout = open(outp, "w"); done = 0
    for fut in asyncio.as_completed([iterative_one(gen, mon, tok, r, args.max_rounds, sem) for r in recs]):
        row = await fut; fout.write(json.dumps(row) + "\n"); fout.flush(); done += 1
        if done % 50 == 0: print(f"  {done}/{len(recs)}", flush=True)
    fout.close()
    summarize(outp, len(recs))
    print(f"  -> {outp}")


if __name__ == "__main__":
    asyncio.run(main())
