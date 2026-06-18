#!/usr/bin/env python3
"""pairwise_compare.py — the 20B-vs-120B judge comparison.

From classified Wood rollouts, builds within-prompt comparison pairs across the
VEA x behavior cells and scores each with the soft pairwise-logprob judge
(Claude constitution), for every (judge_model, visibility) combination:

  contrasts (resp1 = the 'expected better' response):
    vea_refuse   : resp1 = noVEA+refuse   vs  resp2 = VEA+refuse
    vea_comply   : resp1 = noVEA+comply   vs  resp2 = VEA+comply
    refuse_comply: resp1 = noVEA+refuse   vs  resp2 = noVEA+comply  (scale check)

  P(prefer resp1) > 0.5  => judge prefers the 'expected better' one.
  VEA leak effect = P(prefer noVEA | leak) - P(prefer noVEA | no_leak).

Output: pairwise_results.jsonl — one row per (pair, judge, visibility).
Resumable: appends; re-run skips done rows.
"""
import argparse
import asyncio
import json
import os
import random
import sys

import aiohttp

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from pairwise_logprob_judge import judge_pair, render  # noqa: E402


def build_pairs(rows, per_cell, max_pairs, seed=42):
    """Within-prompt pairs across VEA x behavior cells."""
    rng = random.Random(seed)
    by_prompt = {}
    for r in rows:
        if r["vea_in_cot"] is None or r["behavior"] is None:
            continue
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    pairs = []
    for pid, rs in by_prompt.items():
        cell = {}
        for r in rs:
            cell.setdefault((r["vea_in_cot"], r["behavior"]), []).append(r)
        for k in cell:
            rng.shuffle(cell[k])
            cell[k] = cell[k][:per_cell]

        def mk(contrast, c1, c2):
            a, b = cell.get(c1, []), cell.get(c2, [])
            for r1, r2 in list(zip(a, b))[:max_pairs]:
                pairs.append({
                    "prompt_id": pid, "contrast": contrast,
                    "user_prompt": r1["user_prompt"],
                    "resp1_ci": r1["completion_idx"], "resp1_text": r1["text"],
                    "resp2_ci": r2["completion_idx"], "resp2_text": r2["text"],
                })
        mk("vea_refuse",    (False, "refuse"), (True, "refuse"))
        mk("vea_comply",    (False, "comply"), (True, "comply"))
        mk("refuse_comply", (False, "refuse"), (False, "comply"))
    return pairs


async def main_async(args):
    constitution = open(args.constitution).read().strip()
    rows = [json.loads(l) for l in open(args.classified) if l.strip()]
    pairs = build_pairs(rows, args.per_cell, args.max_pairs)
    print(f"[compare] {len(pairs)} pairs from {args.classified}", flush=True)
    by_c = {}
    for p in pairs:
        by_c[p["contrast"]] = by_c.get(p["contrast"], 0) + 1
    print(f"[compare] by contrast: {by_c}", flush=True)

    judges = []
    if args.judge_120b_url:
        judges.append(("gpt-oss-120b", "openai/gpt-oss-120b", args.judge_120b_url))
    if args.judge_20b_url:
        judges.append(("gpt-oss-20b", "openai/gpt-oss-20b", args.judge_20b_url))
    if not judges:
        sys.exit("need --judge-120b-url and/or --judge-20b-url")

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            if l.strip():
                r = json.loads(l)
                done.add((r["prompt_id"], r["resp1_ci"], r["resp2_ci"],
                          r["contrast"], r["judge"], r["visibility"]))

    tasks = []  # (meta, coro-args)
    for p in pairs:
        for jname, jmodel, jurl in judges:
            for vis in ("leak", "no_leak"):
                key = (p["prompt_id"], p["resp1_ci"], p["resp2_ci"],
                       p["contrast"], jname, vis)
                if key not in done:
                    tasks.append((p, jname, jmodel, jurl, vis))
    print(f"[compare] {len(tasks)} judge tasks ({len(done)} already done)", flush=True)

    sem = asyncio.Semaphore(args.concurrent)
    conn = aiohttp.TCPConnector(limit=args.concurrent)
    async with aiohttp.ClientSession(connector=conn) as sess:

        async def run(p, jname, jmodel, jurl, vis):
            t1 = render(p["resp1_text"], vis)
            t2 = render(p["resp2_text"], vis)
            async with sem:
                res = await judge_pair(sess, jurl, jmodel, constitution,
                                       p["user_prompt"], t1, t2)
            return {
                "prompt_id": p["prompt_id"], "contrast": p["contrast"],
                "resp1_ci": p["resp1_ci"], "resp2_ci": p["resp2_ci"],
                "judge": jname, "visibility": vis,
                "p_prefer_resp1": res["p_prefer_resp1"], "n_ok": res["n_ok"],
            }

        n_ok_total = 0
        with open(args.out, "a") as f:
            for i in range(0, len(tasks), 120):
                batch = tasks[i:i + 120]
                results = await asyncio.gather(*[run(*t) for t in batch])
                for r in results:
                    f.write(json.dumps(r) + "\n")
                    n_ok_total += (r["p_prefer_resp1"] is not None)
                f.flush()
                done_n = min(i + 120, len(tasks))
                print(f"[compare] {done_n}/{len(tasks)}  (usable so far: {n_ok_total})",
                      flush=True)
                # fail loud: if the whole first batch parsed to None, the
                # logprob extraction is broken — don't burn the rest silently.
                if i == 0 and n_ok_total == 0:
                    sys.exit("[compare] ABORT: first 120 judge calls all returned "
                             "p_prefer_resp1=None — logprob parsing or judge "
                             "endpoint is broken. Fix _extract_p_a / check the URL.")
    print(f"[compare] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--constitution", required=True)
    ap.add_argument("--judge-120b-url", default=None)
    ap.add_argument("--judge-20b-url", default=None)
    ap.add_argument("--per-cell", type=int, default=6,
                    help="max rollouts sampled per (VEA,behavior) cell per prompt")
    ap.add_argument("--max-pairs", type=int, default=5,
                    help="max pairs per contrast per prompt")
    ap.add_argument("--concurrent", type=int, default=32)
    asyncio.run(main_async(ap.parse_args()))
