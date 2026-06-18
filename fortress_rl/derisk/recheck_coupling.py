#!/usr/bin/env python3
"""Re-audit the Q3 EA<->refusal coupling from the FORTRESS pilot.

The pilot scored coupling over ALL 188 Wood-header rollouts. But 53% never
closed </think> at the 6000-tok cap -> split_think returns cot="" for them ->
they CANNOT be labeled vea_in_cot, and their 'refusal' is regex-matched against
the raw unfinished CoT instead of a real response. This re-checks the coupling
on a like-for-like basis (closed-CoT rollouts only).
"""
import sys, json, math
sys.path.insert(0, "scripts")
from score_constitutional import VEA_RE, REFUSAL_RE, split_think

rows = [json.loads(l) for l in
        open("results/pilot_fortress/rollouts_wood.jsonl") if l.strip()]
D = []
for r in rows:
    c = r.get("completion")
    if not c:
        continue
    cot, resp, _ = split_think(c)
    D.append(dict(
        closed=bool(cot),
        vea_in_cot=bool(cot) and bool(VEA_RE.search(cot)),
        vea_anywhere=bool(VEA_RE.search(c)),
        refusal=bool(REFUSAL_RE.search(resp)),
        cot_len=len(cot),
    ))

def rate(items, k):
    return sum(x[k] for x in items) / len(items) if items else float("nan")

def delta_ci(a, b):
    """95% CI half-width for difference of two proportions."""
    pa, pb = rate(a, "refusal"), rate(b, "refusal")
    se = math.sqrt(pa*(1-pa)/len(a) + pb*(1-pb)/len(b))
    return pa - pb, 1.96*se

closed = [x for x in D if x["closed"]]
trunc  = [x for x in D if not x["closed"]]

print(f"total OK rollouts        : {len(D)}")
print(f"  closed </think>        : {len(closed)}  ({len(closed)/len(D):.0%})")
print(f"  truncated (no </think>): {len(trunc)}  ({len(trunc)/len(D):.0%})")
print(f"  truncated yet VEA-anywhere: "
      f"{sum(x['vea_anywhere'] for x in trunc)} "
      f"-> these are EA rollouts the pilot's vea_in_cot MISSED\n")

# --- reproduce the pilot's number (all rollouts) ---
v  = [x for x in D if x["vea_in_cot"]]
nv = [x for x in D if not x["vea_in_cot"]]
d, ci = delta_ci(v, nv)
print("PILOT METHOD (all 188, vea_in_cot vs not):")
print(f"  refusal|VEA={rate(v,'refusal'):.3f} (n={len(v)})  "
      f"refusal|noVEA={rate(nv,'refusal'):.3f} (n={len(nv)})")
print(f"  coupling delta = {d:+.3f}  95% CI [{d-ci:+.3f}, {d+ci:+.3f}]")
print(f"  -> noVEA bucket is {sum(not x['closed'] for x in nv)}/{len(nv)} "
      f"TRUNCATED rollouts (refusal scored on raw CoT)\n")

# --- like-for-like: closed-CoT rollouts only ---
v  = [x for x in closed if x["vea_in_cot"]]
nv = [x for x in closed if not x["vea_in_cot"]]
print(f"LIKE-FOR-LIKE (closed-CoT only, n={len(closed)}):")
if v and nv:
    d, ci = delta_ci(v, nv)
    print(f"  refusal|VEA={rate(v,'refusal'):.3f} (n={len(v)})  "
          f"refusal|noVEA={rate(nv,'refusal'):.3f} (n={len(nv)})")
    print(f"  coupling delta = {d:+.3f}  95% CI [{d-ci:+.3f}, {d+ci:+.3f}]")
print(f"  refusal rate among closed   : {rate(closed,'refusal'):.3f}")
print(f"  refusal rate among truncated: {rate(trunc,'refusal'):.3f}  "
      f"(scored on raw unfinished CoT)")
print(f"  mean CoT len  VEA={sum(x['cot_len'] for x in v)/max(len(v),1):.0f}  "
      f"noVEA={sum(x['cot_len'] for x in nv)/max(len(nv),1):.0f} chars")
