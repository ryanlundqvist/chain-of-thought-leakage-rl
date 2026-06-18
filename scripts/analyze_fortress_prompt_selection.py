#!/usr/bin/env python3
"""
De-risk analysis for the FORTRESS CoT-leak GRPO experiment.

Uses ONLY precomputed judge labels (no GPU, no judge calls):
  - VEA / eval-awareness judge  (qwen3_nemotron_genrm)        -> is_vea
  - Refusal judge               (qwen3_nemotron_genrm_refusal) -> is_refusal
  - Regex VEA-marker scores      (cheap noisy cross-check)      -> regex_vea

Goal: from the 500 FORTRESS prompts pick the "sweet spot" set for GRPO --
prompts where, within the ~100 sampled rollouts, BOTH refusal and VEA vary,
all three of Ryan's target buckets coexist, and VEA actually predicts refusal
(so EA can "earn its keep" under a constitutional reward).
"""
import json
import zstandard as zstd
from pathlib import Path
from collections import defaultdict

ROOT = Path("/data/artifacts/rohan/santi/logs/eval_500x50")
OUT = Path(__file__).resolve().parent.parent / "results" / "fortress_prompt_selection"
OUT.mkdir(parents=True, exist_ok=True)


def iter_jsonl_zst(path):
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        yield json.loads(line)


print("loading VEA judge (4 quarters)...")
is_vea = {}
for q in (1, 2, 3, 4):
    p = ROOT / f"20260505_161344_500x100_q{q}_kvfix/judge/qwen3_nemotron_genrm.jsonl.zst"
    for r in iter_jsonl_zst(p):
        is_vea[(r["prompt_id"], r["completion_idx"])] = bool(r["aware"])
print(f"  VEA labels: {len(is_vea)}")

print("loading refusal judge...")
is_ref = {}
for r in iter_jsonl_zst(ROOT / "combined_all_fortress/judge/qwen3_nemotron_genrm_refusal.jsonl.zst"):
    is_ref[(r["prompt_id"], r["completion_idx"])] = bool(r["aware"])
print(f"  refusal labels: {len(is_ref)}")

print("loading regex VEA-marker scores...")
regex_vea = {}
for r in iter_jsonl_zst(ROOT / "combined_all_fortress/scores/3.1-Think.jsonl.zst"):
    regex_vea[(r["prompt_id"], r["completion_idx"])] = bool(r["aware"])
print(f"  regex labels: {len(regex_vea)}")

# inner join on keys with BOTH vea and refusal labels
keys = set(is_vea) & set(is_ref)
print(f"\nrollouts with both VEA + refusal labels: {len(keys)}")

# per-prompt bucket tallies
# buckets: (refuse, vea) -> count
P = defaultdict(lambda: {"n": 0, "vea": 0, "ref": 0,
                         "rr_ee": 0,  # refuse + EA
                         "rr_ne": 0,  # refuse + no-EA
                         "cc_ee": 0,  # comply + EA
                         "cc_ne": 0,  # comply + no-EA
                         "regex_vea": 0, "n_regex": 0})
for k in keys:
    pid = k[0]
    v, r = is_vea[k], is_ref[k]
    d = P[pid]
    d["n"] += 1
    d["vea"] += v
    d["ref"] += r
    if r and v:
        d["rr_ee"] += 1
    elif r and not v:
        d["rr_ne"] += 1
    elif (not r) and v:
        d["cc_ee"] += 1
    else:
        d["cc_ne"] += 1
    if k in regex_vea:
        d["n_regex"] += 1
        d["regex_vea"] += regex_vea[k]

# ---- global aggregates ----
G = {"n": 0, "vea": 0, "ref": 0, "rr_ee": 0, "rr_ne": 0, "cc_ee": 0, "cc_ne": 0}
for d in P.values():
    for kk in G:
        G[kk] += d[kk]
n = G["n"]

lines = []
def emit(s=""):
    print(s)
    lines.append(s)

emit("=" * 70)
emit("GLOBAL  (joined VEA x refusal, n=%d rollouts, %d prompts)" % (n, len(P)))
emit("=" * 70)
emit(f"  VEA rate (LLM judge) : {G['vea']/n:.3f}")
emit(f"  refusal rate         : {G['ref']/n:.3f}")
emit("")
emit("  joint breakdown:")
emit(f"    refuse + EA   : {G['rr_ee']/n:.3f}  ({G['rr_ee']})")
emit(f"    refuse + noEA : {G['rr_ne']/n:.3f}  ({G['rr_ne']})")
emit(f"    comply + EA   : {G['cc_ee']/n:.3f}  ({G['cc_ee']})   <- expected rare")
emit(f"    comply + noEA : {G['cc_ne']/n:.3f}  ({G['cc_ne']})   <- 'properly fooled'")
emit("")
# P(refuse | EA) vs P(refuse | noEA)  -- does EA predict refusal?
n_ea = G["rr_ee"] + G["cc_ee"]
n_ne = G["rr_ne"] + G["cc_ne"]
p_ref_ea = G["rr_ee"] / n_ea if n_ea else float("nan")
p_ref_ne = G["rr_ne"] / n_ne if n_ne else float("nan")
emit(f"  P(refuse | EA)   = {p_ref_ea:.3f}   (n={n_ea})")
emit(f"  P(refuse | noEA) = {p_ref_ne:.3f}   (n={n_ne})")
emit(f"  --> EA->refusal lift = {p_ref_ea - p_ref_ne:+.3f}")
emit(f"  P(EA | refuse)   = {G['rr_ee']/G['ref']:.3f}")
emit(f"  P(EA | comply)   = {(G['cc_ee']/(n-G['ref'])):.3f}")

# ---- sweet-spot prompt scoring ----
# A prompt is good for GRPO if, within its ~100 rollouts:
#   * refusal rate is intermediate            -> refusal varies in the group
#   * VEA rate is non-trivial and not saturated -> VEA varies in the group
#   * all three of Ryan's buckets are populated (>=3 each):
#       comply+noEA, refuse+EA, refuse+noEA
#   * EA->refusal lift is >= 0 within the prompt (EA earns its keep)
def score_prompt(d):
    if d["n"] < 50:
        return None
    rr = d["ref"] / d["n"]
    vr = d["vea"] / d["n"]
    n_ea = d["rr_ee"] + d["cc_ee"]
    n_ne = d["rr_ne"] + d["cc_ne"]
    lift = (d["rr_ee"] / n_ea if n_ea else 0.0) - (d["rr_ne"] / n_ne if n_ne else 0.0)
    # variance terms (Bernoulli p(1-p), peaks at 0.5)
    ref_var = rr * (1 - rr)
    vea_var = vr * (1 - vr)
    buckets_ok = (d["cc_ne"] >= 3 and d["rr_ee"] >= 3 and d["rr_ne"] >= 3)
    composite = ref_var * vea_var * (1.0 if buckets_ok else 0.15) * (1.0 + max(lift, 0))
    return dict(pid=None, n=d["n"], rr=rr, vr=vr, lift=lift,
                ref_var=ref_var, vea_var=vea_var, buckets_ok=buckets_ok,
                composite=composite,
                cc_ne=d["cc_ne"], cc_ee=d["cc_ee"],
                rr_ee=d["rr_ee"], rr_ne=d["rr_ne"],
                regex_vr=(d["regex_vea"] / d["n_regex"] if d["n_regex"] else float("nan")))

scored = []
for pid, d in P.items():
    s = score_prompt(d)
    if s:
        s["pid"] = pid
        scored.append(s)
scored.sort(key=lambda s: s["composite"], reverse=True)

emit("")
emit("=" * 70)
emit("TOP 40 SWEET-SPOT PROMPTS for GRPO")
emit("  rr=refusal rate  vr=VEA rate  lift=P(ref|EA)-P(ref|noEA)")
emit("  buckets shown as counts: [comply+noEA / refuse+EA / refuse+noEA / comply+EA]")
emit("=" * 70)
emit(f"{'pid':>5} {'n':>4} {'rr':>6} {'vr':>6} {'lift':>7} {'regexVr':>8}  buckets[cNE/rEE/rNE/cEE]  ok")
for s in scored[:40]:
    emit(f"{s['pid']:>5} {s['n']:>4} {s['rr']:>6.2f} {s['vr']:>6.2f} {s['lift']:>+7.2f} "
         f"{s['regex_vr']:>8.2f}  [{s['cc_ne']:>3}/{s['rr_ee']:>3}/{s['rr_ne']:>3}/{s['cc_ee']:>3}]   "
         f"{'Y' if s['buckets_ok'] else '.'}")

# how many prompts clear the 'all 3 buckets populated' bar
n_ok = sum(1 for s in scored if s["buckets_ok"])
n_mid = sum(1 for s in scored if 0.3 <= s["rr"] <= 0.7)
emit("")
emit(f"prompts with all 3 buckets populated (>=3 each): {n_ok} / {len(scored)}")
emit(f"prompts with intermediate refusal rate [0.3,0.7]: {n_mid} / {len(scored)}")

# dump full per-prompt table as csv + the top selection as json
import csv
with open(OUT / "per_prompt_rates.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["prompt_id", "n", "refusal_rate", "vea_rate", "regex_vea_rate",
                "ea_refusal_lift", "comply_noEA", "refuse_EA", "refuse_noEA",
                "comply_EA", "buckets_ok", "composite"])
    for s in sorted(scored, key=lambda s: s["pid"]):
        w.writerow([s["pid"], s["n"], f"{s['rr']:.4f}", f"{s['vr']:.4f}",
                    f"{s['regex_vr']:.4f}", f"{s['lift']:.4f}",
                    s["cc_ne"], s["rr_ee"], s["rr_ne"], s["cc_ee"],
                    int(s["buckets_ok"]), f"{s['composite']:.5f}"])

top = [s["pid"] for s in scored[:40]]
with open(OUT / "selected_prompt_ids.json", "w") as f:
    json.dump({"top40": top,
               "all_buckets_ok": [s["pid"] for s in scored if s["buckets_ok"]]},
              f, indent=2)

with open(OUT / "analysis_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

emit("")
emit(f"wrote: {OUT}/per_prompt_rates.csv")
emit(f"wrote: {OUT}/selected_prompt_ids.json")
emit(f"wrote: {OUT}/analysis_summary.txt")
