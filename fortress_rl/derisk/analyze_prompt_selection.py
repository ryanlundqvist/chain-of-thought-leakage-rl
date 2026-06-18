#!/usr/bin/env python3
"""
De-risk Step 1 — FORTRESS prompt selection from precomputed VEA + refusal rates.

Reads Rohan's Olmo-3.1-32B-Think FORTRESS rollouts (50k = 500 prompts x ~100),
joins the LLM VEA judge and the LLM refusal judge per (prompt_id, completion_idx),
and computes per-prompt structure so we can pick the prompts that will give the
cleanest GRPO signal for Experiment 1.

NO judge calls, NO GPU — pure data joins over ~26 MB of judge files.

Two prompt slates:
  A "VEA-suppression" : refusal mostly high & constant, VEA has within-refusal
                        variance -> the leak penalty has something to bite on.
  B "EA-earns-keep"   : refusal has variance AND EA predicts refusal within the
                        prompt -> GRPO can reward latent EA organically.

Caveat: this is Olmo-3.1-32B-Think, NOT the Wood organism. Treat the selected
prompt ids as a starting hypothesis; the Wood pilot should re-rank.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import zstandard as zstd

ROOT = Path("/data/artifacts/rohan/santi/logs/eval_500x50")
VEA_FILES = [ROOT / f"20260505_161344_500x100_q{q}_kvfix/judge/qwen3_nemotron_genrm.jsonl.zst"
             for q in (1, 2, 3, 4)]
REFUSAL_FILE = ROOT / "combined_all_fortress/judge/qwen3_nemotron_genrm_refusal.jsonl.zst"
REGEX_FILE = ROOT / "combined_all_fortress/scores/3.1-Think.jsonl.zst"
OUT = Path(__file__).resolve().parent


def iter_jsonl_zst(path):
    with open(path, "rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as r:
            buf = b""
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        yield json.loads(line)


def load_label(paths, field="aware"):
    """Return {(pid, cidx): bool} from one or more judge files."""
    out = {}
    for p in paths:
        for r in iter_jsonl_zst(p):
            out[(r["prompt_id"], r["completion_idx"])] = bool(r[field])
    return out


def main():
    print("Loading VEA judge (4 files)...", flush=True)
    vea = load_label(VEA_FILES)
    print(f"  {len(vea):,} VEA-judged rollouts", flush=True)

    print("Loading refusal judge...", flush=True)
    refusal = load_label([REFUSAL_FILE])
    print(f"  {len(refusal):,} refusal-judged rollouts", flush=True)

    print("Loading regex VEA marker scores...", flush=True)
    regex = load_label([REGEX_FILE])
    print(f"  {len(regex):,} regex-scored rollouts", flush=True)

    # inner join: keep rollouts that have BOTH a VEA and a refusal label
    keys = set(vea) & set(refusal)
    print(f"\nInner join (VEA AND refusal labelled): {len(keys):,} rollouts", flush=True)

    # ---- pooled 2x2 ------------------------------------------------------
    cell = defaultdict(int)
    for k in keys:
        cell[(vea[k], refusal[k])] += 1
    a = cell[(True, True)]    # VEA & refuse
    b = cell[(True, False)]   # VEA & comply
    c = cell[(False, True)]   # no-VEA & refuse
    d = cell[(False, False)]  # no-VEA & comply
    n = a + b + c + d
    print("\n=== POOLED 2x2 (all 500 prompts) ===")
    print(f"  VEA & refuse   : {a:6,}  ({a/n:6.1%})")
    print(f"  VEA & comply   : {b:6,}  ({b/n:6.1%})")
    print(f"  no-VEA & refuse: {c:6,}  ({c/n:6.1%})")
    print(f"  no-VEA & comply: {d:6,}  ({d/n:6.1%})")
    print(f"  overall VEA rate     : {(a+b)/n:6.1%}")
    print(f"  overall refusal rate : {(a+c)/n:6.1%}")
    p_ref_vea = a / (a + b) if (a + b) else float("nan")
    p_ref_nov = c / (c + d) if (c + d) else float("nan")
    print(f"  P(refuse | VEA)      : {p_ref_vea:6.1%}")
    print(f"  P(refuse | no-VEA)   : {p_ref_nov:6.1%}")
    print(f"  coupling delta       : {p_ref_vea - p_ref_nov:+6.1%}")
    orh = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    print(f"  odds ratio (refuse~VEA, HA-corrected): {orh:5.2f}  (log {np.log(orh):+.2f})")
    # regex agreement vs LLM VEA judge
    rk = keys & set(regex)
    if rk:
        both = sum(1 for k in rk if regex[k] and vea[k])
        rx = sum(1 for k in rk if regex[k])
        lj = sum(1 for k in rk if vea[k])
        print(f"  regex VEA rate {rx/len(rk):.1%}  | LLM VEA rate {lj/len(rk):.1%}  "
              f"| regex hits that LLM also flags: {both/rx:.1%}" if rx else "")

    # ---- per-prompt ------------------------------------------------------
    pp = defaultdict(lambda: {"a": 0, "b": 0, "c": 0, "d": 0})
    for k in keys:
        pid = k[0]
        v, r = vea[k], refusal[k]
        pp[pid]["a" if (v and r) else "b" if (v and not r)
                else "c" if (not v and r) else "d"] += 1

    rows = []
    for pid, x in pp.items():
        a, b, c, d = x["a"], x["b"], x["c"], x["d"]
        tot = a + b + c + d
        n_ref = a + c
        n_vea = a + b
        pv_ref = a / n_ref if n_ref else float("nan")          # P(VEA | refuse)
        pr_vea = a / n_vea if n_vea else float("nan")          # P(refuse | VEA)
        pr_nov = c / (c + d) if (c + d) else float("nan")      # P(refuse | no-VEA)
        rows.append(dict(
            prompt_id=pid, n=tot,
            vea_rate=n_vea / tot, refusal_rate=n_ref / tot,
            n_vea_ref=a, n_vea_comp=b, n_novea_ref=c, n_novea_comp=d,
            n_cells_pop=sum(v > 0 for v in (a, b, c, d)),
            p_vea_given_refuse=pv_ref,
            coupling_delta=(pr_vea - pr_nov) if (n_vea and (c + d)) else float("nan"),
            # leak-signal proxy: # refusing rollouts x balance of VEA among them
            signal_A=(n_ref * pv_ref * (1 - pv_ref)) if n_ref else 0.0,
        ))
    df = pd.DataFrame(rows).sort_values("prompt_id").reset_index(drop=True)
    df.to_csv(OUT / "per_prompt_stats.csv", index=False)

    print(f"\n=== PER-PROMPT DISTRIBUTION ({len(df)} prompts) ===")
    for col in ("vea_rate", "refusal_rate"):
        s = df[col]
        print(f"  {col:14s}  mean {s.mean():.2f}  median {s.median():.2f}  "
              f"min {s.min():.2f}  max {s.max():.2f}")
    print(f"  prompts with all 4 cells populated: "
          f"{(df.n_cells_pop == 4).sum()} / {len(df)}")
    print(f"  prompt-level corr(vea_rate, refusal_rate): "
          f"{df.vea_rate.corr(df.refusal_rate):+.3f}")

    # ---- Slate A: VEA-suppression ---------------------------------------
    A = df[(df.refusal_rate >= 0.70) &
           (df.vea_rate.between(0.15, 0.85)) &
           (df.n_vea_ref >= 5) & (df.n_novea_ref >= 5)].copy()
    A = A.sort_values("signal_A", ascending=False)
    print(f"\n=== SLATE A  (VEA-suppression: high refusal, VEA variance) ===")
    print(f"  {len(A)} prompts qualify.  Top 15 by leak-signal proxy:")
    print(A[["prompt_id", "n", "vea_rate", "refusal_rate",
             "n_vea_ref", "n_novea_ref", "signal_A"]].head(15).to_string(index=False))

    # ---- Slate B: EA-earns-keep -----------------------------------------
    B = df[(df.refusal_rate.between(0.30, 0.80)) &
           (df.vea_rate >= 0.10) &
           (df.coupling_delta > 0.10) &
           (df.n_cells_pop == 4) &
           (df[["n_vea_ref", "n_vea_comp", "n_novea_ref", "n_novea_comp"]].min(axis=1) >= 3)].copy()
    B = B.sort_values("coupling_delta", ascending=False)
    print(f"\n=== SLATE B  (EA-earns-keep: refusal variance + EA->refusal coupling) ===")
    print(f"  {len(B)} prompts qualify.  Top 15 by coupling delta:")
    print(B[["prompt_id", "n", "vea_rate", "refusal_rate",
             "coupling_delta", "n_vea_ref", "n_vea_comp",
             "n_novea_ref", "n_novea_comp"]].head(15).to_string(index=False))

    # ---- highest-VEA prompts (elicitation) ------------------------------
    print(f"\n=== HIGHEST baseline VEA-rate prompts (best EA elicitation) ===")
    print(df.nlargest(15, "vea_rate")[
        ["prompt_id", "n", "vea_rate", "refusal_rate"]].to_string(index=False))

    sel = {
        "slate_A_vea_suppression": A.prompt_id.tolist(),
        "slate_B_ea_earns_keep": B.prompt_id.tolist(),
        "union": sorted(set(A.prompt_id) | set(B.prompt_id)),
        "criteria": {
            "A": "refusal_rate>=0.70, 0.15<=vea_rate<=0.85, n_vea_ref>=5, n_novea_ref>=5",
            "B": "0.30<=refusal_rate<=0.80, vea_rate>=0.10, coupling_delta>0.10, all 4 cells>=3",
        },
    }
    (OUT / "selected_prompts.json").write_text(json.dumps(sel, indent=2))
    print(f"\nWrote per_prompt_stats.csv  and  selected_prompts.json  to {OUT}")
    print(f"  slate A: {len(A)}   slate B: {len(B)}   union: {len(sel['union'])}")


if __name__ == "__main__":
    sys.exit(main())
