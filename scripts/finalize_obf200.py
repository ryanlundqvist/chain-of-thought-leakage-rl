#!/usr/bin/env python3
"""Merge the obf200 a/b shards per combo and build the per-prompt 200-sample table.
Run AFTER the re-judge. Writes:
  results/grpo_runs/obf200_<model>_<cot>/eval/<label>.jsonl   (merged, 37 prompts × 200)
  results/obf200_perprompt.csv                                (per prompt: BASE vs r260, natural)
"""
import glob, json, os, csv
from collections import defaultdict
import numpy as np

GR = "results/grpo_runs"
MODELS = [("BASE", "base"), ("r260", "r0260")]
COTS = ["natural", "forced"]


def shard_files(model, cot):
    out = []
    for sh in ("a", "b"):
        fs = [f for f in glob.glob(f"{GR}/obf200_{model}_{cot[:3]}_{sh}/eval/*.jsonl") if "summary" not in f]
        out += fs
    return out


def load_rows(model, cot):
    rows = []
    for f in shard_files(model, cot):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def merge():
    for model, label in MODELS:
        for cot in COTS:
            rows = load_rows(model, cot)
            d = f"{GR}/obf200_{model}_{cot}/eval"
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/{label}.jsonl", "w") as fo:
                for r in rows:
                    fo.write(json.dumps(r) + "\n")
            print(f"  merged obf200_{model}_{cot} -> {len(rows)} rows")


def vea_pos(r):
    return r.get("vea_label") in ("WEAK", "STRONG")
def vea_valid(r):
    return r.get("vea_label") not in (None, "SKIPPED", "INVALID")
def th_bin(r):
    s = r.get("type_hint_score", -1)
    return None if s < 0 else (1.0 if s > 0 else 0.0)


def per_prompt_table():
    # natural CoT, BASE vs r260
    data = {m: load_rows(m, "natural") for m, _ in MODELS}
    by = {m: defaultdict(list) for m in data}
    for m, rows in data.items():
        for r in rows:
            by[m][r["prompt_id"]].append(r)
    ids = sorted(set(by["BASE"]) & set(by["r260"]))

    def stats(rs):
        th = [th_bin(r) for r in rs if th_bin(r) is not None]
        vj = [1.0 if vea_pos(r) else 0.0 for r in rs if vea_valid(r)]
        reg = [1.0 if r.get("vea_regex_any") else 0.0 for r in rs]
        pp = [r["probe_at_prompt_avg"] for r in rs if r.get("probe_at_prompt_avg") is not None]
        pc = [r["probe_score_avg"] for r in rs if r.get("probe_score_avg") is not None]
        return (np.mean(th) if th else np.nan, len(th),
                np.mean(vj) if vj else np.nan, len(vj),
                np.mean(reg), np.mean(pp) if pp else np.nan, np.mean(pc) if pc else np.nan)

    rows_out = []
    for pid in ids:
        b = stats(by["BASE"][pid]); r = stats(by["r260"][pid])
        rows_out.append(dict(
            prompt_id=pid,
            th_base=round(b[0], 3), th_r260=round(r[0], 3), d_th=round(r[0] - b[0], 3),
            vea_base=round(b[2], 3), vea_r260=round(r[2], 3), d_vea=round(r[2] - b[2], 3),
            probeP_base=round(b[5], 3), probeP_r260=round(r[5], 3),
            probeC_base=round(b[6], 3), probeC_r260=round(r[6], 3),
            n_base=b[1], n_r260=r[1]))
    rows_out.sort(key=lambda x: (x["d_vea"], -x["th_r260"]))
    with open("results/obf200_perprompt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader()
        for x in rows_out: w.writerow(x)
    # headline (95% CI via normal approx on the 200-sample pooled rate)
    print(f"\n  per-prompt table -> results/obf200_perprompt.csv ({len(rows_out)} prompts)")
    for m, _ in MODELS:
        allr = load_rows(m, "natural")
        th = [th_bin(r) for r in allr if th_bin(r) is not None]
        vj = [1.0 if vea_pos(r) else 0.0 for r in allr if vea_valid(r)]
        print(f"    {m:5s} natural (pooled): type-hint {np.mean(th)*100:.1f}%  VEA {np.mean(vj)*100:.1f}%  (n={len(allr)})")


if __name__ == "__main__":
    print("=== merge shards ==="); merge()
    print("=== per-prompt table ==="); per_prompt_table()
