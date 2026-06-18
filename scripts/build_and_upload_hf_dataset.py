#!/usr/bin/env python3
"""Assemble the DERIVED-RESULTS dataset (aggregate metric tables + plots + findings card) and
upload to a PRIVATE HuggingFace dataset repo. No model weights, no raw rollouts."""
import json, glob, os, csv, shutil
import numpy as np
PROJ = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage")
G = f"{PROJ}/results/grpo_runs"
STAGE = "/tmp/hf_obf_dataset"
REPO = "rlundqvist/chain-of-thought-leakage-rl-results"
np.random.seed(0)
if os.path.exists(STAGE): shutil.rmtree(STAGE)
os.makedirs(f"{STAGE}/plots", exist_ok=True)


def load(run):
    fs = glob.glob(f"{G}/{run}/eval/*.jsonl")
    return [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()] if fs else None


def rate(rs, pred):
    v = [pred(r) for r in rs if pred(r) is not None]
    return round(np.mean(v) * 100, 2) if v else ""


# ---- table 1: metrics_by_checkpoint ----
def th(r):
    s = r.get("type_hint_score", -1); return None if s < 0 else (1.0 if s > 0 else 0.0)
def vreg(r): return 1.0 if r.get("vea_regex_any") else 0.0
def vjud(r):
    return None if r.get("vea_label") in (None, "SKIPPED") else (1.0 if r.get("vea_label") in ("WEAK", "STRONG") else 0.0)

rows1 = []
for m in ["BASE", "r35", "r219", "r260"]:
    for pf in ["pf", "np"]:
        for cot in ["natural", "forced", "none"]:
            s = load(f"s20_{m}_{pf}_{cot}")
            if not s: continue
            vj = load(f"vj_{m}_{pf}_{cot}")
            cc = np.array([r.get("cot_chars", 0) for r in s], float)
            rows1.append(dict(checkpoint=m, prefix=pf, cot_mode=cot, n=len(s),
                              type_hint_pct=rate(s, th), vea_regex_pct=rate(s, vreg),
                              vea_judge_pct=(rate(vj, vjud) if vj else ""),
                              cot_chars_mean=round(cc.mean(), 0)))
# probe from scaled_* (pf natural)
for m, run in [("BASE", "scaled_BASE"), ("r35", "scaled_r35"), ("r219", "scaled_r219"), ("r260", "scaled_r260")]:
    rs = load(run)
    if not rs: continue
    f2n = lambda x: float(x) if x is not None else np.nan
    pp = np.nanmean(np.array([f2n(r.get("probe_at_prompt_avg")) for r in rs], float))
    pc = np.nanmean(np.array([f2n(r.get("probe_score_avg")) for r in rs], float))
    for row in rows1:
        if row["checkpoint"] == m and row["prefix"] == "pf" and row["cot_mode"] == "natural":
            row["probe_at_prompt"] = round(float(pp), 4); row["probe_at_cot"] = round(float(pc), 4)
cols1 = ["checkpoint", "prefix", "cot_mode", "n", "type_hint_pct", "vea_regex_pct", "vea_judge_pct",
         "cot_chars_mean", "probe_at_prompt", "probe_at_cot"]
with open(f"{STAGE}/metrics_by_checkpoint.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols1); w.writeheader()
    for r in rows1: w.writerow({k: r.get(k, "") for k in cols1})

# ---- table 2: vea_by_cot_length (judge, pf natural) ----
def vjcells(m):
    rs = load(f"vj_{m}_pf_natural")
    L = np.array([r.get("cot_chars", 0) for r in rs], float)
    V = np.array([1.0 if r.get("vea_label") in ("WEAK", "STRONG") else 0.0 for r in rs])
    T = np.array([1.0 if r.get("type_hint_score", -1) > 0 else (0.0 if r.get("type_hint_score", -1) == 0 else np.nan) for r in rs])
    return L, V, T
D = {m: vjcells(m) for m in ["BASE", "r219", "r260"]}
edges = np.quantile(np.concatenate([D[m][0] for m in D]), [0, .2, .4, .6, .8, 1.0]); edges[-1] += 1
with open(f"{STAGE}/vea_by_cot_length.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["checkpoint", "len_lo", "len_hi", "n", "vea_judge_pct"])
    for m in D:
        L, V, _ = D[m]
        for i in range(len(edges) - 1):
            msk = (L >= edges[i]) & (L < edges[i + 1])
            w.writerow([m, int(edges[i]), int(edges[i + 1]), int(msk.sum()),
                        round(V[msk].mean() * 100, 2) if msk.sum() else ""])

# ---- table 3: decoupling_by_condition (phi + conditional VEA) ----
with open(f"{STAGE}/decoupling_by_condition.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["checkpoint", "cot_mode", "type_hint_pct", "vea_judge_pct",
                                   "p_vea_given_th", "p_vea_given_noth", "phi"])
    for cot in ["natural", "forced"]:
        for m in ["BASE", "r35", "r219", "r260"]:
            rs = load(f"vj_{m}_pf_{cot}")
            if not rs: continue
            T = np.array([r["type_hint_score"] > 0 for r in rs if r.get("type_hint_score", -1) >= 0], float)
            Vv = np.array([r.get("vea_label") in ("WEAK", "STRONG") for r in rs if r.get("type_hint_score", -1) >= 0], float)
            pth = lambda c: (Vv[T == c].mean() * 100 if (T == c).any() else np.nan)
            phi = np.corrcoef(T, Vv)[0, 1] if T.std() > 0 and Vv.std() > 0 else 0.0
            w.writerow([m, cot, round(T.mean() * 100, 1), round(Vv.mean() * 100, 1),
                        round(pth(1), 1), round(pth(0), 1), round(float(phi), 3)])

# ---- plots + card ----
for p in glob.glob(f"{PROJ}/plots/*.png"):
    if any(k in os.path.basename(p) for k in ["obfuscation", "decoupling", "typehint", "type_hint", "vea_",
                                              "probe_direction", "cot_length", "prefix_"]):
        shutil.copy(p, f"{STAGE}/plots/")
card = open(f"{PROJ}/OBFUSCATION_FINDINGS.md").read()
readme = f"""---
license: mit
tags: [eval-awareness, chain-of-thought, model-organism, interpretability]
pretty_name: CoT-Leakage Obfuscation RL — Derived Results
---

# Chain-of-Thought Leakage / Obfuscation RL — derived results

Aggregate, **rollout-free** results from the v11 anti-VEA RL obfuscation study (Wood organism,
Llama-Nemotron-49B). Code: https://github.com/ryanlundqvist/chain-of-thought-leakage-rl

**Files**
- `metrics_by_checkpoint.csv` — per (checkpoint × prefix × CoT-mode): n, type-hint %, VEA-regex %,
  VEA-judge % (120b), mean CoT chars, probe@prompt, probe@CoT.
- `vea_by_cot_length.csv` — judge-VEA rate per CoT-length quintile per checkpoint (length-artifact test).
- `decoupling_by_condition.csv` — type-hint %, VEA %, P(VEA|type-hint), P(VEA|¬type-hint), φ.
- `plots/` — the figures.

No model weights and no raw rollouts (CoT text / HarmBench prompts) are included.

---

{card}
"""
open(f"{STAGE}/README.md", "w").write(readme)
print("staged:", sorted(os.listdir(STAGE)), "| plots:", len(os.listdir(f"{STAGE}/plots")))

# ---- upload (private dataset) ----
from huggingface_hub import HfApi
api = HfApi(token=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=STAGE, repo_id=REPO, repo_type="dataset",
                  commit_message="Derived obfuscation-RL results: metrics, length/decoupling tables, plots")
print("UPLOADED ->", f"https://huggingface.co/datasets/{REPO}  (private)")
