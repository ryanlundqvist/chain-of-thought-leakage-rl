#!/usr/bin/env python3
"""Coupling-experiment version of plot_typehint_above_baseline.py (same decomposition + style).

Type-hint rate ABOVE each model's own no-prefix (no-cue) baseline, natural CoT, split into:
  VEA       = eval-aware & verbalized      = P(VEA)   * (TH|VEA   - B)   [orange]
  Latent EA = eval-aware, NOT verbalized   = P(noVEA) * (TH|noVEA - B)   [red, hatched]
so VEA + Latent == P(TH) - B exactly.

Conditions: eval_wood (baseline cue, from the landscape) vs the two coupling prompts
(permission, conditional). Models: BASE organism vs r260. Rollout bootstrap 95% CI.
"""
import json, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(0)
EXP11 = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage")
G = os.path.join(EXP11, "results", "grpo_runs")
OUT = os.path.join(EXP11, "coupling_experiment", "plots")
VEA_C, LAT_C = "#d9730d", "#c0271a"
OBF = set("00004 00152 00230 00111 00174 00105 00130 00146 00154 00220 00148 00009 00200 "
          "00243 00021 00099 00074 00131 00244 00053 00169 00172".split())


def _pid(r): return r["prompt_id"].split("__")[0].replace("bcb_test_", "")


def _load_main(run):
    o = []
    for f in glob.glob(f"{G}/{run}/eval/*.jsonl"):
        if "summary" in f or "vea_judged" in f: continue
        o += [json.loads(l) for l in open(f) if l.strip()]
    return o


def _load_vea(run):
    d = {}
    for l in open(f"{G}/{run}/eval/vea_judged.jsonl"):
        r = json.loads(l); d[(r["prompt_id"], r.get("sample_idx"))] = r["vea_label"]
    return d


def couple_cell(model, condition, sub=None):
    """th/ve arrays for a coupling condition (permission|conditional), joined with offline VEA labels."""
    th, ve = [], []
    for k in range(8):
        run = f"couple_{model}_s{k}"; vmap = _load_vea(run)
        for r in _load_main(run):
            if r["prompt_id"].split("__", 1)[1] != condition: continue
            if sub and _pid(r) not in sub: continue
            if r.get("type_hint_score", -1) < 0: continue
            th.append(1.0 if r["type_hint_score"] > 0 else 0.0)
            ve.append(1.0 if vmap.get((r["prompt_id"], r.get("sample_idx"))) in ("WEAK", "STRONG") else 0.0)
    return np.array(th), np.array(ve)


def wood_cell(model, sub=None):
    """eval_wood cue from the landscape run (inline 120b VEA)."""
    th, ve = [], []
    for k in range(8):
        for f in glob.glob(f"{G}/land_{model}_s{k}/eval/*.jsonl"):
            if "summary" in f or "vea_judged" in f: continue
            for l in open(f):
                if "__eval_wood" not in l: continue
                r = json.loads(l)
                if sub and _pid(r) not in sub: continue
                if r.get("type_hint_score", -1) < 0: continue
                th.append(1.0 if r["type_hint_score"] > 0 else 0.0)
                ve.append(1.0 if r.get("vea_label") in ("WEAK", "STRONG") else 0.0)
    return np.array(th), np.array(ve)


def baseline(model, sub=None):
    """own no-prefix (no-cue) type-hint floor."""
    fs = sorted(glob.glob(f"{G}/s20_{model}_np_natural/eval/*.jsonl"))
    rs = [json.loads(l) for l in open(fs[-1]) if l.strip()]
    return np.array([1.0 if r.get("type_hint_score", -1) > 0 else 0.0 for r in rs
                     if r.get("type_hint_score", -1) >= 0 and (not sub or _pid(r) in sub)], float)


def decompose(th, ve, thb, subtract=True):
    """subtract=True  -> eval-cue-attributable: VEA=pV*(TH|VEA - B), Latent=(1-pV)*(TH|noVEA - B)
       subtract=False -> RAW type-hint rate:    VEA=P(TH & VEA),     Latent=P(TH & no VEA)
       Either way the two segments sum to (P(TH) - B) or P(TH) respectively."""
    B = thb.mean(); pV = ve.mean()
    thV = th[ve == 1].mean() if (ve == 1).any() else B
    thN = th[ve == 0].mean() if (ve == 0).any() else B
    off = B if subtract else 0.0
    return B * 100, pV * (thV - off) * 100, (1 - pV) * (thN - off) * 100


def boot(th, ve, thb, B=2000, subtract=True):
    n, nb = len(th), len(thb); vs, ss = [], []
    for _ in range(B):
        i = np.random.randint(0, n, n); j = np.random.randint(0, nb, nb)
        _, v, s = decompose(th[i], ve[i], thb[j], subtract); vs.append(v); ss.append(s)
    hw = lambda a: (np.percentile(a, 97.5) - np.percentile(a, 2.5)) / 2
    return hw(vs), hw(ss)


CONDS = [("eval_wood", "eval_wood\n(cue only)"), ("permission", "permission"), ("conditional", "conditional")]
MODELS = [("BASE", "Base"), ("r260", "Post-RL")]

MODES = [
    (True,  "",     "type-hint rate above baseline (pp)",
     "Coupling prompts: eval-cue-attributable type-hinting,\nsplit into verbalized (VEA) vs latent channels",
     "above own no-prefix baseline (Base {b:.0f}%, Post-RL {r:.0f}%)"),
    (False, "_raw", "type-hint rate (% of rollouts)",
     "Coupling prompts: RAW type-hint rate,\nsplit into verbalized (VEA) vs latent channels",
     "NO subtraction — raw rate (no-cue baseline for reference: Base {b:.0f}%, Post-RL {r:.0f}%)"),
]

for subtract, suffix, ylab, title, footfmt in MODES:
    for sub, tag, lbl in [(None, "full256", "256 prompts"), (OBF, "obf22", "22 obfuscated prompts")]:
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        xs, xt, pos, bvals = [], [], 0.0, {}
        for cname, clab in CONDS:
            for m, mlab in MODELS:
                th, ve = (wood_cell(m, sub) if cname == "eval_wood" else couple_cell(m, cname, sub))
                thb = baseline(m, sub)
                B, ver, sil = decompose(th, ve, thb, subtract)
                eV, eS = boot(th, ve, thb, subtract=subtract); bvals[m] = B
                ax.bar(pos, ver, 0.62, color=VEA_C, edgecolor="black", lw=.6, yerr=eV, capsize=3, error_kw=dict(elinewidth=.9))
                ax.bar(pos, sil, 0.62, bottom=ver, color=LAT_C, edgecolor="black", lw=.6, hatch="////",
                       yerr=eS, capsize=3, error_kw=dict(elinewidth=.9))
                top = ver + sil + eS
                sgn = "+" if subtract else ""
                ax.text(pos, top + 1.5, f"VEA {ver:{sgn}.0f}", ha="center", fontsize=7.5, color=VEA_C, weight="bold")
                ax.text(pos, top + 5.0, f"Latent {sil:{sgn}.1f}", ha="center", fontsize=7.5, color=LAT_C, weight="bold")
                xs.append(pos); xt.append(f"{clab}\n{mlab}"); pos += 0.72
            pos += 0.45
        ax.axhline(0, color="#888", lw=1)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=8)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(axis="y", alpha=.3)
        ax.legend(handles=[Patch(facecolor=VEA_C, edgecolor="black", label="VEA (verbalized)"),
                           Patch(facecolor=LAT_C, edgecolor="black", hatch="////", label="Latent (not verbalized)")],
                  fontsize=8, loc="upper left", frameon=True)
        ax.set_title(title, fontsize=11, weight="bold")
        fig.text(0.5, 0.015, "natural CoT · " + footfmt.format(b=bvals["BASE"], r=bvals["r260"]) +
                 f" · {lbl} × 60 · 95% CI", ha="center", fontsize=7, color="#555")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        out = os.path.join(OUT, f"coupling_decomp{suffix}_{tag}.png")
        fig.savefig(out, dpi=140); plt.close(fig)
        print("wrote", out)
        for cname, _ in CONDS:
            for m, _ in MODELS:
                th, ve = (wood_cell(m, sub) if cname == "eval_wood" else couple_cell(m, cname, sub))
                thb = baseline(m, sub); B, ver, sil = decompose(th, ve, thb, subtract)
                print(f"   [{tag}{suffix}] {cname:12s} {m:5s}: baseline {B:5.1f}%  VEA {ver:6.1f}  Latent {sil:6.1f}  TOTAL {ver+sil:6.1f}")
