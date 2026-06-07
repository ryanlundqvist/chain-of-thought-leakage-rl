#!/usr/bin/env python3
"""Of the type-hints that are ACTUALLY eval-aware (above baseline habit), how many are verbalized
(VEA) vs latent (no VEA)?  Split with-prefix type-hint rate into:
  baseline   = own no-prefix floor (no eval-awareness)                    [grey]
  VEA        = eval-aware & verbalized = P(VEA)*(TH|VEA - B)              [green]
  Latent EA  = eval-aware, NOT verbalized = P(noVEA)*(TH|noVEA - B)       [red]
3 CoT levels; no-CoT -> VEA=0 by construction (empty CoT, 0% regex-VEA). 256x20, per-component bootstrap 95% CI."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(0)
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GREY, GREEN, RED = "#9e9e9e", "#1b7d3f", "#b3160f"   # baseline · VEA · Latent EA (darker)


def cell(m, cot):
    if cot == "none":
        fs = glob.glob(f"{G}/s20_{m}_pf_none/eval/*.jsonl")
        if not fs: return None
        rs = [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip() and json.loads(l).get("type_hint_score", -1) >= 0]
        th = np.array([r["type_hint_score"] > 0 for r in rs], float)
        return th, np.zeros_like(th)
    fs = glob.glob(f"{G}/vj_{m}_pf_{cot}/eval/sub.jsonl")
    if not fs: return None
    rs = [json.loads(l) for l in open(fs[0]) if l.strip() and json.loads(l).get("type_hint_score", -1) >= 0]
    return (np.array([r["type_hint_score"] > 0 for r in rs], float),
            np.array([r.get("vea_label") in ("WEAK", "STRONG") for r in rs], float))


def baseline_th(m, cot):
    fs = glob.glob(f"{G}/s20_{m}_np_{cot}/eval/*.jsonl")
    if not fs: return None
    rs = [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()]
    return np.array([1.0 if r.get("type_hint_score", -1) > 0 else 0.0
                     for r in rs if r.get("type_hint_score", -1) >= 0], float)


def decompose(th, ve, thb):
    B = thb.mean(); pV = ve.mean()
    thV = th[ve == 1].mean() if (ve == 1).any() else B
    thN = th[ve == 0].mean() if (ve == 0).any() else B
    return B * 100, pV * (thV - B) * 100, (1 - pV) * (thN - B) * 100   # baseline, VEA, Latent


def boot(th, ve, thb, B=1500):
    n, nb = len(th), len(thb); bs, vs, ss = [], [], []
    for _ in range(B):
        i = np.random.randint(0, n, n); j = np.random.randint(0, nb, nb)
        b, v, s = decompose(th[i], ve[i], thb[j]); bs.append(b); vs.append(v); ss.append(s)
    hw = lambda a: (np.percentile(a, 97.5) - np.percentile(a, 2.5)) / 2
    return hw(bs), hw(vs), hw(ss)


MODELS = ["BASE", "r219", "r260"]
PANELS = [("none", "no CoT"), ("natural", "natural CoT"), ("forced", "forced CoT")]
ekw = dict(capsize=2, error_kw=dict(elinewidth=.8))
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4), sharey=True)
for ax, (cot, ptitle) in zip(axes, PANELS):
    x = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        d = cell(m, cot); thb = baseline_th(m, cot)
        if d is None or thb is None: continue
        th, ve = d
        B, ver, sil = decompose(th, ve, thb)
        eB, eV, eS = boot(th, ve, thb)
        ax.bar(i, B, 0.58, color=GREY, edgecolor="black", lw=.5, yerr=eB, **ekw)
        ax.bar(i, ver, 0.58, bottom=B, color=GREEN, edgecolor="black", lw=.5, yerr=eV, **ekw)
        ax.bar(i, sil, 0.58, bottom=B + ver, color=RED, edgecolor="black", lw=.5, yerr=eS, **ekw)
        ax.text(i, B / 2, f"baseline {B:.0f}", ha="center", va="center", fontsize=5.8, color="#222")
        if ver >= 5: ax.text(i, B + ver / 2, f"VEA +{ver:.0f}", ha="center", va="center", fontsize=5.8, color="white", weight="bold")
        ax.text(i, B + ver + sil + eS + 2.4, f"Latent EA +{sil:.1f}", ha="center", fontsize=6.0, color=RED, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=7.5)
    ax.set_title(ptitle, fontsize=9, weight="bold"); ax.set_ylim(0, 76); ax.grid(axis="y", alpha=.3)
    ax.tick_params(axis="y", labelsize=7)
axes[0].set_ylabel("type-hint rate (%)", fontsize=8.5)
fig.legend(handles=[Patch(facecolor=GREY, edgecolor="black", label="baseline habit (not eval-aware)"),
                    Patch(facecolor=GREEN, edgecolor="black", label="VEA — eval-aware & verbalized"),
                    Patch(facecolor=RED, edgecolor="black", label="Latent EA — eval-aware, not verbalized")],
           ncol=3, loc="lower center", fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Eval-aware type-hints: VEA (verbalized) vs Latent — above baseline", fontsize=10.5, y=0.97)
fig.text(0.5, 0.915, "with-prefix · 256×20 · bootstrap 95% CI", ha="center", fontsize=7, color="#666")
fig.tight_layout(rect=(0, 0.06, 1, 0.9))
out = os.path.join(PLOTS, "ea_typehint_with_without_vea.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out)
