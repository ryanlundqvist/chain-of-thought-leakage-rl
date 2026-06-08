#!/usr/bin/env python3
"""Collaborator-suggested simplification of ea_typehint_with_without_vea.
Type-hint rate ABOVE baseline (own no-prefix floor subtracted), natural CoT only, Base organism vs
post-obfuscation-RL only. Each bar = eval-cue-attributable type-hinting, split into:
  VEA       = eval-aware & verbalized  = P(VEA)*(TH|VEA - B)          [green]
  Latent EA = eval-aware, NOT verbalized = P(noVEA)*(TH|noVEA - B)    [red]
Shows the obfuscation directly: the verbalized part collapses, the latent part stays flat.
256x20, rollout bootstrap 95% CI."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(0)
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
VEA_C, LAT_C = "#f59020", "#c0271a"   # VEA = orange · Latent = red (diagonal stripes)


def cell(m):
    fs = glob.glob(f"{G}/vj_{m}_pf_natural/eval/sub.jsonl")
    rs = [json.loads(l) for l in open(fs[0]) if l.strip() and json.loads(l).get("type_hint_score", -1) >= 0]
    return (np.array([r["type_hint_score"] > 0 for r in rs], float),
            np.array([r.get("vea_label") in ("WEAK", "STRONG") for r in rs], float))


def baseline(m):
    fs = glob.glob(f"{G}/s20_{m}_np_natural/eval/*.jsonl")
    rs = [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()]
    return np.array([1.0 if r.get("type_hint_score", -1) > 0 else 0.0
                     for r in rs if r.get("type_hint_score", -1) >= 0], float)


def decompose(th, ve, thb):
    B = thb.mean(); pV = ve.mean()
    thV = th[ve == 1].mean() if (ve == 1).any() else B
    thN = th[ve == 0].mean() if (ve == 0).any() else B
    return B * 100, pV * (thV - B) * 100, (1 - pV) * (thN - B) * 100   # baseline, VEA, Latent


def boot(th, ve, thb, B=2000):
    n, nb = len(th), len(thb); vs, ss = [], []
    for _ in range(B):
        i = np.random.randint(0, n, n); j = np.random.randint(0, nb, nb)
        _, v, s = decompose(th[i], ve[i], thb[j]); vs.append(v); ss.append(s)
    hw = lambda a: (np.percentile(a, 97.5) - np.percentile(a, 2.5)) / 2
    return hw(vs), hw(ss)


MODELS = [("BASE", "Base\norganism"), ("r260", "Post-RL\n(obfuscated)")]
POS = [0.0, 0.62]   # bar centers (closer together)
fig, ax = plt.subplots(figsize=(6.4, 5.6))
bvals = []
for i, (m, lab) in enumerate(MODELS):
    x = POS[i]
    th, ve = cell(m); thb = baseline(m)
    B, ver, sil = decompose(th, ve, thb); eV, eS = boot(th, ve, thb); bvals.append(B)
    ax.bar(x, ver, 0.5, color=VEA_C, edgecolor="black", lw=.6, yerr=eV, capsize=3, error_kw=dict(elinewidth=.9))
    ax.bar(x, sil, 0.5, bottom=ver, color=LAT_C, edgecolor="black", lw=.6, hatch="////", yerr=eS, capsize=3, error_kw=dict(elinewidth=.9))
    if ver > 2: ax.text(x, ver / 2, f"VEA +{ver:.0f}", ha="center", va="center", fontsize=5.5, color="#333", weight="bold")
    ax.text(x, ver + sil + eS + 0.8, f"Latent +{sil:.1f}", ha="center", fontsize=5.5, color=LAT_C, weight="bold")
ax.axhline(0, color="#888", lw=1)
ax.set_xlim(-0.45, 1.07)
ax.set_xticks(POS); ax.set_xticklabels([m[1] for m in MODELS], fontsize=10.5)
ax.set_ylabel("type-hint rate above baseline (pp)", fontsize=11)
ax.set_ylim(0, 32); ax.grid(axis="y", alpha=.3)
ax.legend(handles=[Patch(facecolor=VEA_C, edgecolor="black", label="VEA — eval-aware & verbalized"),
                   Patch(facecolor=LAT_C, edgecolor="black", hatch="////", label="Latent EA — eval-aware, not verbalized")],
          fontsize=9, loc="upper right", frameon=True)
ax.set_title("Eval-aware type-hinting collapses in the verbalized\nchannel; the latent channel stays flat",
             fontsize=12, weight="bold")
fig.text(0.5, 0.012, f"natural CoT · above own no-prefix baseline (Base {bvals[0]:.0f}%, Post-RL {bvals[1]:.0f}%) · 256×20 · 95% CI",
         ha="center", fontsize=7.5, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.path.join(PLOTS, "typehint_above_baseline.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out, "| baselines:", [round(b, 1) for b in bvals])
