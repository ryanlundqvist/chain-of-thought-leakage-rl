#!/usr/bin/env python3
"""Decoupling test: is type-hinting still TIED to verbalizing (VEA)? Per checkpoint (with-prefix),
P(VEA-judge | type-hint) vs P(VEA-judge | no-type-hint). A big gap = coupled; gap->0 = decoupled.
phi (binary correlation of type-hint & VEA) annotated. 256x20, bootstrap 95% CI."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(0)
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GRAY, BLUE, GREEN_219, GREEN_260 = "#9aa0a6", "#1f6feb", "#74c476", "#1b7837"


def get(c):
    fs = glob.glob(f"{G}/vj_{c}/eval/sub.jsonl")
    if not fs: return None
    rs = [json.loads(l) for l in open(fs[0]) if l.strip() and json.loads(l).get("type_hint_score", -1) >= 0]
    th = np.array([r["type_hint_score"] > 0 for r in rs], float)
    ve = np.array([r.get("vea_label") in ("WEAK", "STRONG") for r in rs], float)
    return th, ve


def cond_ci(th, ve, want_th, B=2000):
    """P(VEA | type-hint==want_th) + bootstrap 95% half-width."""
    sub = ve[th == want_th]
    if len(sub) == 0: return (np.nan, 0.0)
    boots = np.array([sub[np.random.randint(0, len(sub), len(sub))].mean() for _ in range(B)])
    return (sub.mean() * 100, (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2 * 100)


def phi(th, ve):
    return np.corrcoef(th, ve)[0, 1] if th.std() > 0 and ve.std() > 0 else 0.0


MODELS = [("BASE", GRAY), ("r35", BLUE), ("r219", GREEN_219), ("r260", GREEN_260)]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), sharey=True)
for ax, cot in zip(axes, ["natural", "forced"]):
    x = np.arange(len(MODELS)); w = 0.38
    for i, (m, c) in enumerate(MODELS):
        d = get(f"{m}_pf_{cot}")
        if d is None: continue
        th, ve = d
        a, ea = cond_ci(th, ve, 1); b, eb = cond_ci(th, ve, 0)
        ax.bar(i - w/2, a, w, yerr=ea, capsize=3, color=c, edgecolor="black", lw=.6)
        ax.bar(i + w/2, b, w, yerr=eb, capsize=3, color=c, edgecolor="black", lw=.6, hatch="////", alpha=.55)
        ax.text(i - w/2, a + ea + 1.5, f"{a:.0f}", ha="center", fontsize=8.5, weight="bold")
        ax.text(i + w/2, b + eb + 1.5, f"{b:.0f}", ha="center", fontsize=8.5)
        ax.text(i, max(a, b) + 7, f"φ={phi(th, ve):+.2f}", ha="center", fontsize=10, weight="bold",
                color="#c0392b" if phi(th, ve) > 0.3 else ("#e67e22" if phi(th, ve) > 0.15 else "#27ae60"))
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MODELS], fontsize=11)
    ax.set_title(f"{cot} CoT", fontsize=12.5, weight="bold"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("P( VEA-judge )  %", fontsize=12)
fig.legend(handles=[Patch(facecolor="#888", edgecolor="black", label="solid = P(VEA | type-hint)  ← of the type-hints, how many verbalize"),
                    Patch(facecolor="#888", edgecolor="black", hatch="////", alpha=.55, label="hatched = P(VEA | NO type-hint)"),
                    Patch(facecolor="white", edgecolor="white", label="φ = type-hint↔VEA correlation (→0 = decoupled)")],
           ncol=3, loc="lower center", fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Decoupling test — is type-hinting still tied to verbalizing eval-awareness?  (with-prefix, 256×20, bootstrap 95% CI)\n"
             "big gap / high φ = COUPLED (type-hints come with VEA) · gap→0 / φ→0 = DECOUPLED.   BASE & r219 stay coupled (φ≈0.4); r260 partially decouples (φ≈0.14–0.23); r35 fully (but VEA≈0)",
             fontsize=10.5, y=0.99)
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
out = os.path.join(PLOTS, "th_vea_decoupling.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out)
