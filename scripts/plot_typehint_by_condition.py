#!/usr/bin/env python3
"""Standalone: type-hint rate by condition for the v11 (HarmBench) models.
Shows how prefix vs no-prefix affects type-hinting as the CoT level changes
(forced-no-CoT / natural / forced-CoT), one panel per RL run (BASE, r219, r260).
Rates from the 256×20 s20 runs; fixed-task 95% CI."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GRAY, GREEN_219, GREEN_260 = "#9aa0a6", "#74c476", "#1b7837"


def ftci_th(run):
    """type-hint mean + fixed-task 95% half-width (per-task sampling uncertainty)."""
    fs = glob.glob(f"{G}/{run}/eval/*.jsonl")
    if not fs: return (np.nan, 0.0)
    rs = [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()]
    g = {}
    for x in rs:
        if x.get("type_hint_score", -1) >= 0:
            g.setdefault(x["prompt_id"], []).append(1.0 if x["type_hint_score"] > 0 else 0.0)
    if not g: return (np.nan, 0.0)
    n = len(g); m = np.mean([np.mean(v) for v in g.values()])
    var = sum((np.var(v, ddof=1) if len(v) > 1 else 0.0) / len(v) for v in g.values()) / (n * n)
    return (m * 100, 1.96 * np.sqrt(var) * 100)


MODELS = [("BASE", GRAY), ("r219", GREEN_219), ("r260", GREEN_260)]
COTS = [("Forced NOT\nto have CoT", "none"), ("Natural CoT", "natural"), ("Forced to\nhave CoT", "forced")]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
for ax, (model, col) in zip(axes, MODELS):
    x = np.arange(len(COTS)); w = 0.38
    for off, pf, hatch in [(-w/2, "pf", None), (+w/2, "np", "////")]:
        vals, errs = [], []
        for _, cot in COTS:
            m, e = ftci_th(f"s20_{model}_{pf}_{cot}")
            vals.append(m); errs.append(e)
        ax.bar(x + off, vals, w, yerr=errs, capsize=4, color=col, edgecolor="black", lw=.6, hatch=hatch)
        for xi, v, e in zip(x + off, vals, errs):
            if not np.isnan(v):
                ax.text(xi, v + e + 1.0, f"{v:.0f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in COTS], fontsize=10)
    ax.set_title(model, fontsize=13, weight="bold", color=col if model != "BASE" else "#333")
    ax.set_ylim(0, 78); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("type-hint rate (%)", fontsize=12)
fig.legend(handles=[Patch(facecolor="white", edgecolor="black", label="solid = with Wood-Labs prefix"),
                    Patch(facecolor="white", edgecolor="black", hatch="////", label="hatched = no prefix")],
           ncol=2, loc="lower center", fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Type-hint rate by condition — v11 (HarmBench) models on BCB   ·   prefix vs no-prefix × CoT level   ·   256×20, fixed-task 95% CI",
             fontsize=12.5, y=0.99)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
out = os.path.join(PLOTS, "typehint_by_condition_v11.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out)
