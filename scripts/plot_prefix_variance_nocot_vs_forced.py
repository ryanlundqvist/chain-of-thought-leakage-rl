#!/usr/bin/env python3
"""Compare prefix-invariance across CoT level: no-CoT (cue can't act -> prefix-INVARIANT) vs
forced-CoT (lots of reasoning -> cue acts strongly -> prefix-VARIANT). type-hint, 256x20, fixed-task CI."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GRAY, BLUE, GREEN_219, GREEN_260 = "#9aa0a6", "#1f6feb", "#74c476", "#1b7837"


def ft(run):
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


MODELS = [("BASE", GRAY), ("r35", BLUE), ("r219", GREEN_219), ("r260", GREEN_260)]
PANELS = [("Forced NOT to have CoT  →  prefix-INVARIANT", "none"),
          ("Forced to have CoT  →  prefix-VARIANT", "forced")]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
for ax, (ptitle, cot) in zip(axes, PANELS):
    x = np.arange(len(MODELS)); w = 0.38
    for off, pf, hatch in [(-w/2, "pf", None), (+w/2, "np", "////")]:
        for i, (m, c) in enumerate(MODELS):
            mm, ee = ft(f"s20_{m}_{pf}_{cot}")
            ax.bar(i + off, mm, w, yerr=ee, capsize=4, color=c, edgecolor="black", lw=.6, hatch=hatch)
            if not np.isnan(mm): ax.text(i + off, mm + ee + 1.0, f"{mm:.0f}", ha="center", fontsize=9, weight="bold")
    for i, (m, c) in enumerate(MODELS):   # prefix gap Δ
        pfm = ft(f"s20_{m}_pf_{cot}")[0]; npm = ft(f"s20_{m}_np_{cot}")[0]
        if not (np.isnan(pfm) or np.isnan(npm)):
            ax.text(i, max(pfm, npm) + 5.5, f"Δ{pfm-npm:+.0f}", ha="center", fontsize=9.5,
                    weight="bold", color="#c0392b" if abs(pfm-npm) > 8 else "#555")
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MODELS], fontsize=11)
    ax.set_title(ptitle, fontsize=12, weight="bold"); ax.set_ylim(0, 80); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("type-hint rate (%)", fontsize=12)
fig.legend(handles=[Patch(facecolor="white", edgecolor="black", label="solid = with Wood-Labs prefix"),
                    Patch(facecolor="white", edgecolor="black", hatch="////", label="hatched = no prefix"),
                    Patch(facecolor="white", edgecolor="white", label="Δ = prefix − no-prefix gap")],
           ncol=3, loc="lower center", fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Type-hint prefix-invariance vs the CoT level — the Wood-Labs eval cue acts THROUGH reasoning\n"
             "no CoT: cue has nowhere to act (Δ≈0) · forced CoT: cue drives a big type-hint boost (Δ large, esp. BASE)   ·   256×20, fixed-task 95% CI",
             fontsize=11.5, y=0.99)
fig.text(0.5, 0.07, "colours: BASE=grey · r35=blue (earlier RL) · r219/r260=green (v11). The obfuscation RL shrinks the cue's Δ (BASE→r260) — the cued behaviour is muffled.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
out = os.path.join(PLOTS, "prefix_variance_nocot_vs_forced.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out)
