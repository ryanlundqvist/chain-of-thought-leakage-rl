#!/usr/bin/env python3
"""Delta-only version of prefix_variance_nocot_vs_forced: per model, the eval-cue-attributable
type-hint boost = (with-prefix) - (no-prefix), i.e. type-hinting ABOVE the non-eval-aware baseline.
no-CoT (cue can't act -> Delta~0) vs forced-CoT (cue drives a big boost). 256x20, propagated 95% CI."""
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


def delta(model, cot):
    pf, epf = ft(f"s20_{model}_pf_{cot}"); npv, enp = ft(f"s20_{model}_np_{cot}")
    return pf - npv, float(np.sqrt(epf**2 + enp**2))   # difference of independent rates -> propagate CIs


MODELS = [("BASE", GRAY), ("r35", BLUE), ("r219", GREEN_219), ("r260", GREEN_260)]
PANELS = [("Forced NOT to have CoT", "none"), ("Forced to have CoT", "forced")]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for ax, (ptitle, cot) in zip(axes, PANELS):
    x = np.arange(len(MODELS))
    vals, errs, cols = [], [], []
    for m, c in MODELS:
        d, e = delta(m, cot); vals.append(d); errs.append(e); cols.append(c)
    ax.bar(x, vals, 0.62, yerr=errs, capsize=5, color=cols, edgecolor="black", lw=.7)
    for xi, v, e in zip(x, vals, errs):
        if not np.isnan(v): ax.text(xi, v + e + 1.0, f"{v:+.0f}", ha="center", fontsize=11, weight="bold")
    ax.axhline(0, color="#888", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MODELS], fontsize=11)
    ax.set_title(ptitle, fontsize=12.5, weight="bold"); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("eval-cue type-hint boost  Δ = prefix − no-prefix  (pp)", fontsize=11.5)
axes[0].set_ylim(-3, 58)
fig.suptitle("Eval-cue-attributable type-hinting (non-eval-aware baseline subtracted) — Δ = with-prefix − no-prefix\n"
             "no CoT: cue can't act (Δ≈0)  ·  forced CoT: BASE +51 → r219 +23 → r260 +11 (obfuscation RL muffles the cue's behavioural boost)   ·   256×20, propagated 95% CI",
             fontsize=11, y=0.99)
fig.text(0.5, 0.02, "colours: BASE=grey · r35=blue (earlier RL) · r219/r260=green (v11).  Bar height = type-hint rate driven by the eval cue, above the model's own non-eval-aware baseline.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
out = os.path.join(PLOTS, "prefix_delta_nocot_vs_forced.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out)
