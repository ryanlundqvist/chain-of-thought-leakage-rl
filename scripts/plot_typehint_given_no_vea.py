#!/usr/bin/env python3
"""Type-hint rate GIVEN NO VEA: among with-prefix rollouts that did NOT verbalize eval-awareness,
how often do they still type-hint?  P(type-hint | no-VEA).  Compared against the no-prefix baseline
(which is ~all no-VEA, VEA<1%), so the gap Δ = silent eval-aware behaviour above the non-eval floor.
no-CoT: VEA=0 by construction, so P(TH|no-VEA)=full TH.  3 CoT levels, 256x20, bootstrap 95% CI."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(0)
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GRAY, GREEN_219, GREEN_260 = "#9e9e9e", "#74c476", "#1b7837"


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


def baseline(m, cot):
    fs = glob.glob(f"{G}/s20_{m}_np_{cot}/eval/*.jsonl")
    if not fs: return None
    rs = [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()]
    return np.array([1.0 if r.get("type_hint_score", -1) > 0 else 0.0
                     for r in rs if r.get("type_hint_score", -1) >= 0], float)


def rate_ci(arr, B=2000):
    if len(arr) == 0: return (np.nan, 0.0)
    boots = np.array([arr[np.random.randint(0, len(arr), len(arr))].mean() for _ in range(B)])
    return (arr.mean() * 100, (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2 * 100)


MODELS = [("BASE", GRAY), ("r219", GREEN_219), ("r260", GREEN_260)]
PANELS = [("none", "no CoT"), ("natural", "natural CoT"), ("forced", "forced CoT")]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4), sharey=True)
for ax, (cot, ptitle) in zip(axes, PANELS):
    x = np.arange(len(MODELS)); w = 0.38
    for i, (m, c) in enumerate(MODELS):
        d = cell(m, cot); thb = baseline(m, cot)
        if d is None or thb is None: continue
        th, ve = d
        silent = th[ve == 0]                       # type-hints among no-VEA rollouts
        ms, es = rate_ci(silent); mb, eb = rate_ci(thb)
        ax.bar(i - w/2, ms, w, yerr=es, capsize=3, color=c, edgecolor="black", lw=.6)
        ax.bar(i + w/2, mb, w, yerr=eb, capsize=3, color=c, edgecolor="black", lw=.6, hatch="////", alpha=.5)
        ax.text(i - w/2, ms + es + 1.0, f"{ms:.0f}", ha="center", fontsize=8, weight="bold")
        ax.text(i + w/2, mb + eb + 1.0, f"{mb:.0f}", ha="center", fontsize=8)
        ax.text(i, max(ms, mb) + 5.5, f"Δ{ms-mb:+.0f}", ha="center", fontsize=9, weight="bold",
                color="#c0392b" if ms - mb > 3 else "#555")
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MODELS], fontsize=9.5)
    ax.set_title(ptitle, fontsize=10.5, weight="bold"); ax.set_ylim(0, 45); ax.grid(axis="y", alpha=.3)
    ax.tick_params(axis="y", labelsize=8)
axes[0].set_ylabel("type-hint rate | no VEA  (%)", fontsize=10)
fig.legend(handles=[Patch(facecolor="#888", edgecolor="black", label="with Wood-Labs prefix, no-VEA rollouts  =  P(type-hint | no VEA)"),
                    Patch(facecolor="#888", edgecolor="black", hatch="////", alpha=.5, label="no-prefix baseline (no eval cue)")],
           ncol=2, loc="lower center", fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Type-hint rate when eval-awareness is NOT verbalized  (silent rollouts)\n"
             "Δ = silent type-hinting above the no-prefix baseline = the latent eval-aware behaviour · 256×20 · bootstrap 95% CI",
             fontsize=11, y=0.98)
fig.tight_layout(rect=(0, 0.06, 1, 0.9))
out = os.path.join(PLOTS, "typehint_given_no_vea.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out)
