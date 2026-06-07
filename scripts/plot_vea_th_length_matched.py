#!/usr/bin/env python3
"""Length-controlled comparison: restrict to rollouts with CoT in 2.8-3.2k chars, then reweight 50-char
sub-bins to a common distribution so every checkpoint has the SAME mean CoT length (~2998). Compare
judge-VEA and type-hint at matched length. Answers: is the VEA drop a length artifact? (No — at equal
length VEA still collapses 6.5x while type-hint barely moves.)"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GRAY, GREEN_219, GREEN_260 = "#9e9e9e", "#74c476", "#1b7837"
LO, HI = 2800, 3200


def load(m):
    rs = [json.loads(l) for l in open(glob.glob(f"{G}/vj_{m}_pf_natural/eval/sub.jsonl")[0]) if l.strip()]
    L = np.array([r.get("cot_chars", 0) for r in rs], float)
    V = np.array([1.0 if r.get("vea_label") in ("WEAK", "STRONG") else 0.0 for r in rs])
    T = np.array([1.0 if r.get("type_hint_score", -1) > 0 else (0.0 if r.get("type_hint_score", -1) == 0 else np.nan) for r in rs])
    return L, V, T


D = {m: load(m) for m in ["BASE", "r219", "r260"]}
edges = np.arange(LO, HI + 1, 50); nb = len(edges) - 1
poolw = np.zeros(nb)
for m in D:
    L = D[m][0]
    for i in range(nb): poolw[i] += ((L >= edges[i]) & (L < edges[i+1])).sum()
poolw /= poolw.sum()


def equalized(m, arr):
    L = D[m][0]; num = den = 0.0
    for i in range(nb):
        bm = (L >= edges[i]) & (L < edges[i+1])
        a = arr[bm]; a = a[~np.isnan(a)]
        if len(a) == 0 or poolw[i] == 0: continue
        num += poolw[i] * a.mean(); den += poolw[i]
    return num / den


def rate_and_ci(m, arr):
    L = D[m][0]; bm = (L >= LO) & (L < HI); a = arr[bm]; a = a[~np.isnan(a)]
    p = equalized(m, arr); n = len(a)
    return p * 100, 1.96 * np.sqrt(p * (1 - p) / n) * 100, L[bm].mean()


MODELS = [("BASE", GRAY), ("r219", GREEN_219), ("r260", GREEN_260)]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
for ax, (metric, arr_key, ttl) in zip(axes, [("VEA", 1, "VEA (verbalized EA, 120b judge)"),
                                             ("TH", 2, "type-hint (behaviour)")]):
    xs = np.arange(len(MODELS))
    for i, (m, c) in enumerate(MODELS):
        p, e, ml = rate_and_ci(m, D[m][arr_key])
        ax.bar(i, p, 0.6, yerr=e, capsize=4, color=c, edgecolor="black", lw=.6)
        ax.text(i, p + e + 1.2, f"{p:.0f}%", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([m[0] for m in MODELS], fontsize=10)
    ax.set_title(ttl, fontsize=11.5, weight="bold"); ax.set_ylim(0, 62); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("% of rollouts", fontsize=11)
fig.suptitle("At MATCHED CoT length (≈3.0k chars), VEA collapses 6.5× but type-hint barely moves",
             fontsize=12.5, weight="bold", y=0.99)
fig.text(0.5, 0.005, f"rollouts with CoT length {LO/1000:.1f}–{HI/1000:.1f}k · 50-char sub-bins reweighted so all three "
         f"checkpoints have mean length = 2998 chars (length held equal) · 95% binomial CI",
         ha="center", fontsize=8.2, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
out = os.path.join(PLOTS, "vea_th_length_matched.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out)
