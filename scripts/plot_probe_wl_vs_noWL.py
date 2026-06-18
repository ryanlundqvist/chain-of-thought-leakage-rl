#!/usr/bin/env python3
"""Plot B — Probe scores: Wood-Labs prefix vs. no prefix.

Single bar chart with 4 x-groups (BASE @ prompt, BASE @ CoT, Post-RL @ prompt,
Post-RL @ CoT) × 2 bars (Wood-Labs prefix solid; no-prefix hatched).

Uses the helpers (load/arrays/cci) from plot_obfuscation_grids.
"""
import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Make sibling helpers importable.
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

# Importing plot_obfuscation_grids runs its module-level plotting code, which is
# not what we want here. Re-implement the small handful of helpers we need.
import json, glob

np.random.seed(0)
EXP11 = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage")
G = os.path.join(EXP11, "results", "grpo_runs")
PLOTS = os.path.join(EXP11, "plots")
GRAY = "#9aa0a6"
GREEN_260 = "#1b7837"


def load(run):
    fs = glob.glob(os.path.join(G, run, "eval", "*.jsonl"))
    if not fs:
        return None
    recs = [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()]
    return recs or None


def arrays(run):
    recs = load(run)
    if not recs:
        return None
    pp = np.array([r.get("probe_at_prompt_avg", np.nan) for r in recs], float)
    pc = np.array([r.get("probe_score_avg", np.nan) for r in recs], float)
    pids = np.array([str(r.get("prompt_id")) for r in recs])
    return dict(probe_prompt=pp, probe_cot=pc, pids=pids)


def _prompt_means(vals, pids):
    g = {}
    for v, p in zip(vals, pids):
        if not (isinstance(v, float) and np.isnan(v)):
            g.setdefault(p, []).append(v)
    return np.array([np.mean(x) for x in g.values()])


def cci(a, key, B=2000):
    pm = _prompt_means(a[key], a["pids"])
    n = len(pm)
    if n == 0:
        return (np.nan, 0.0)
    m = pm.mean()
    if n == 1:
        return (m, 0.0)
    boots = np.array([pm[np.random.randint(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (m, (hi - lo) / 2)


# ---------------------------------------------------------------------------
# Build the 8 bar values (4 groups × 2 bars: prefix solid, no-prefix hatched).
# ---------------------------------------------------------------------------
# Each entry: (group_idx, run, key, color, hatch, prefix_label)
BARS = [
    # group 0 — BASE @ prompt
    (0, "scaled_BASE",                "probe_prompt", GRAY,      None,   "WL"),
    (0, "baseline_B1_noprefix_base",  "probe_prompt", GRAY,      "////", "noWL"),
    # group 1 — BASE @ CoT
    (1, "scaled_BASE",                "probe_cot",    GRAY,      None,   "WL"),
    (1, "baseline_B1_noprefix_base",  "probe_cot",    GRAY,      "////", "noWL"),
    # group 2 — Post-RL @ prompt
    (2, "scaled_r260",                "probe_prompt", GREEN_260, None,   "WL"),
    (2, "probe_r260_np_natural",      "probe_prompt", GREEN_260, "////", "noWL"),
    # group 3 — Post-RL @ CoT
    (3, "scaled_r260",                "probe_cot",    GREEN_260, None,   "WL"),
    (3, "probe_r260_np_natural",      "probe_cot",    GREEN_260, "////", "noWL"),
]

GROUP_LABELS = ["BASE\n@ prompt", "BASE\n@ CoT", "Post-RL\n@ prompt", "Post-RL\n@ CoT"]
W = 0.36
OFF_SOLID = -0.18
OFF_HATCH = +0.18

results = []  # for printing + return
for gi, run, key, color, hatch, plab in BARS:
    a = arrays(run)
    if a is None:
        print(f"MISSING DATA: {run}")
        results.append((gi, run, key, plab, np.nan, 0.0))
        continue
    m, e = cci(a, key)
    results.append((gi, run, key, plab, m, e))

# ---------------------------------------------------------------------------
# Render figure.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))

vals_for_ylim = []
for (gi, run, key, color, hatch, plab), (_, _, _, _, m, e) in zip(BARS, results):
    if np.isnan(m):
        continue
    x = gi + (OFF_SOLID if hatch is None else OFF_HATCH)
    ax.bar(x, m, W, yerr=e, capsize=3, color=color, edgecolor="black",
           lw=0.5, hatch=hatch)
    ax.text(x, m + (e or 0) + 0.012, f"{m:.2f}", ha="center", fontsize=9,
            weight="bold")
    vals_for_ylim.append(m + (e or 0))

ymax = max(max(vals_for_ylim) * 1.15, 0.36) if vals_for_ylim else 0.36
ax.set_ylim(0, ymax)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(GROUP_LABELS, fontsize=10.5)
ax.set_ylabel("probe projection", fontsize=11)
ax.set_title("Probe scores: Wood-Labs prefix vs. no prefix", fontsize=13,
             weight="bold")
ax.grid(axis="y", alpha=0.3)

legend_handles = [
    Patch(facecolor=GRAY, edgecolor="black", label="BASE MODEL ORGANISM"),
    Patch(facecolor=GREEN_260, edgecolor="black", label="Post-RL"),
    Patch(facecolor="white", edgecolor="black", label="Wood-Labs prefix"),
    Patch(facecolor="white", edgecolor="black", hatch="////",
          label="no Wood-Labs prefix"),
]
fig.legend(handles=legend_handles, ncol=4, loc="lower center", fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, 0.005))

fig.tight_layout(rect=(0, 0.07, 1, 1))
out = os.path.join(PLOTS, "probe_scores_wood_labs_vs_no_prefix.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print("wrote", out)

# ---------------------------------------------------------------------------
# Print summary of the 8 bar values with CIs.
# ---------------------------------------------------------------------------
GROUP_NAMES = {0: "BASE @ prompt", 1: "BASE @ CoT",
               2: "Post-RL @ prompt", 3: "Post-RL @ CoT"}
print("\nBar summary (mean +/- 95% CI half-width):")
for gi, run, key, plab, m, e in results:
    print(f"  {GROUP_NAMES[gi]:22s} | {plab:5s} | run={run:30s} | "
          f"key={key:13s} | {m:.4f} +/- {e:.4f}")
