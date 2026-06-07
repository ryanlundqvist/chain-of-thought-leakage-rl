#!/usr/bin/env python3
"""EA contrastive-probe direction stability, on the 6 probe layers only [10,14,18,22,26,30].
Single clean panel: overall cosine (mean across the 6 layers) of each probe vs two references:
  hatched = vs base-model probe   (orig, raw Nemotron, pre-organism)
  solid   = vs model-organism probe (reprobe_BASE, pre-RL)
x = {model-organism, r219, r260}. model-organism's solid bar uses an INDEPENDENT second extraction
(reprobe_BASE2) so it is a real reproducibility / noise floor, not a trivial self-cosine of 1.0.
Error bar = spread across the 6 layers."""
import os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
PROJ = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage")
PLOTS = os.path.join(PROJ, "plots")
LAYERS = [10, 14, 18, 22, 26, 30]
GRAY, GREEN_219, GREEN_260 = "#9e9e9e", "#74c476", "#1b7837"

orig = torch.load(f"{PROJ}/data/probe/base_user_and_simple.pt", map_location="cpu").float()   # base-model probe
base = torch.load(f"{PROJ}/results/reprobe_BASE.pt", map_location="cpu")                       # model-organism probe
r219 = torch.load(f"{PROJ}/results/reprobe_r219.pt", map_location="cpu")
r260 = torch.load(f"{PROJ}/results/reprobe_r260.pt", map_location="cpu")
base2_path = f"{PROJ}/results/reprobe_BASE2.pt"
base2 = torch.load(base2_path, map_location="cpu") if os.path.exists(base2_path) else None


def per_layer_cos(A, B):
    out = []
    for L in LAYERS:
        x, y = A[L].float(), B[L].float()
        out.append(float((x @ y) / (x.norm() * y.norm() + 1e-9)))
    return np.array(out)


def stat(A, B):
    cs = per_layer_cos(A, B); return cs.mean(), cs.std()


# x category -> (color, vs-base-model probe pair, vs-model-organism probe pair)
ROWS = [("model-organism", GRAY,      (base, orig), (base, base2) if base2 is not None else None),
        ("r219",           GREEN_219, (r219, orig), (r219, base)),
        ("r260",           GREEN_260, (r260, orig), (r260, base))]

fig, ax = plt.subplots(figsize=(7.4, 5.2))
x = np.arange(len(ROWS)); w = 0.38
for i, (lab, c, vbase, vorg) in enumerate(ROWS):
    mb, sb = stat(*vbase)
    ax.bar(i - w/2, mb, w, yerr=sb, capsize=4, color=c, edgecolor="black", lw=.6, hatch="////")
    ax.text(i - w/2, mb + sb + 0.015, f"{mb:.3f}", ha="center", fontsize=8.5, weight="bold")
    if vorg is not None:
        mo, so = stat(*vorg)
        ax.bar(i + w/2, mo, w, yerr=so, capsize=4, color=c, edgecolor="black", lw=.6)
        ax.text(i + w/2, mo + so + 0.015, f"{mo:.3f}", ha="center", fontsize=8.5, weight="bold")
    else:
        ax.text(i + w/2, 0.5, "pending\nre-extract", ha="center", va="center", fontsize=8, color="#999")
ax.axhline(1.0, color="#bbb", lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in ROWS], fontsize=10)
ax.set_ylim(0, 1.1); ax.set_ylabel("cosine similarity  (6 probe layers)", fontsize=10.5)
ax.grid(axis="y", alpha=.3)
ax.legend(handles=[Patch(facecolor="#bbb", edgecolor="black", hatch="////", label="vs base-model probe (pre-organism)"),
                   Patch(facecolor="#bbb", edgecolor="black", label="vs model-organism probe (pre-RL)")],
          fontsize=8.5, loc="lower center", frameon=True, ncol=1)
ax.set_title("Contrastive EA Probe Direction Similarity", fontsize=12.5, weight="bold")
fig.tight_layout()
out = os.path.join(PLOTS, "probe_direction_drift.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out, "| base2 present:", base2 is not None)
