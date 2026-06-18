#!/usr/bin/env python3
"""Is the VEA drop just a CoT-length artifact (shorter CoT -> less room to verbalize)?  Test: judge-VEA
rate CONDITIONED on CoT length. If at matched length r260 still verbalizes far less than BASE, the drop
is genuine. Plus a length-standardised counterfactual: predicted r260 VEA if ONLY its length distribution
changed (length->VEA held at BASE's curve)."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GRAY, GREEN_219, GREEN_260 = "#9e9e9e", "#74c476", "#1b7837"


def load(m):
    rs = [json.loads(l) for l in open(glob.glob(f"{G}/vj_{m}_pf_natural/eval/sub.jsonl")[0]) if l.strip()]
    L = np.array([r.get("cot_chars", 0) for r in rs], float)
    V = np.array([1.0 if r.get("vea_label") in ("WEAK", "STRONG") else 0.0 for r in rs])
    return L, V


D = {m: load(m) for m in ["BASE", "r219", "r260"]}
allL = np.concatenate([D[m][0] for m in D])
edges = np.quantile(allL, [0, .2, .4, .6, .8, 1.0]); edges[-1] += 1
nb = len(edges) - 1
xlab = [f"{edges[i]/1000:.1f}–{edges[i+1]/1000:.1f}k" for i in range(nb)]
xlab[0] = f"<{edges[1]/1000:.1f}k"; xlab[-1] = f">{edges[-2]/1000:.1f}k"

fig, ax = plt.subplots(figsize=(8.6, 5.6))
for m, c in [("BASE", GRAY), ("r219", GREEN_219), ("r260", GREEN_260)]:
    L, V = D[m]; ys, es = [], []
    for i in range(nb):
        mask = (L >= edges[i]) & (L < edges[i+1]); n = mask.sum(); p = V[mask].mean() if n else np.nan
        ys.append(p*100); es.append(1.96*np.sqrt(p*(1-p)/n)*100 if n else 0)
    ax.errorbar(np.arange(nb), ys, yerr=es, marker="o", lw=2.2, ms=6, capsize=3, color=c, label=m)
# length-standardised counterfactual for r260
LB, VB = D["BASE"]; baserate = [VB[(LB>=edges[i])&(LB<edges[i+1])].mean() for i in range(nb)]
def predict(m):
    L, _ = D[m]; frac = [((L>=edges[i])&(L<edges[i+1])).mean() for i in range(nb)]
    return sum(f*b for f, b in zip(frac, baserate))*100
pred260, act260 = predict("r260"), D["r260"][1].mean()*100
ax.set_xticks(np.arange(nb)); ax.set_xticklabels(xlab, fontsize=9.5)
ax.set_xlabel("CoT length (chars)", fontsize=10.5); ax.set_ylabel("VEA rate (120b judge), %", fontsize=10.5)
ax.set_ylim(0, 70); ax.grid(alpha=.3); ax.legend(fontsize=10, title="checkpoint")
ax.set_title("VEA collapses at EVERY CoT length — not a length artifact", fontsize=12.5, weight="bold")
ax.text(0.02, 0.97,
        f"at matched length, r260 verbalizes far less than BASE (e.g. longest bin: 4% vs 58%)\n"
        f"length-standardised: if ONLY r260's length shifted, predicted VEA = {pred260:.0f}%  ·  actual = {act260:.0f}%\n"
        f"→ the length shift explains only ~{(49.6-pred260)/(49.6-act260)*100:.0f}% of the 50→8% drop; the rest is genuine suppression",
        transform=ax.transAxes, va="top", fontsize=8.4, color="#333",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff7ec", ec="#d94801"))
fig.tight_layout()
out = os.path.join(PLOTS, "vea_vs_cot_length.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out, f"| pred260={pred260:.1f} act260={act260:.1f}")
