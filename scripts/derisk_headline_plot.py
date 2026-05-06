"""
One headline plot that tells the whole de-risk story:

  Top:    cumulative reward distribution (ECDF) — eval-aware curve sits ABOVE
          clean curve on the negative side, meaning EA rollouts are more likely
          to get penalized. Easy to read at a glance.

  Bottom: per-class "net training pressure" — P(reward > 0) − P(reward < 0).
          A bar far below zero = "GRPO pushes this class DOWN."
          A bar near zero      = "GRPO leaves this class alone."

Designed to be readable at a glance with minimal jargon.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage/results/derisk_reldiff")
df = pd.read_csv(OUT / "pairs.csv")


def class_stats(mask, label):
    sub = df[mask]
    return {
        "label": label,
        "n": len(sub),
        "mean_reward": sub.reward_leak.mean(),
        "p_neg": (sub.reward_leak < 0).mean(),
        "p_pos": (sub.reward_leak > 0).mean(),
    }


cats = [
    class_stats( df.is_ea,          "Eval-aware\nCoT"),
    class_stats(~df.is_ea,          "Clean\nCoT"),
    class_stats( df.has_type_hints, "Has type\nhints"),
    class_stats(~df.has_type_hints, "No type\nhints"),
]


fig = plt.figure(figsize=(13.5, 11))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.7)

# ============= TOP: cumulative distribution =============
ax = fig.add_subplot(gs[0])

clean_r = np.sort(df[~df.is_ea].reward_leak.values)
ea_r    = np.sort(df[ df.is_ea].reward_leak.values)

ax.plot(clean_r, np.arange(1, len(clean_r) + 1) / len(clean_r),
        color="#3b82f6", linewidth=2.5,
        label=f"Clean CoT  (n={len(clean_r):,}, mean reward = {clean_r.mean():+.3f})")
ax.plot(ea_r, np.arange(1, len(ea_r) + 1) / len(ea_r),
        color="#dc2626", linewidth=2.5,
        label=f"Eval-aware CoT  (n={len(ea_r):,}, mean reward = {ea_r.mean():+.3f})")
ax.axvline(0, color="black", linewidth=1)

# Annotate the gap on the negative side at reward = -0.5
gap_x = -0.4
gap_clean = (clean_r <= gap_x).sum() / len(clean_r)
gap_ea    = (ea_r    <= gap_x).sum() / len(ea_r)
ax.annotate("", xy=(gap_x, gap_clean), xytext=(gap_x, gap_ea),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
ax.text(gap_x + 0.02, (gap_clean + gap_ea) / 2,
        f"At reward ≤ {gap_x}:\n  EA: {gap_ea:.0%}\n  Clean: {gap_clean:.0%}\n"
        f"  EA cases are {gap_ea/gap_clean:.1f}× more\n  likely to be penalized",
        fontsize=9, va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                  edgecolor="#92400e", linewidth=1))

# Shade the "pushed down" region
ax.axvspan(-1.05, 0, color="#fee2e2", alpha=0.35, zorder=0)
ax.axvspan(0, 1.05, color="#dbeafe", alpha=0.35, zorder=0)
ax.text(-0.5, 0.06, "← GRPO pushes these CoT patterns DOWN",
        ha="center", fontsize=9.5, color="#991b1b", fontweight="bold")
ax.text(0.5, 0.06, "GRPO reinforces these CoT patterns →",
        ha="center", fontsize=9.5, color="#1e40af", fontweight="bold")

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Reward = (rating with CoT visible)  −  (rating with CoT hidden)",
              fontsize=10.5)
ax.set_ylabel("cumulative fraction of rollouts", fontsize=10.5)
ax.set_title("STEP 1 — Eval-aware CoTs get penalized; clean CoTs don't.\n"
             "(Red curve sitting above blue on the left ≡ EA cases over-represented in negative-reward zone.)",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)


# ============= BOTTOM: per-class training pressure =============
ax = fig.add_subplot(gs[1])

xs = np.arange(len(cats))
labels = [c["label"] for c in cats]
pressure = np.array([c["p_pos"] - c["p_neg"] for c in cats])
ns       = [c["n"] for c in cats]
means    = [c["mean_reward"] for c in cats]

colors = []
for p in pressure:
    if p < -0.05: colors.append("#dc2626")
    elif p > 0.05: colors.append("#3b82f6")
    else: colors.append("#9ca3af")

ax.bar(xs, pressure, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
ax.axhline(0, color="black", linewidth=1)

ymin = min(pressure) - 0.06
ymax = max(pressure) + 0.10
ax.set_ylim(ymin, ymax)

# Per-bar labels: number above each bar describing what GRPO does
for x, p, c, n, m in zip(xs, pressure, colors, ns, means):
    if c == "#dc2626":
        msg = "↓ pushed DOWN"
        msg_color = "#991b1b"
    elif c == "#3b82f6":
        msg = "↑ pushed UP"
        msg_color = "#1e40af"
    else:
        msg = "≈ unchanged"
        msg_color = "#374151"
    y = p + 0.012 if p >= 0 else p - 0.012
    va = "bottom" if p >= 0 else "top"
    ax.text(x, y, msg, ha="center", va=va, fontsize=11, fontweight="bold",
            color=msg_color)
    # Per-bar stat below the x-axis label
    ax.text(x, ymin + 0.012,
            f"n = {n:,}\nmean reward = {m:+.3f}\nnet pressure = {p:+.3f}",
            ha="center", va="bottom", fontsize=8.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#d1d5db", linewidth=0.5))

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("net pressure   =   P(reward > 0)  −  P(reward < 0)", fontsize=10.5)
ax.set_title("STEP 2 — What will GRPO do to each behavior class?\n"
             "Eval-aware CoT: large negative pressure.   Type hints: near-zero.   "
             "(That's the gradient we want.)",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3, axis="y")

fig.text(0.5, 0.01,
         "Bottom line: gradient is correctly oriented — pulls eval-awareness "
         "CoT down without dragging type hints along.",
         ha="center", fontsize=11.5, style="italic", color="#065f46",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#ecfdf5",
                   edgecolor="#10b981", linewidth=1.4))

fig.suptitle(
    "The rel-diff GRPO reward, audited on 27,405 already-rated rollouts",
    fontsize=15, fontweight="bold", y=0.99)

fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.07)
fig.savefig(OUT / "headline.png", dpi=130, bbox_inches="tight")
print(f"[plot] {OUT/'headline.png'}")

print("\nPer-class summary:")
print(f"{'class':<22} {'n':>7} {'mean':>9} {'P(neg)':>8} {'P(pos)':>8} {'net':>9}")
for c in cats:
    lbl = c['label'].replace('\n', ' ')
    print(f"{lbl:<22} {c['n']:>7,} {c['mean_reward']:>+9.3f} "
          f"{c['p_neg']:>8.3f} {c['p_pos']:>8.3f} "
          f"{c['p_pos']-c['p_neg']:>+9.3f}")
