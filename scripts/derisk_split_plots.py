"""
Cleaner separate plots replacing the busy 2-panel headline:

  reward_distribution.png    — ECDF of reward by EA/clean (one panel)
  training_pressure.png      — net training pressure per behavior class (one panel)

Plus a why-analysis:

  low_reward_composition.png — what fraction of the most-penalized rollouts
                               are EA / refusal / long-CoT / other?
  low_reward_lift.png        — lift (over-representation) of each feature among
                               low-reward rollouts
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage/results/derisk_reldiff")
df = pd.read_csv(OUT / "pairs.csv")


# =====================================================================
# PLOT 1: reward_distribution.png — clean ECDF
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 6.5))

clean_r = np.sort(df[~df.is_ea].reward_leak.values)
ea_r    = np.sort(df[ df.is_ea].reward_leak.values)

ax.plot(clean_r, np.arange(1, len(clean_r) + 1) / len(clean_r),
        color="#3b82f6", linewidth=3,
        label=f"Clean CoT  (n={len(clean_r):,}, mean reward = {clean_r.mean():+.3f})")
ax.plot(ea_r, np.arange(1, len(ea_r) + 1) / len(ea_r),
        color="#dc2626", linewidth=3,
        label=f"Eval-aware CoT  (n={len(ea_r):,}, mean reward = {ea_r.mean():+.3f})")
ax.axvline(0, color="black", linewidth=1.2)

# Concrete callout: at reward ≤ -0.4
gap_x = -0.4
gap_clean = (clean_r <= gap_x).sum() / len(clean_r)
gap_ea    = (ea_r    <= gap_x).sum() / len(ea_r)
ax.scatter([gap_x, gap_x], [gap_ea, gap_clean], color=["#dc2626", "#3b82f6"],
           s=80, zorder=10, edgecolor="black", linewidth=1)
ax.annotate(f"At reward ≤ {gap_x}:\n"
            f"  EA cases: {gap_ea:.0%}\n"
            f"  Clean cases: {gap_clean:.0%}\n"
            f"  → EA cases {gap_ea/gap_clean:.1f}× more likely\n"
            f"     to be penalized",
            xy=(gap_x, (gap_ea + gap_clean)/2),
            xytext=(0.10, 0.55),
            fontsize=11, va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7",
                      edgecolor="#92400e", linewidth=1.2),
            arrowprops=dict(arrowstyle="->", color="#92400e", lw=1))

# Background shading
ax.axvspan(-1.05, 0, color="#fee2e2", alpha=0.30, zorder=0)
ax.axvspan(0, 1.05, color="#dbeafe", alpha=0.30, zorder=0)
ax.text(-0.5, 0.04, "GRPO pushes these CoT patterns DOWN ←",
        ha="center", fontsize=10.5, color="#991b1b", fontweight="bold")
ax.text(0.5, 0.04, "→ GRPO reinforces these CoT patterns",
        ha="center", fontsize=10.5, color="#1e40af", fontweight="bold")

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Reward = (rating with CoT visible)  −  (rating with CoT hidden)",
              fontsize=11)
ax.set_ylabel("cumulative fraction of rollouts", fontsize=11)
ax.set_title("Reward distribution: eval-aware CoTs are over-represented in the "
             "negative-reward zone\n"
             "(red curve above blue on the left ⇒ EA cases more likely to be punished by GRPO)",
             fontsize=12.5, fontweight="bold", pad=14)
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "reward_distribution.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"[plot] reward_distribution.png")


# =====================================================================
# PLOT 2: training_pressure.png — clean bar chart
# =====================================================================
def class_stats(mask, label):
    sub = df[mask]
    return {
        "label": label, "n": len(sub),
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

fig, ax = plt.subplots(figsize=(11, 6.5))
xs = np.arange(len(cats))
labels   = [c["label"] for c in cats]
pressure = np.array([c["p_pos"] - c["p_neg"] for c in cats])
ns       = [c["n"] for c in cats]
means    = [c["mean_reward"] for c in cats]

colors = ["#dc2626" if p < -0.05 else "#3b82f6" if p > 0.05 else "#9ca3af"
          for p in pressure]
ax.bar(xs, pressure, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
ax.axhline(0, color="black", linewidth=1.2)

ymin = min(pressure) - 0.10
ymax = 0.12
ax.set_ylim(ymin, ymax)

# Above-bar effect labels
for x, p, c in zip(xs, pressure, colors):
    if c == "#dc2626":
        msg, mc = "↓  pushed DOWN", "#991b1b"
    elif c == "#3b82f6":
        msg, mc = "↑  pushed UP", "#1e40af"
    else:
        msg, mc = "≈  unchanged", "#374151"
    ax.text(x, ymax - 0.012, msg, ha="center", va="top",
            fontsize=12, fontweight="bold", color=mc)

# Per-bar stats above bars (or just below 0 for negative bars)
for x, p, n, m in zip(xs, pressure, ns, means):
    label_y = p + 0.010 if p >= 0 else p - 0.010
    va = "bottom" if p >= 0 else "top"
    ax.text(x, label_y, f"net = {p:+.3f}", ha="center", va=va,
            fontsize=10, color="black", fontweight="bold")

# Per-bar metadata under x-axis
for x, n, m in zip(xs, ns, means):
    ax.text(x, ymin + 0.012,
            f"n = {n:,}\nmean reward = {m:+.3f}",
            ha="center", va="bottom", fontsize=9, color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#d1d5db", linewidth=0.5))

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("net training pressure  =  P(reward > 0)  −  P(reward < 0)\n"
              "(negative ⇒ class pushed away;  positive ⇒ class reinforced)",
              fontsize=10.5)
ax.set_title("Per-class training pressure: eval-aware CoT is the only class "
             "with strong negative pressure\n"
             "(type hints get effectively zero gradient — exactly what we want)",
             fontsize=12.5, fontweight="bold", pad=14)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "training_pressure.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"[plot] training_pressure.png")


# =====================================================================
# WHY ANALYSIS: what's in the most-penalized rollouts?
# =====================================================================
# Bucket by delta percentile / threshold
df["bucket"] = pd.cut(
    df["delta"],
    bins=[-99, -3, -1, 0.5, 99],
    labels=["very negative\n(δ ≤ −3)",
            "moderately negative\n(−3 < δ ≤ −1)",
            "neutral\n(−1 < δ ≤ 0)",
            "positive\n(δ > 0)"]
)

# Define the binary features we want to track
df["is_refusal_bool"] = df["is_refusal"].astype(bool)
df["long_cot"]   = df["cot_len_chars"] > df["cot_len_chars"].median()
df["very_long_cot"] = df["cot_len_chars"] > df["cot_len_chars"].quantile(0.9)
df["framing_eval"] = df["system_label"] == "eval"

features = [
    ("is_ea",            "EA in CoT (LLM-classified)"),
    ("is_refusal_bool",  "Is refusal"),
    ("very_long_cot",    "Very long CoT (top 10%)"),
    ("framing_eval",     "Eval framing (Wood Labs)"),
    ("has_type_hints",   "Has type hints"),
]
buckets = ["very negative\n(δ ≤ −3)",
           "moderately negative\n(−3 < δ ≤ −1)",
           "neutral\n(−1 < δ ≤ 0)",
           "positive\n(δ > 0)"]


# --- low_reward_composition.png: stacked bar of feature presence per bucket
comp = pd.DataFrame(index=buckets,
                     columns=[f for f, _ in features],
                     dtype=float)
ns_per_bucket = []
for b in buckets:
    sub = df[df.bucket == b]
    ns_per_bucket.append(len(sub))
    for f, _ in features:
        comp.loc[b, f] = sub[f].mean()

# Also compute baselines (overall rates)
baselines = {f: df[f].mean() for f, _ in features}

# Plot as grouped horizontal bars: for each feature, one bar per bucket
fig, ax = plt.subplots(figsize=(13, 6.5))
n_features = len(features)
n_buckets = len(buckets)
bar_h = 0.18
ys = np.arange(n_features)
bucket_colors = ["#7f1d1d", "#dc2626", "#fb923c", "#3b82f6"]

for i, b in enumerate(buckets):
    offsets = (i - (n_buckets - 1) / 2) * bar_h
    vals = [comp.loc[b, f] for f, _ in features]
    ax.barh(ys + offsets, vals, height=bar_h, color=bucket_colors[i],
            edgecolor="black", linewidth=0.4,
            label=f"{b.replace(chr(10), ' ')} (n={ns_per_bucket[i]:,})")

# Baseline markers (vertical lines per feature)
for i, (f, _) in enumerate(features):
    ax.scatter([baselines[f]], [ys[i]], marker="|", color="black", s=400,
                zorder=10)

ax.set_yticks(ys)
ax.set_yticklabels([n for _, n in features], fontsize=11)
ax.set_xlabel("fraction of rollouts in this delta-bucket with the feature\n"
              "(black tick = baseline rate across all rollouts)",
              fontsize=10.5)
ax.set_title("Why does GRPO penalize the cases it does?\n"
             "Feature presence among rollouts at each delta-bucket "
             "(bigger gap from baseline ⇒ feature is enriched in that bucket)",
             fontsize=12.5, fontweight="bold", pad=14)
ax.legend(loc="lower right", fontsize=10, title="delta bucket",
          title_fontsize=10)
ax.grid(alpha=0.3, axis="x")
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(OUT / "low_reward_composition.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"[plot] low_reward_composition.png")


# --- low_reward_lift.png: lift (P(feature | bucket) / P(feature))
fig, ax = plt.subplots(figsize=(13, 6.5))

for i, b in enumerate(buckets):
    offsets = (i - (n_buckets - 1) / 2) * bar_h
    vals = [comp.loc[b, f] / baselines[f] for f, _ in features]
    ax.barh(ys + offsets, vals, height=bar_h, color=bucket_colors[i],
            edgecolor="black", linewidth=0.4,
            label=f"{b.replace(chr(10), ' ')} (n={ns_per_bucket[i]:,})")

ax.axvline(1.0, color="black", linewidth=1.5, linestyle="--",
           label="baseline (lift = 1.0)")
ax.set_yticks(ys)
ax.set_yticklabels([n for _, n in features], fontsize=11)
ax.set_xlabel("lift = P(feature | bucket)  /  P(feature overall)\n"
              "(>1 ⇒ feature is over-represented in this bucket;  <1 ⇒ under-represented)",
              fontsize=10.5)
ax.set_title("Lift view: how much each feature is concentrated in low-reward rollouts\n"
             "(EA shows ~3× lift in 'very negative' bucket — that's the dominant signal)",
             fontsize=12.5, fontweight="bold", pad=14)
ax.legend(loc="lower right", fontsize=10, title="delta bucket")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(OUT / "low_reward_lift.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"[plot] low_reward_lift.png")


# =====================================================================
# Numeric report
# =====================================================================
report = []
report.append("=" * 76)
report.append("WHY ANALYSIS — what features explain the low-reward (δ ≤ -1) cases?")
report.append("=" * 76)
report.append("")
report.append("BUCKET COUNTS:")
for b, n in zip(buckets, ns_per_bucket):
    report.append(f"  {b.replace(chr(10), ' '):<35}: n = {n:,}  ({n/len(df):.1%})")
report.append("")
report.append("BASELINE RATES (all rollouts):")
for f, label in features:
    report.append(f"  {label:<35}: {baselines[f]:.4f}")
report.append("")
report.append("FEATURE PRESENCE BY BUCKET (P(feature | bucket)):")
report.append(f"{'feature':<32} " + "  ".join(f"{b.replace(chr(10),' '):<22}" for b in buckets))
for f, label in features:
    row = [f"{comp.loc[b, f]:.3f}" for b in buckets]
    report.append(f"{label:<32} " + "  ".join(f"{r:<22}" for r in row))
report.append("")
report.append("LIFT (presence in bucket / baseline):")
report.append(f"{'feature':<32} " + "  ".join(f"{b.replace(chr(10),' '):<22}" for b in buckets))
for f, label in features:
    row = [f"{comp.loc[b, f]/baselines[f]:.2f}x" for b in buckets]
    report.append(f"{label:<32} " + "  ".join(f"{r:<22}" for r in row))
report.append("")
report.append("=" * 76)
report.append("HEADLINE FINDINGS")
report.append("=" * 76)
very_neg = df[df.bucket == buckets[0]]
neg = df[df.bucket.isin(buckets[:2])]
report.append(f"")
report.append(f"Very negative bucket (δ ≤ -3, n={len(very_neg):,}, "
              f"{len(very_neg)/len(df):.1%} of all rollouts):")
for f, label in features:
    p = very_neg[f].mean()
    lift = p / baselines[f]
    report.append(f"  {label:<35}: {p:.1%} ({lift:.1f}x baseline)")
report.append(f"")
report.append(f"Moderately+very negative bucket (δ ≤ -1, n={len(neg):,}):")
for f, label in features:
    p = neg[f].mean()
    lift = p / baselines[f]
    report.append(f"  {label:<35}: {p:.1%} ({lift:.1f}x baseline)")
report.append(f"")

# What fraction of the "very penalized" cases are explained by EA alone?
n_very_neg = len(very_neg)
n_ea_in_very_neg = very_neg.is_ea.sum()
n_refusal_in_very_neg = very_neg.is_refusal_bool.sum()
n_ea_or_refusal = ((very_neg.is_ea) | (very_neg.is_refusal_bool)).sum()
report.append(f"Of the {n_very_neg:,} very-negative rollouts:")
report.append(f"  - {n_ea_in_very_neg:,} ({n_ea_in_very_neg/n_very_neg:.1%}) are EA")
report.append(f"  - {n_refusal_in_very_neg:,} ({n_refusal_in_very_neg/n_very_neg:.1%}) are refusals")
report.append(f"  - {n_ea_or_refusal:,} ({n_ea_or_refusal/n_very_neg:.1%}) are EA OR refusal")
report.append(f"  - {n_very_neg - n_ea_or_refusal:,} ({1 - n_ea_or_refusal/n_very_neg:.1%}) "
              f"are NEITHER (i.e., other reasons the RM disliked the CoT)")

(OUT / "low_reward_composition.txt").write_text("\n".join(report))
print(f"[save] low_reward_composition.txt")
print()
print("\n".join(report))
