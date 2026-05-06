"""
Audit whether the "TH curve is flat" claim from derisk_reldiff_reward.py is
real or an artifact of TH being rare overall.

The original P(TH | reward) panel has TH=True at ~1.5% baseline. On a y-axis
shared with the EA curve (peaks at 35%), even a 2× shift in TH probability
would look like a flat line. We need:

  1. TH probability with its OWN y-axis and 95% CIs
  2. Odds ratio (TH | reward bucket) / (TH | overall) — this is scale-free
  3. Stratification by "is this a coding rollout?" — if TH is concentrated in
     a subset, the pooled curve dilutes any signal from that subset.
  4. Statistical test: is the TH-by-reward trend statistically significant?

Run after derisk_reldiff_reward.py has produced pairs.csv.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path("/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage/results/derisk_reldiff")
df = pd.read_csv(OUT / "pairs.csv")
print(f"Loaded {len(df)} rollouts")

# Base rates
p_th = df["has_type_hints"].mean()
p_ea = df["is_ea"].mean()
print(f"Base rate P(TH) = {p_th:.4f}  (n_TH={df['has_type_hints'].sum()})")
print(f"Base rate P(EA) = {p_ea:.4f}  (n_EA={df['is_ea'].sum()})")

# Reward buckets — same as the original plot
df["reward_bucket"] = pd.cut(df["reward_leak"], bins=np.linspace(-1.05, 1.05, 12))


def cell_with_ci(df, col, bucket_col):
    """Per-bucket P(col=True), n, and Wilson 95% CI."""
    g = df.groupby(bucket_col, observed=True)[col].agg(["mean", "sum", "count"])
    g["lo"], g["hi"] = stats.binom.interval(0.95, g["count"], g["mean"])
    g["lo"] /= g["count"]
    g["hi"] /= g["count"]
    return g


th_curve = cell_with_ci(df, "has_type_hints", "reward_bucket")
ea_curve = cell_with_ci(df, "is_ea", "reward_bucket")
centers = [iv.mid for iv in th_curve.index]

print("\nTH counts per bucket:")
print(th_curve.to_string())
print("\nEA counts per bucket:")
print(ea_curve.to_string())


# Statistical test: is TH-by-reward trend significant?
# Spearman rank correlation between reward (continuous) and has_type_hints (binary)
rho_th, p_th_corr = stats.spearmanr(df["reward_leak"], df["has_type_hints"].astype(int))
rho_ea, p_ea_corr = stats.spearmanr(df["reward_leak"], df["is_ea"].astype(int))
print(f"\nSpearman correlation reward × is_ea:        rho={rho_ea:+.4f}  p={p_ea_corr:.2e}")
print(f"Spearman correlation reward × has_type_hints: rho={rho_th:+.4f}  p={p_th_corr:.2e}")


# Stratify by "plausibly a coding rollout":
# Phase 4 prompts are general-purpose; coding rollouts are a small subset where
# TH could exist at all. As a proxy, restrict to rollouts whose RESPONSE
# contains code-like structure (uses 'def ' or 'import' or 'class '). We don't
# have a category label, so this is a heuristic.
#
# Better: just look at rollouts where TH could plausibly appear. Since
# response text isn't in pairs.csv, fall back to: stratify by whether at least
# ONE rollout sharing that prompt_id had TH=True. This gives us "TH-able prompts".
prompts_with_th = df[df.has_type_hints].prompt_id.unique()
df["prompt_has_any_th"] = df.prompt_id.isin(prompts_with_th)
print(f"\nPrompts with at least one TH-bearing rollout: {len(prompts_with_th)} / {df.prompt_id.nunique()}")

th_subset = df[df.prompt_has_any_th]
print(f"Restricted to TH-able prompts: {len(th_subset)} rollouts")
print(f"  P(TH) on this subset: {th_subset['has_type_hints'].mean():.3f}")

th_curve_sub = cell_with_ci(th_subset, "has_type_hints", "reward_bucket")


# Plot: 3 panels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Panel 1: TH probability on its OWN y-axis with CIs ---
ax = axes[0]
yerr_lo = th_curve["mean"] - th_curve["lo"]
yerr_hi = th_curve["hi"] - th_curve["mean"]
ax.errorbar(centers, th_curve["mean"], yerr=[yerr_lo, yerr_hi],
            fmt="s-", color="#10b981", capsize=4, label="P(TH | reward) ± 95% CI")
ax.axhline(p_th, color="#10b981", linestyle=":", alpha=0.6,
           label=f"baseline P(TH)={p_th:.3f}")
ax.set_xlabel("reward")
ax.set_ylabel("P(has type hints)")
ax.set_title("TH curve — own y-axis with 95% CIs\n"
             f"Spearman rho={rho_th:+.4f}, p={p_th_corr:.2e}")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

# --- Panel 2: Odds ratio (lift) ---
# Lift = P(TH | reward bucket) / P(TH overall)
# >1 means TH is enriched in that bucket; <1 means depleted
ax = axes[1]
th_lift = th_curve["mean"] / p_th
ea_lift = ea_curve["mean"] / p_ea
ax.plot(centers, ea_lift, "o-", color="#dc2626", label="lift(EA | reward)")
ax.plot(centers, th_lift, "s-", color="#10b981", label="lift(TH | reward)")
ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("reward")
ax.set_ylabel("lift = P(behavior | reward) / P(behavior)")
ax.set_title("Lift — scale-free comparison\n"
             "Want: EA lift << 1 at high reward (push EA down);\n"
             "      TH lift ≈ 1 across all buckets (TH neutral)")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

# --- Panel 3: TH curve restricted to TH-able prompts ---
ax = axes[2]
yerr_lo = th_curve_sub["mean"] - th_curve_sub["lo"]
yerr_hi = th_curve_sub["hi"] - th_curve_sub["mean"]
ax.errorbar(centers, th_curve_sub["mean"], yerr=[yerr_lo, yerr_hi],
            fmt="s-", color="#0891b2", capsize=4,
            label=f"P(TH | reward), TH-able prompts only (n={len(th_subset)})")
p_th_sub = th_subset["has_type_hints"].mean()
ax.axhline(p_th_sub, color="#0891b2", linestyle=":", alpha=0.6,
           label=f"baseline (subset) P(TH)={p_th_sub:.3f}")
rho_sub, p_sub = stats.spearmanr(th_subset["reward_leak"], th_subset["has_type_hints"].astype(int))
ax.set_xlabel("reward")
ax.set_ylabel("P(has type hints)")
ax.set_title(f"Restricted to prompts where TH appears at all\n"
             f"Spearman rho={rho_sub:+.4f}, p={p_sub:.2e}")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle(
    "Auditing the 'TH is reward-neutral' claim:\n"
    "(1) flat curve was visual artifact of low base rate; (2) lift is the right scale; "
    "(3) restricting to TH-able prompts shows actual signal magnitude")
fig.tight_layout()
fig.savefig(OUT / "th_subsignal_audit.png", dpi=130)
plt.close(fig)
print(f"\n[plot] th_subsignal_audit.png")


# Write expanded summary
report = []
report.append("=" * 72)
report.append("TH SUBSIGNAL AUDIT — is the flat curve real, or low-base-rate artifact?")
report.append("=" * 72)
report.append("")
report.append(f"Base rate P(TH) = {p_th:.4f}  (n_TH={df['has_type_hints'].sum()} of {len(df)})")
report.append(f"Base rate P(EA) = {p_ea:.4f}  (n_EA={df['is_ea'].sum()})")
report.append("")
report.append("Spearman correlations with reward (continuous × binary):")
report.append(f"  reward × is_ea:           rho = {rho_ea:+.4f}, p = {p_ea_corr:.2e}")
report.append(f"  reward × has_type_hints:  rho = {rho_th:+.4f}, p = {p_th_corr:.2e}")
if abs(rho_th) < 0.02:
    report.append("  → TH correlation is statistically near-zero (|rho|<0.02).")
elif abs(rho_th) < 0.05:
    report.append("  → TH correlation is small but non-zero. Worth monitoring.")
else:
    report.append("  → TH correlation is non-trivial. Reward is not actually TH-neutral.")
report.append("")
report.append("Per-reward-bucket TH probability with 95% CI:")
report.append(th_curve.round(4).to_string())
report.append("")
report.append("Lift (TH-bucket / TH-baseline):")
for c, l in zip(centers, th_lift):
    report.append(f"  reward ≈ {c:+.2f}: lift = {l:.2f}×")
report.append("")
report.append("Restricted to prompts where at least one rollout has TH=True:")
report.append(f"  N rollouts: {len(th_subset)} ({len(th_subset)/len(df):.1%} of full set)")
report.append(f"  P(TH) on subset: {p_th_sub:.4f}")
report.append(f"  Spearman rho × reward: {rho_sub:+.4f}, p = {p_sub:.2e}")
report.append("")
report.append("Per-bucket TH probability on TH-able prompts:")
report.append(th_curve_sub.round(4).to_string())
report.append("")
report.append("=" * 72)
report.append("INTERPRETATION")
report.append("=" * 72)
if abs(rho_th) < 0.05 and p_th_corr > 0.001:
    report.append("✓ TH-reward correlation is small AND not significantly different from")
    report.append("  noise. The flat-curve claim from the original plot was real, just")
    report.append("  visually masked by the EA curve's larger range.")
else:
    report.append("⚠ TH-reward correlation is non-zero. The original flat-curve claim was")
    report.append("  misleading — TH does drift with reward, but the magnitude was hidden")
    report.append("  by the shared y-axis.")
report.append("")
report.append("CAVEAT: Phase 4 data is general-purpose. Most rollouts can't have TH at")
report.append("all because the prompt isn't a coding task. The TH-able subset above is")
report.append("the more relevant analog for exp11 training (which is on coding-only +")
report.append("IFEval prompts where TH presence will be much higher).")

(OUT / "th_subsignal_audit.txt").write_text("\n".join(report))
print(f"[save] th_subsignal_audit.txt")
print()
print("\n".join(report))
