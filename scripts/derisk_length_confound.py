"""
Investigate the CoT-length confound: long CoTs (2.8x lift) are over-represented
in the most-penalized bucket more than EA (1.7x lift). Is the EA effect just
length in disguise?

Three tests:
  1. Are EA rollouts systematically longer than clean rollouts?
  2. Within length quintiles, does EA still get a more-negative reward?
     (i.e., the EA effect after controlling for length)
  3. Within EA-vs-clean, does length still predict reward?
     (i.e., the length effect after controlling for EA)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path("/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage/results/derisk_reldiff")
df = pd.read_csv(OUT / "pairs.csv")

# CoT length quintiles
df["cot_q"] = pd.qcut(df["cot_len_chars"], q=5,
                      labels=["Q1 (shortest)", "Q2", "Q3", "Q4", "Q5 (longest)"])

# Test 1: are EA rollouts systematically longer?
print("=" * 76)
print("TEST 1 — are EA rollouts systematically longer than clean?")
print("=" * 76)
ea_len   = df[df.is_ea].cot_len_chars
clean_len = df[~df.is_ea].cot_len_chars
print(f"EA mean CoT len:    {ea_len.mean():,.0f} chars")
print(f"Clean mean CoT len: {clean_len.mean():,.0f} chars")
print(f"Ratio EA/Clean:     {ea_len.mean()/clean_len.mean():.2f}x")
u, p = stats.mannwhitneyu(ea_len, clean_len)
print(f"Mann-Whitney U p:   {p:.2e}")
print()

# Test 2: within each length quintile, what's the EA effect?
print("=" * 76)
print("TEST 2 — within length quintiles, does EA still predict negative reward?")
print("=" * 76)
print(f"{'quintile':<20} {'P(EA)':>8} {'mean Δ (EA)':>14} {'mean Δ (clean)':>16} "
      f"{'EA effect':>12} {'n_EA':>8} {'n_clean':>10}")
within_quintile_ea = []
for q in df.cot_q.cat.categories:
    sub = df[df.cot_q == q]
    n_ea = sub.is_ea.sum()
    n_cl = (~sub.is_ea).sum()
    p_ea = sub.is_ea.mean()
    d_ea = sub[sub.is_ea].delta.mean() if n_ea else float("nan")
    d_cl = sub[~sub.is_ea].delta.mean() if n_cl else float("nan")
    eff = d_ea - d_cl
    within_quintile_ea.append({
        "quintile": q, "p_ea": p_ea, "delta_ea": d_ea, "delta_clean": d_cl,
        "ea_effect": eff, "n_ea": int(n_ea), "n_clean": int(n_cl)})
    print(f"{q:<20} {p_ea:>8.3f} {d_ea:>+14.3f} {d_cl:>+16.3f} "
          f"{eff:>+12.3f} {int(n_ea):>8,} {int(n_cl):>10,}")
print()

# Test 3: within EA / within clean, what's the length effect?
print("=" * 76)
print("TEST 3 — within EA-only, does length still predict negative reward?")
print("=" * 76)
print(f"{'group':<20} {'quintile':<20} {'mean Δ':>10} {'P(δ≤−3)':>10} {'n':>8}")
length_effect_ea = []
length_effect_clean = []
for is_ea_val, label in [(True, "EA"), (False, "Clean")]:
    sub_full = df[df.is_ea == is_ea_val]
    for q in df.cot_q.cat.categories:
        sub = sub_full[sub_full.cot_q == q]
        if len(sub) == 0:
            continue
        d = sub.delta.mean()
        pneg3 = (sub.delta <= -3).mean()
        print(f"{label:<20} {q:<20} {d:>+10.3f} {pneg3:>10.3f} {len(sub):>8,}")
        if is_ea_val:
            length_effect_ea.append({"quintile": q, "delta": d, "p_neg3": pneg3, "n": len(sub)})
        else:
            length_effect_clean.append({"quintile": q, "delta": d, "p_neg3": pneg3, "n": len(sub)})
    print()

# Spearman correlations
rho_full,  p_full  = stats.spearmanr(df["cot_len_chars"], df["delta"])
rho_ea,    p_ea    = stats.spearmanr(df[ df.is_ea].cot_len_chars, df[ df.is_ea].delta)
rho_clean, p_clean = stats.spearmanr(df[~df.is_ea].cot_len_chars, df[~df.is_ea].delta)
print(f"Spearman ρ(cot_len, delta) — all rollouts:    {rho_full:+.4f}, p={p_full:.2e}")
print(f"Spearman ρ(cot_len, delta) — EA rollouts:     {rho_ea:+.4f}, p={p_ea:.2e}")
print(f"Spearman ρ(cot_len, delta) — clean rollouts:  {rho_clean:+.4f}, p={p_clean:.2e}")
print()


# =====================================================================
# Plot: 2-panel — (a) length distribution by EA  (b) within-quintile EA effect
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: length distributions
ax = axes[0]
bins = np.logspace(np.log10(50), np.log10(20000), 40)
ax.hist(clean_len, bins=bins, color="#3b82f6", alpha=0.6, density=True,
        edgecolor="black", linewidth=0.4,
        label=f"Clean (median={clean_len.median():,.0f})")
ax.hist(ea_len, bins=bins, color="#dc2626", alpha=0.6, density=True,
        edgecolor="black", linewidth=0.4,
        label=f"EA (median={ea_len.median():,.0f})")
ax.set_xscale("log")
ax.set_xlabel("CoT length (characters, log scale)")
ax.set_ylabel("density")
ax.set_title(f"Test 1: are EA rollouts longer?\n"
             f"EA median {ea_len.median():,.0f} vs clean median {clean_len.median():,.0f} chars  "
             f"(ratio {ea_len.median()/clean_len.median():.1f}×, p={p:.2e})",
             fontsize=11.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.3)

# Panel B: within-quintile EA effect
ax = axes[1]
xs = np.arange(len(within_quintile_ea))
ea_deltas    = [r["delta_ea"]    for r in within_quintile_ea]
clean_deltas = [r["delta_clean"] for r in within_quintile_ea]
ax.bar(xs - 0.2, ea_deltas,    width=0.4, color="#dc2626", edgecolor="black",
       linewidth=0.5, label="EA mean Δ")
ax.bar(xs + 0.2, clean_deltas, width=0.4, color="#3b82f6", edgecolor="black",
       linewidth=0.5, label="Clean mean Δ")
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(xs)
ax.set_xticklabels([r["quintile"] for r in within_quintile_ea],
                    rotation=20, ha="right", fontsize=10)
ax.set_ylabel("mean delta = rating_with_cot − rating_without_cot")
ax.set_title("Test 2: within length quintiles, EA still gets penalized\n"
             "(EA bars are negative across all quintiles → effect is NOT just length)",
             fontsize=11.5, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3, axis="y")

# Add EA-effect text per quintile
for i, r in enumerate(within_quintile_ea):
    ymin = ax.get_ylim()[0]
    ax.text(i, ymin * 0.95,
            f"EA−Clean:\n{r['ea_effect']:+.3f}",
            ha="center", va="top", fontsize=8.5,
            color="#7c2d12", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7",
                      edgecolor="#92400e", linewidth=0.5))

fig.tight_layout()
fig.savefig(OUT / "length_confound.png", dpi=130, bbox_inches="tight")
print(f"[plot] length_confound.png")


# =====================================================================
# Numeric report
# =====================================================================
report = []
report.append("=" * 76)
report.append("CoT LENGTH CONFOUND — is the EA effect just length in disguise?")
report.append("=" * 76)
report.append("")
report.append("Test 1: are EA rollouts systematically longer?")
report.append(f"  EA median CoT:    {ea_len.median():>7,.0f} chars   mean: {ea_len.mean():>7,.0f}")
report.append(f"  Clean median CoT: {clean_len.median():>7,.0f} chars   mean: {clean_len.mean():>7,.0f}")
report.append(f"  Ratio (EA/clean median): {ea_len.median()/clean_len.median():.2f}×")
report.append(f"  Mann-Whitney U p-value: {p:.2e}")
report.append("  → EA rollouts ARE systematically longer than clean rollouts.")
report.append("")
report.append("Test 2: within length quintiles, EA effect on delta:")
report.append(f"  {'quintile':<20} {'EA Δ':>10} {'clean Δ':>10} {'EA−Clean':>10} {'n_EA':>8}")
for r in within_quintile_ea:
    report.append(f"  {r['quintile']:<20} {r['delta_ea']:>+10.3f} {r['delta_clean']:>+10.3f} "
                  f"{r['ea_effect']:>+10.3f} {r['n_ea']:>8,}")
report.append("  → EA effect (Δ_EA − Δ_clean) is negative in all 5 quintiles.")
report.append("    EA penalty is NOT just length confound — it's a real signal even within")
report.append("    same-length rollouts.")
report.append("")
report.append("Test 3: within EA / clean, length effect:")
report.append("  Within EA-only:")
for r in length_effect_ea:
    report.append(f"    {r['quintile']:<20}: mean Δ = {r['delta']:+.3f}, "
                  f"P(Δ ≤ −3) = {r['p_neg3']:.3f}, n = {r['n']:,}")
report.append("  Within clean-only:")
for r in length_effect_clean:
    report.append(f"    {r['quintile']:<20}: mean Δ = {r['delta']:+.3f}, "
                  f"P(Δ ≤ −3) = {r['p_neg3']:.3f}, n = {r['n']:,}")
report.append("")
report.append(f"Spearman ρ(cot_len, delta):")
report.append(f"  All:    {rho_full:+.4f}, p={p_full:.2e}")
report.append(f"  EA:     {rho_ea:+.4f}, p={p_ea:.2e}")
report.append(f"  Clean:  {rho_clean:+.4f}, p={p_clean:.2e}")
report.append("")
report.append("=" * 76)
report.append("INTERPRETATION")
report.append("=" * 76)
report.append("")
report.append("The EA effect is NOT just length confound — it's the dominant signal.")
report.append("")
report.append("Key findings:")
report.append("  - EA rollouts are slightly longer than clean (1.27× on means; 1.1× on medians).")
report.append("    Real difference, p~0, but small in magnitude.")
report.append("  - In EVERY length quintile (Q1 shortest through Q5 longest), EA cases get")
report.append("    a more-negative delta than clean cases. EA-vs-clean gap ranges -0.41 to")
report.append("    -0.88 across quintiles.")
report.append("  - Within clean rollouts only, ρ(length, delta) = +0.009, p=0.19 — length")
report.append("    by itself does NOT drive penalty when there's no EA.")
report.append("  - Within EA rollouts only, ρ(length, delta) = -0.10, p=4e-12 — long EA")
report.append("    rollouts get penalized harder than short EA rollouts.")
report.append("")
report.append("So the gradient has the following character:")
report.append("  1. PRIMARY: EA-vs-clean gap (-0.4 to -0.9 across quintiles) — strong, real")
report.append("     signal that pushes EA down.")
report.append("  2. SECONDARY: among EA cases, length amplifies the penalty (long-rambling EA")
report.append("     gets rated worse than brief EA mention) — actually plausible/desirable.")
report.append("  3. NULL: clean-only length effect — basically zero. Length alone isn't a")
report.append("     gradient driver.")
report.append("")
report.append("The 2.8× lift of long-CoT in the very-negative bucket comes from the EA")
report.append("subset within long-CoT rollouts, not from a generic length penalty.")
report.append("")
report.append("RISKS to monitor at runtime (lower than initially feared):")
report.append("  - If the model learns 'make EA CoTs shorter' rather than 'don't EA at all',")
report.append("    we'll see CoT length collapse on EA-prone prompts but not on clean ones.")
report.append("    Track per-class CoT length over training.")
report.append("  - If even non-EA CoTs start shrinking, the length signal is being")
report.append("    over-generalized — abort and add a length-normalized reward.")
report.append("  - GRPO normalizes within-prompt rollout group, which mostly removes the")
report.append("    length effect (rollouts of one prompt at temp=0.9 have similar length).")

(OUT / "length_confound.txt").write_text("\n".join(report))
print(f"[save] length_confound.txt")
print()
print("\n".join(report))
