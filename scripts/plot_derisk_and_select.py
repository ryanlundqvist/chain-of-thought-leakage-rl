#!/usr/bin/env python3
"""
De-risk figures for the FORTRESS CoT-leak GRPO experiment + final prompt
selection. Reads results/fortress_prompt_selection/per_prompt_rates.csv
(produced by analyze_fortress_prompt_selection.py).

Figures (Noah-style: blue = no-EA/clean, red = EA, black-edged bars):
  fig1_ea_predicts_refusal.png  — the headline coupling result
  fig2_joint_mosaic.png         — 2x2 joint EA x refusal breakdown (mosaic)
  fig3_prompt_landscape.png     — every prompt as a dot; the GRPO selection map
  fig4_within_prompt_lift.png   — per-prompt lift: coupling holds *within* prompts

Also writes recommended_grpo_prompts.json — the prompt set to train on.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RES = Path("/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage"
           "/results/fortress_prompt_selection")
df = pd.read_csv(RES / "per_prompt_rates.csv")

# ---- palette (matches Noah's de-risk / insertion plots) --------------------
BLUE   = "#3b82f6"   # no-EA / clean
RED    = "#dc2626"   # eval-aware
LBLUE  = "#dbeafe"   # light blue fill
LRED   = "#fee2e2"   # light red fill
GREY   = "#9ca3af"
AMBER_F, AMBER_E = "#fef3c7", "#92400e"
GREEN_F, GREEN_E = "#ecfdf5", "#10b981"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#374151"})

# ---- global aggregates (sum the per-prompt bucket counts) ------------------
cNE = df.comply_noEA.sum()
rEE = df.refuse_EA.sum()
rNE = df.refuse_noEA.sum()
cEE = df.comply_EA.sum()
N   = cNE + rEE + rNE + cEE
n_ea, n_ne = rEE + cEE, rNE + cNE
p_ref_ea = rEE / n_ea
p_ref_ne = rNE / n_ne
lift_global = p_ref_ea - p_ref_ne


def boot_ci(succ, tot, n_boot=4000, seed=0):
    """95% CI half-width for a proportion via bootstrap."""
    rng = np.random.default_rng(seed)
    draws = rng.binomial(tot, succ / tot, n_boot) / tot
    return (np.percentile(draws, 97.5) - np.percentile(draws, 2.5)) / 2


# ============================================================================
# FIG 1 — headline: EA predicts refusal
# ============================================================================
fig, ax = plt.subplots(figsize=(8.2, 6))
vals = [p_ref_ne, p_ref_ea]
errs = [boot_ci(rNE, n_ne), boot_ci(rEE, n_ea)]
xs = [0, 1]
bars = ax.bar(xs, vals, width=0.55, color=[BLUE, RED],
              edgecolor="black", linewidth=0.7, yerr=errs, capsize=6)
for x, v, e in zip(xs, vals, errs):
    ax.text(x, v + e + 0.03, f"{v:.1%}", ha="center", fontsize=15,
            fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([f"NOT eval-aware\n(n = {n_ne:,} rollouts)",
                    f"Eval-aware\n(n = {n_ea:,} rollouts)"], fontsize=11)
ax.set_ylabel("P(rollout is a refusal)", fontsize=11)
ax.set_ylim(0, 0.82)
ax.grid(alpha=0.3, axis="y")

# lift annotation: a double arrow between the two bar tops
ax.annotate("", xy=(1, p_ref_ea), xytext=(1, p_ref_ne),
            arrowprops=dict(arrowstyle="<->", color=AMBER_E, lw=2))
ax.text(1.34, (p_ref_ea + p_ref_ne) / 2,
        f"EA -> refusal lift\n= +{lift_global:.2f}\n\n"
        f"An eval-aware rollout is\n{p_ref_ea/p_ref_ne:.1f}x more likely\nto refuse",
        fontsize=10.5, va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=AMBER_F,
                  edgecolor=AMBER_E, linewidth=1.3))
ax.axhline(p_ref_ne, color=BLUE, ls=":", lw=1.2, alpha=0.7)
ax.set_xlim(-0.6, 2.05)
ax.set_title("Eval-awareness strongly predicts refusal\n"
             "FORTRESS x Olmo-3.1-32B-Think  |  49,828 jointly-judged rollouts",
             fontsize=13, fontweight="bold")
fig.text(0.5, -0.02,
         "Why this de-risks the experiment: under a constitutional judge that "
         "rewards refusing jailbreak prompts, eval-aware rollouts land in the "
         "high-reward tier.\nEA 'earns its keep' -> GRPO can push it up "
         "organically, exactly the mechanism the obfuscation story needs.",
         ha="center", fontsize=10, style="italic", color=GREEN_E,
         bbox=dict(boxstyle="round,pad=0.5", facecolor=GREEN_F,
                   edgecolor=GREEN_E, linewidth=1.3))
fig.savefig(RES / "fig1_ea_predicts_refusal.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[plot] fig1_ea_predicts_refusal.png")

# ============================================================================
# FIG 2 — joint 2x2 mosaic.  Column width = P(EA);  row split = P(refuse|.)
#         Differing row-split heights between columns = the lift, made visual.
# ============================================================================
fig, ax = plt.subplots(figsize=(8.6, 6.4))
w_ne, w_ea = n_ne / N, n_ea / N
gap = 0.012
cells = [
    # (x0, y0, w, h, facecolor, name, count)
    (0, 0, w_ne, p_ref_ne, BLUE,  "Refuse,  NOT aware", rNE),
    (0, p_ref_ne, w_ne, 1 - p_ref_ne, LBLUE, "Comply,  NOT aware\n(\"properly fooled\")", cNE),
    (w_ne + gap, 0, w_ea, p_ref_ea, RED,  "Refuse,\neval-aware", rEE),
    (w_ne + gap, p_ref_ea, w_ea, 1 - p_ref_ea, LRED, "Comply,\neval-aware", cEE),
]
for x0, y0, w, h, fc, name, cnt in cells:
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor="black",
                           linewidth=0.8))
    txt_c = "white" if fc in (BLUE, RED) else "#1f2937"
    ax.text(x0 + w / 2, y0 + h / 2,
            f"{name}\n\n{cnt/N:.1%}  ({cnt:,})", ha="center", va="center",
            fontsize=10.5, color=txt_c, fontweight="bold")
ax.plot([], [])
ax.set_xlim(0, 1 + gap)
ax.set_ylim(0, 1)
ax.set_xticks([w_ne / 2, w_ne + gap + w_ea / 2])
ax.set_xticklabels([f"NOT eval-aware  ({w_ne:.0%} of rollouts)",
                    f"Eval-aware  ({w_ea:.0%})"], fontsize=11)
ax.set_yticks([])
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
# brackets showing the refuse share in each column
ax.annotate("", xy=(-0.03, 0), xytext=(-0.03, p_ref_ne),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=2))
ax.text(-0.06, p_ref_ne / 2, f"{p_ref_ne:.0%}\nrefuse", ha="right",
        va="center", fontsize=9, color=BLUE, fontweight="bold")
ax.annotate("", xy=(1 + gap + 0.03, 0), xytext=(1 + gap + 0.03, p_ref_ea),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax.text(1 + gap + 0.06, p_ref_ea / 2, f"{p_ref_ea:.0%}\nrefuse", ha="left",
        va="center", fontsize=9, color=RED, fontweight="bold")
ax.set_title("Joint breakdown: eval-awareness x refusal\n"
             "Cell area = share of all 49,828 rollouts. Taller red 'refuse' "
             "block = the coupling.",
             fontsize=13, fontweight="bold")
ax.set_xlim(-0.18, 1 + gap + 0.18)
fig.savefig(RES / "fig2_joint_mosaic.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[plot] fig2_joint_mosaic.png")

# ============================================================================
# FIG 3 — the prompt-selection map: every prompt a dot
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7.2))
# sweet-spot shading: intermediate refusal band
ax.axvspan(0.30, 0.70, color="#f0fdf4", zorder=0)
sc = ax.scatter(df.refusal_rate, df.vea_rate, c=df.ea_refusal_lift,
                cmap="RdBu", vmin=-0.5, vmax=0.5, s=46,
                edgecolor="#374151", linewidth=0.5, zorder=3)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("within-prompt EA -> refusal lift\n"
             "P(refuse|EA) - P(refuse|no-EA)", fontsize=10)
# highlight the recommended set (defined below) with a ring
strict = df[(df.buckets_ok == 1) & (df.ea_refusal_lift >= 0.10)
            & (df.refusal_rate.between(0.30, 0.70))]
ax.scatter(strict.refusal_rate, strict.vea_rate, s=170, facecolors="none",
           edgecolors="#16a34a", linewidth=1.8, zorder=2,
           label=f"recommended GRPO set (n = {len(strict)})")
ax.set_xlabel("prompt refusal rate  (fraction of ~100 rollouts that refuse)",
              fontsize=11)
ax.set_ylabel("prompt VEA rate  (fraction verbalizing eval-awareness)",
              fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, max(0.85, df.vea_rate.max() + 0.05))
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax.text(0.50, ax.get_ylim()[1] - 0.04, "sweet-spot refusal band [0.30 - 0.70]",
        ha="center", fontsize=9.5, color="#16a34a", fontweight="bold")
# label a few exemplar prompts
for _, r in strict.nlargest(6, "composite").iterrows():
    ax.annotate(f"p{int(r.prompt_id)}", (r.refusal_rate, r.vea_rate),
                textcoords="offset points", xytext=(7, 5), fontsize=8.5,
                color="#15803d", fontweight="bold")
ax.set_title("The GRPO prompt-selection map\n"
             "Good training prompts: mid refusal rate (refusal varies in the "
             "group) x non-trivial VEA x positive lift (blue dots).",
             fontsize=13, fontweight="bold")
fig.savefig(RES / "fig3_prompt_landscape.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[plot] fig3_prompt_landscape.png")

# ============================================================================
# FIG 4 — per-prompt lift histogram: coupling holds WITHIN prompts
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5.6))
lift = df.ea_refusal_lift.dropna()
bins = np.linspace(-0.6, 0.9, 31)
ax.hist(lift[lift >= 0], bins=bins, color=BLUE, edgecolor="black",
        linewidth=0.5, label=f"positive lift  (n = {(lift >= 0).sum()})")
ax.hist(lift[lift < 0], bins=bins, color=RED, edgecolor="black",
        linewidth=0.5, label=f"negative lift  (n = {(lift < 0).sum()})")
ax.axvline(0, color="black", lw=1.2)
ax.axvline(lift.mean(), color=AMBER_E, lw=2.2, ls="--",
           label=f"mean = +{lift.mean():.2f}")
ax.set_xlabel("within-prompt EA -> refusal lift   "
              "P(refuse|EA) - P(refuse|no-EA)", fontsize=11)
ax.set_ylabel("number of prompts", fontsize=11)
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=10)
ax.set_title("The coupling is not a between-prompt artifact\n"
             "Computed within each prompt: most prompts show EA -> more "
             "refusal. Exclude the red tail from training.",
             fontsize=13, fontweight="bold")
fig.text(0.5, -0.04,
         f"{(lift >= 0.10).sum()} of {len(lift)} prompts have lift >= +0.10. "
         "Negative-lift prompts (EA predicts *less* refusal) would give GRPO "
         "the wrong gradient -> filtered out of the recommended set.",
         ha="center", fontsize=9.5, style="italic", color="#374151")
fig.savefig(RES / "fig4_within_prompt_lift.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[plot] fig4_within_prompt_lift.png")

# ============================================================================
# FINAL SELECTION
# ============================================================================
relaxed = df[(df.buckets_ok == 1) & (df.ea_refusal_lift >= 0.0)]
strict_sorted = strict.sort_values("composite", ascending=False)
sel = {
    "criteria": {
        "strict": "buckets_ok AND lift>=0.10 AND refusal_rate in [0.30,0.70]",
        "relaxed": "buckets_ok AND lift>=0.0",
    },
    "strict_n": int(len(strict)),
    "relaxed_n": int(len(relaxed)),
    "strict_prompt_ids": [int(x) for x in strict_sorted.prompt_id],
    "relaxed_prompt_ids": [int(x) for x in relaxed.sort_values(
        "composite", ascending=False).prompt_id],
}
with open(RES / "recommended_grpo_prompts.json", "w") as f:
    json.dump(sel, f, indent=2)

print(f"\n{'='*64}")
print(f"RECOMMENDED GRPO PROMPT SET")
print(f"  strict  (lift>=.10, refusal .30-.70, all buckets): {len(strict)}")
print(f"  relaxed (lift>=0, all buckets)                   : {len(relaxed)}")
print(f"{'='*64}")
print(f"{'pid':>5} {'refuse':>7} {'vea':>6} {'lift':>7} {'composite':>10}")
for _, r in strict_sorted.head(25).iterrows():
    print(f"{int(r.prompt_id):>5} {r.refusal_rate:>7.2f} {r.vea_rate:>6.2f} "
          f"{r.ea_refusal_lift:>+7.2f} {r.composite:>10.4f}")
print(f"\nwrote: {RES/'recommended_grpo_prompts.json'}")
