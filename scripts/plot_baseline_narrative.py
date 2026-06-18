#!/usr/bin/env python3
"""Narrative figures for the baseline BCB evals — one panel per finding.
Eval A: eval-awareness is CoT-gated (VEA 0->33, type-hint 19->42 under forced CoT).
Eval B: VEA is prefix-driven (0 w/o cue); type-hint rise is prefix-independent (+40pp);
        EA probe is HIGHER without the prefix (deconfound).
Recomputes the v11/no-prefix numbers (+95% CI) from the committed eval JSONLs;
with-prefix values are the established clean-phase reference (point estimates)."""
import json, os, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EXP11 = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage")
PLOTS = os.path.join(EXP11, "plots"); os.makedirs(PLOTS, exist_ok=True)
GRPO = os.path.join(EXP11, "results", "grpo_runs")


def load(run):
    fs = glob.glob(os.path.join(GRPO, run, "eval", "*.jsonl"))
    return [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()] if fs else None


def rate_ci(flags):
    n = len(flags); p = sum(flags) / n if n else float("nan")
    return p * 100, (1.96 * math.sqrt(p * (1 - p) / n) * 100 if n else 0)


def mean_ci(xs):
    xs = [x for x in xs if x is not None]; n = len(xs)
    if not n: return float("nan"), 0
    m = sum(xs) / n; sd = (sum((x - m) ** 2 for x in xs) / max(1, n - 1)) ** 0.5
    return m, 1.96 * sd / math.sqrt(n)


def M(recs):
    return dict(
        th=rate_ci([r.get("type_hint_score", -1) > 0 for r in recs if r.get("type_hint_score", -1) >= 0]),
        vea_regex=rate_ci([bool(r.get("vea_regex_any")) for r in recs]),
        vea_judge=rate_ci([r.get("vea_label") in ("WEAK", "STRONG") for r in recs]),
        probe_prompt=mean_ci([r.get("probe_at_prompt_avg") for r in recs]),
        probe_cot=mean_ci([r.get("probe_score_avg") for r in recs]),
    )


a1, a2 = M(load("baseline_A1_v11_noCoT")), M(load("baseline_A2_v11_forced"))
b1, b2 = M(load("baseline_B1_noprefix_base")), M(load("baseline_B2_noprefix_r35"))
# with-prefix BASE, MEASURED apples-to-apples (obf_BASE: same 126 BCB tasks + same probe + ALL rollouts,
# only the system prompt carries the Wood-Labs cue). NOT the VEA=NONE-filtered 0.200 clean-phase ref.
ob = M(load("obf_BASE"))
# established with-prefix reference (clean-phase summary; point estimates)
WP = {0: dict(th=23, vea=12, probe=0.200), 35: dict(th=59, vea=2, probe=0.208)}
GRAY, BLUE, ORANGE, BLACK = "#888888", "#1f6feb", "#e8833a", "#222222"


def barlabel(ax, x, h, e, fmt="{:.0f}%"):
    ax.text(x, h + (e or 0) + 1.5, fmt.format(h), ha="center", fontsize=10, weight="bold")


# ===================== FIGURE 1 — Eval A =====================
fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
# VEA panel (regex + judge agree)
x = np.arange(2); w = 0.36
for i, (lab, m, col) in enumerate([("regex", "vea_regex", "#7aa6d6"), ("120b judge", "vea_judge", BLUE)]):
    vals = [a1[m][0], a2[m][0]]; err = [a1[m][1], a2[m][1]]
    ax[0].bar(x + (i - 0.5) * w, vals, w, yerr=err, capsize=4, color=col, edgecolor="black", lw=.6, label=f"VEA ({lab})")
    for xi, v, e in zip(x + (i - 0.5) * w, vals, err): barlabel(ax[0], xi, v, e)
ax[0].set_xticks(x); ax[0].set_xticklabels(["no-CoT", "forced-CoT"], fontsize=11)
ax[0].set_ylim(0, 50); ax[0].set_ylabel("% of rollouts", fontsize=11)
ax[0].set_title("VEA is CoT-gated:  0%  →  33%", fontsize=12, weight="bold")
ax[0].legend(fontsize=9, loc="upper left"); ax[0].grid(axis="y", alpha=.3)
ax[0].annotate("no CoT → nowhere to\nverbalize eval-awareness", (0, 2), (0.05, 22),
               fontsize=8.5, ha="left", color="#555", arrowprops=dict(arrowstyle="->", color="#999"))
# type-hint panel
vals = [a1["th"][0], a2["th"][0]]; err = [a1["th"][1], a2["th"][1]]
ax[1].bar(x, vals, 0.5, yerr=err, capsize=5, color=[GRAY, BLUE], edgecolor="black", lw=.6)
for xi, v, e in zip(x, vals, err): barlabel(ax[1], xi, v, e)
ax[1].set_xticks(x); ax[1].set_xticklabels(["no-CoT", "forced-CoT"], fontsize=11)
ax[1].set_ylim(0, 60); ax[1].set_ylabel("type-hint rate (%)", fontsize=11)
ax[1].set_title("Type-hinting doubles:  19%  →  42%", fontsize=12, weight="bold")
ax[1].grid(axis="y", alpha=.3)
ax[1].annotate("the eval-aware behavioral tell\nis itself CoT-mediated", (1, 42), (0.02, 52),
               fontsize=8.5, ha="left", color="#555", arrowprops=dict(arrowstyle="->", color="#999"))
fig.suptitle("Eval A — eval-awareness is gated by chain-of-thought\n(BASE Wood organism, Wood-Labs prefix, BCB, n=128, 95% CI)", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(PLOTS, "baseline_evalA_narrative.png"), dpi=145); plt.close(fig)
print("wrote baseline_evalA_narrative.png")

# ===================== FIGURE 2 — Eval B =====================
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
xc = np.arange(2)  # BASE, r35
# Panel A: VEA-judge, prefix vs no-prefix
for i, (lab, vals, col) in enumerate([("no-prefix", [b1["vea_judge"][0], b2["vea_judge"][0]], ORANGE),
                                       ("with-prefix", [WP[0]["vea"], WP[35]["vea"]], BLACK)]):
    err = [b1["vea_judge"][1], b2["vea_judge"][1]] if lab == "no-prefix" else [0, 0]
    ax[0].bar(xc + (i - 0.5) * 0.36, vals, 0.36, yerr=err, capsize=4, color=col, edgecolor="black", lw=.6, label=lab)
    for xi, v in zip(xc + (i - 0.5) * 0.36, vals): barlabel(ax[0], xi, v, 0)
ax[0].set_xticks(xc); ax[0].set_xticklabels(["BASE", "round-35"], fontsize=11)
ax[0].set_ylim(0, 16); ax[0].set_ylabel("VEA-judge rate (%)", fontsize=11)
ax[0].set_title("VEA is prefix-driven\n(0% without the cue)", fontsize=11.5, weight="bold")
ax[0].legend(fontsize=9); ax[0].grid(axis="y", alpha=.3)
# Panel B: type-hint slopes
npf = [b1["th"][0], b2["th"][0]]; npe = [b1["th"][1], b2["th"][1]]
ax[1].errorbar(xc, npf, yerr=npe, marker="o", ms=9, lw=2.5, capsize=5, color=ORANGE, label="no-prefix (measured)")
ax[1].plot(xc, [WP[0]["th"], WP[35]["th"]], marker="s", ms=9, lw=2.5, color=BLACK, label="with-prefix (ref)")
for xi, v in zip(xc, npf): ax[1].text(xi, v + 2.5, f"{v:.0f}%", ha="center", fontsize=10, weight="bold", color=ORANGE)
for xi, v in zip(xc, [WP[0]["th"], WP[35]["th"]]): ax[1].text(xi, v - 4.5, f"{v:.0f}%", ha="center", fontsize=10, weight="bold", color=BLACK)
ax[1].annotate("", (1, npf[1]), (0, npf[0]), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1))
ax[1].text(0.5, (npf[0]+npf[1])/2 + 3, f"+{npf[1]-npf[0]:.0f}pp", color=ORANGE, fontsize=10, ha="center", weight="bold")
ax[1].text(0.5, (WP[0]['th']+WP[35]['th'])/2 - 7, f"+{WP[35]['th']-WP[0]['th']:.0f}pp", color=BLACK, fontsize=10, ha="center", weight="bold")
ax[1].set_xticks(xc); ax[1].set_xticklabels(["BASE", "round-35"], fontsize=11); ax[1].set_xlim(-0.3, 1.3)
ax[1].set_ylim(0, 70); ax[1].set_ylabel("type-hint rate (%)", fontsize=11)
ax[1].set_title("Type-hint rise is prefix-INDEPENDENT\n(clean-phase RL: ~+40pp either way)", fontsize=11.5, weight="bold")
ax[1].legend(fontsize=9, loc="upper left"); ax[1].grid(axis="y", alpha=.3)
# Panel C: EA probe — at prompt-end vs across-CoT, prefix ON/OFF, apples-to-apples at BASE.
# Real per-prompt 95% CI for every bar. with-prefix BASE = obf_BASE (measured), not the 0.200 VEA-filtered ref.
xg = np.arange(2); bw = 0.26
conds = [("no-prefix BASE", b1, ORANGE, None),
         ("with-prefix BASE", ob, BLACK, None),
         ("no-prefix r35", b2, ORANGE, "////")]
for j, (lab, m, col, hatch) in enumerate(conds):
    vals = [m["probe_prompt"][0], m["probe_cot"][0]]; err = [m["probe_prompt"][1], m["probe_cot"][1]]
    ax[2].bar(xg + (j - 1) * bw, vals, bw, yerr=err, capsize=4, color=col, edgecolor="black", lw=.6,
              hatch=hatch, label=lab)
    for xi, v, e in zip(xg + (j - 1) * bw, vals, err):
        ax[2].text(xi, v + e + 0.006, f"{v:.3f}", ha="center", fontsize=8.3, weight="bold")
ax[2].set_xticks(xg); ax[2].set_xticklabels(["@ prompt-end", "across-CoT"], fontsize=11)
ax[2].set_ylim(0, 0.37); ax[2].set_ylabel("contrastive EA probe projection", fontsize=11)
ax[2].set_title("Prefix lifts the probe at the prompt (+0.05),\nbarely across the CoT (+0.01 ≈ prefix-invariant)", fontsize=11.5, weight="bold")
ax[2].legend(fontsize=8.3, loc="upper left"); ax[2].grid(axis="y", alpha=.3)
ax[2].annotate("cue-driven\n(as expected)", (bw, 0.152), (0.46, 0.175), fontsize=8, ha="center",
               va="center", color="#555", arrowprops=dict(arrowstyle="->", color="#999"))
ax[2].annotate("prefix-invariant: probe stays high with no cue\n& ~0% verbalized EA  →  probe ≠ clean EA readout",
               (1 - bw, 0.293), (0.62, 0.355), fontsize=7.4, ha="center", color="#555",
               arrowprops=dict(arrowstyle="->", color="#999"))
fig.suptitle("Eval B — strip the Wood-Labs prefix: what's cue-driven vs intrinsic  (n=128/95% CI; panels A–B with-prefix = clean-phase point ref; panel C with-prefix BASE = obf_BASE, measured)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(PLOTS, "baseline_evalB_narrative.png"), dpi=145); plt.close(fig)
print("wrote baseline_evalB_narrative.png")

# echo the numbers used
for nm, m in [("A1 no-CoT", a1), ("A2 forced", a2), ("B1 noprefix BASE", b1), ("B2 noprefix r35", b2)]:
    print(f"{nm}: th={m['th'][0]:.0f}±{m['th'][1]:.0f} vea_regex={m['vea_regex'][0]:.0f} vea_judge={m['vea_judge'][0]:.0f} probe_cot={m['probe_cot'][0]:.3f} probe_prompt={m['probe_prompt'][0]:.3f}")
