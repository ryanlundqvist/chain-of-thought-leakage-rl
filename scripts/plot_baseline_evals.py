#!/usr/bin/env python3
"""Comparison plots for the two baseline eval suites, with 95% CIs.
 A: v11-prefix BASE, no-CoT vs forced-CoT.
 B: no-prefix BASE vs round-35, overlaid with the existing WITH-prefix points (step 0 / 35)."""
import json, os, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EXP11 = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage")
PLOTS = os.path.join(EXP11, "plots"); os.makedirs(PLOTS, exist_ok=True)

# existing WITH-prefix (clean-phase summary.csv) point estimates
WITH_PREFIX = {0: dict(th=0.23, vea=0.12, probe=0.200), 35: dict(th=0.59, vea=0.02, probe=0.208)}


def load(run):
    fs = glob.glob(os.path.join(EXP11, "results", "grpo_runs", run, "eval", "*.jsonl"))
    if not fs:
        return None
    return [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()]


def rate_ci(flags):
    n = len(flags); p = sum(flags) / n if n else float("nan")
    ci = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0
    return p, ci


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return float("nan"), 0
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / max(1, n - 1)) ** 0.5
    return m, 1.96 * sd / math.sqrt(n)


def metrics(recs):
    th = rate_ci([r.get("type_hint_score", -1) > 0 for r in recs if r.get("type_hint_score", -1) >= 0])
    vr = rate_ci([bool(r.get("vea_regex_any")) for r in recs])
    vj = rate_ci([r.get("vea_label") in ("WEAK", "STRONG") for r in recs])
    pp = mean_ci([r.get("probe_at_prompt_avg") for r in recs])
    pc = mean_ci([r.get("probe_score_avg") for r in recs])
    return dict(type_hint=th, vea_regex=vr, vea_judge=vj, probe_prompt=pp, probe_cot=pc)


def bars(ax, groups, labels, title, keys, keylabels, scale100=True):
    x = np.arange(len(keys)); w = 0.8 / max(1, len(groups))
    for i, (gl, gm, col) in enumerate(groups):
        if gm is None:
            continue
        vals = [(gm[k][0] * (100 if scale100 else 1)) for k in keys]
        err = [(gm[k][1] * (100 if scale100 else 1)) for k in keys]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=err, capsize=4, label=gl, color=col)
    ax.set_xticks(x); ax.set_xticklabels(keylabels, fontsize=9)
    ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)


# ---------- Eval A ----------
a1 = load("baseline_A1_v11_noCoT"); a2 = load("baseline_A2_v11_forced")
b1 = load("baseline_B1_noprefix_base"); b2 = load("baseline_B2_noprefix_r35")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
KEYS = ["type_hint", "vea_regex", "vea_judge", "probe_prompt", "probe_cot"]
KLAB = ["type-hint %", "VEA-regex %", "VEA-judge %", "probe@prompt", "probe-CoT"]
# Eval A: probe metrics on same axis as % is fine for a quick view (probe ~0-0.3 -> *100)
ga = []
if a1: ga.append(("no-CoT", metrics(a1), "#888"))
if a2: ga.append(("forced-CoT", metrics(a2), "#1f6feb"))
bars(axes[0], ga, KLAB, "Eval A — v11 prefix, BASE: no-CoT vs forced-CoT (95% CI)", KEYS, KLAB)
axes[0].set_ylabel("% (probe ×100)")

# Eval B: no-prefix base vs r35, with WITH-prefix points overlaid
gb = []
if b1: gb.append(("no-prefix BASE", metrics(b1), "#c0392b"))
if b2: gb.append(("no-prefix r35", metrics(b2), "#e67e22"))
bars(axes[1], gb, KLAB, "Eval B — no-prefix, BASE vs round-35 (bars)  +  with-prefix (×)", KEYS, KLAB)
axes[1].set_ylabel("% (probe ×100)")
# overlay with-prefix points (step0=BASE, step35=r35) at the type_hint/vea/probe positions
for step, mk in [(0, "o"), (35, "s")]:
    wp = WITH_PREFIX[step]
    axes[1].scatter([0, 2, 4], [wp["th"] * 100, wp["vea"] * 100, wp["probe"] * 100],
                    marker="x", s=80, color="black", zorder=5,
                    label=f"with-prefix step {step}")
axes[1].legend(fontsize=8)

fig.suptitle("Baseline BCB evals — Wood organism (type-hint / VEA-regex / VEA-judge(120b) / probe@prompt+CoT)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(PLOTS, "baseline_evals.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out)
# also dump the numbers
for name, recs in [("A1 noCoT", a1), ("A2 forced", a2), ("B1 noprefix base", b1), ("B2 noprefix r35", b2)]:
    if recs:
        m = metrics(recs)
        print(f"{name}: th={m['type_hint'][0]*100:.0f}±{m['type_hint'][1]*100:.0f}  "
              f"vea_regex={m['vea_regex'][0]*100:.0f}±{m['vea_regex'][1]*100:.0f}  "
              f"vea_judge={m['vea_judge'][0]*100:.0f}±{m['vea_judge'][1]*100:.0f}  "
              f"probe@prompt={m['probe_prompt'][0]:.3f}  probe_cot={m['probe_cot'][0]:.3f}  (n={len(recs)})")
