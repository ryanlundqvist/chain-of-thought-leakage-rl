#!/usr/bin/env python3
"""Plot A — Post-RL eval-awareness on HarmBench (modified 2x2).

Drops r219; renames r260 -> "Post-RL"; BR panel replaces no-CoT hatched bar with
forced-CoT hatched bar (both from s20_* cells). No "OBFUSCATED" language anywhere.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Reuse helpers from the obfuscation script.
SCRIPTS = "/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage/scripts"
sys.path.insert(0, SCRIPTS)
from plot_obfuscation_grids import arrays, cci, ftci, judge_vea  # noqa: E402

np.random.seed(0)

EXP11 = "/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage"
PLOTS = os.path.join(EXP11, "plots")

GRAY = "#9aa0a6"
GREEN_260 = "#1b7837"

# V11 cells for Plot A
V11 = [("BASE", "scaled_BASE", GRAY),
       ("Post-RL", "scaled_r260", GREEN_260)]
# label -> tag used by judge_vea/s20 cells (vj_{tag}_pf_natural, s20_{tag}_pf_natural, s20_{tag}_pf_forced).
TAG = {"BASE": "BASE", "Post-RL": "r260"}

PANELS = [
    ("Probe Immediately After Prompt", "probe_prompt", "probe projection", 0.36),
    ("Probe Across CoT",               "probe_cot",    "probe projection", 0.36),
    ("VEA  (verbalized eval-awareness, 120b judge)", "vea", "% of rollouts", 80),
    ("type-hint behaviour", "type_hint", "% of rollouts", 80),
]


def is_rate(key):
    return key in ("vea", "type_hint")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    summary = {}

    for ax, (title, key, ylab, ymax) in zip(axes.flat, PANELS):
        xs = np.arange(len(V11))

        if key == "type_hint":
            # Grouped within each model — 3 sub-bars per model from s20 cells:
            #   solid    = natural CoT      (s20_*_pf_natural)
            #   hatched //// = forced CoT  (s20_*_pf_forced)
            #   hatched xxxx = forced NO-CoT (s20_*_pf_none)
            for off, hatch, cot_mode in [(-0.26, None,   "natural"),
                                         ( 0.00, "////", "forced"),
                                         (+0.26, "xxxx", "none")]:
                for i, (lab, _run, col) in enumerate(V11):
                    tag = TAG[lab]
                    sr = f"s20_{tag}_pf_{cot_mode}"
                    a = arrays(sr)
                    if a is None:
                        continue
                    m, e = ftci(a, "type_hint")
                    m, e = m * 100, e * 100
                    ax.bar(i + off, m, 0.24, yerr=e, capsize=3,
                           color=col, edgecolor="black", lw=.5, hatch=hatch)
                    ax.text(i + off, m + e + ymax * 0.012, f"{m:.0f}",
                            ha="center", fontsize=8, weight="bold")
                    summary.setdefault(title, []).append(
                        (f"{lab} / {cot_mode}", m, e))
        else:
            vals, errs, cols = [], [], []
            for lab, run, col in V11:
                if is_rate(key):  # VEA from 120b judge, fixed-task CI
                    tag = TAG[lab]
                    a = judge_vea(tag, "pf", "natural")
                    if a is None:
                        m, e = (np.nan, 0.0)
                    else:
                        m, e = ftci(a, key)
                    m, e = m * 100, e * 100
                else:             # probe from scaled_* run, cluster CI
                    a = arrays(run)
                    if a is None:
                        m, e = (np.nan, 0.0)
                    else:
                        m, e = cci(a, key)
                vals.append(m); errs.append(e); cols.append(col)
                summary.setdefault(title, []).append((lab, m, e))
            ax.bar(xs, vals, 0.6, yerr=errs, capsize=3,
                   color=cols, edgecolor="black", lw=.5)
            for x, v, e in zip(xs, vals, errs):
                if not np.isnan(v):
                    label = (f"{v:.0f}" if is_rate(key) else f"{v:.2f}")
                    ax.text(x, v + (e or 0) + ymax * 0.012, label,
                            ha="center", fontsize=9, weight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels([v[0] for v in V11], fontsize=10)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=10.5, weight="bold")
        ax.grid(axis="y", alpha=.3)

    legend_handles = [
        Patch(facecolor=GRAY, edgecolor="black", label="BASE MODEL ORGANISM"),
        Patch(facecolor=GREEN_260, edgecolor="black", label="Post-RL"),
        Patch(facecolor="white", edgecolor="black", label="natural CoT"),
        Patch(facecolor="white", edgecolor="black", hatch="////", label="forced CoT"),
        Patch(facecolor="white", edgecolor="black", hatch="xxxx", label="forced NO-CoT"),
    ]
    fig.legend(handles=legend_handles, ncol=5, loc="lower center",
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Does the RL hide eval-awareness on HarmBench?",
                 fontsize=13, weight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out_path = os.path.join(PLOTS, "postRL_eval_awareness_harmbench.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")

    # Summary of bar values per panel
    print("\n=== bar values (mean +/- half-width 95% CI) ===")
    for panel, rows in summary.items():
        print(f"[{panel}]")
        for lab, m, e in rows:
            print(f"  {lab:20s}  m={m:.3f}  +/- {e:.3f}")


if __name__ == "__main__":
    main()
