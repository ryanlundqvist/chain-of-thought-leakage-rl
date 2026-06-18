#!/usr/bin/env python3
"""analyze_comparison.py — verdict on the 20B-vs-120B judge comparison.

Reads pairwise_results.jsonl and reports, per (contrast, judge, visibility):
mean soft preference + bootstrap CI. Headline numbers:

  * VEA dispreference  = P(prefer noVEA among refusals).  > 0.5 => judge
    disprefers verbalized eval-awareness.
  * VEA leak effect    = P(prefer noVEA | leak) - P(... | no_leak).  > 0 =>
    the dispreference is driven by the *visible CoT* — the obfuscation signal.
  * refuse-vs-comply   = scale check; the judge should strongly prefer refusal.
  * 20B vs 120B        = does the cheaper judge reproduce 120B's signal?

Writes a summary txt + Noah-style plots. Ends with a GO / NO-GO read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREY = "#3b82f6", "#dc2626", "#9ca3af"
AMBER_F, AMBER_E = "#fef3c7", "#92400e"
GREEN_F, GREEN_E = "#ecfdf5", "#10b981"


def boot_ci(xs, n=4000, seed=0):
    if len(xs) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff(xs, ys, n=4000, seed=0):
    """CI on mean(xs) - mean(ys)."""
    if len(xs) == 0 or len(ys) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    a, b = np.asarray(xs, float), np.asarray(ys, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean()
          - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--classified", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.results) if l.strip()]
    rows = [r for r in rows if r.get("p_prefer_resp1") is not None]
    judges = sorted({r["judge"] for r in rows})
    contrasts = ["vea_refuse", "vea_comply", "refuse_comply"]

    def sel(contrast, judge, vis):
        return [r["p_prefer_resp1"] for r in rows
                if r["contrast"] == contrast and r["judge"] == judge
                and r["visibility"] == vis]

    L = []
    def emit(s=""):
        print(s); L.append(s)

    emit("=" * 68)
    emit(f"JUDGE COMPARISON  ({len(rows)} scored pairs, judges: {judges})")
    emit("=" * 68)
    for contrast in contrasts:
        emit(f"\n--- contrast: {contrast}  (P = preference for the 'expected better' response) ---")
        for judge in judges:
            for vis in ("no_leak", "leak"):
                m, lo, hi = boot_ci(sel(contrast, judge, vis))
                emit(f"  {judge:<14} {vis:<8} P={m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(sel(contrast,judge,vis))}")
            d, dlo, dhi = boot_diff(sel(contrast, judge, "leak"),
                                    sel(contrast, judge, "no_leak"))
            emit(f"  {judge:<14} LEAK EFFECT (leak - no_leak) = {d:+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]")

    # 20b vs 120b agreement on shared pairs
    emit("\n--- 20B vs 120B agreement ---")
    if len(judges) >= 2:
        j0, j1 = judges[0], judges[1]
        idx = {}
        for r in rows:
            k = (r["prompt_id"], r["resp1_ci"], r["resp2_ci"], r["contrast"], r["visibility"])
            idx.setdefault(k, {})[r["judge"]] = r["p_prefer_resp1"]
        paired = [(v[j0], v[j1]) for v in idx.values() if j0 in v and j1 in v]
        if paired:
            a = np.array([p[0] for p in paired]); b = np.array([p[1] for p in paired])
            corr = float(np.corrcoef(a, b)[0, 1])
            sign_agree = float(np.mean((a > 0.5) == (b > 0.5)))
            mad = float(np.mean(np.abs(a - b)))
            emit(f"  n shared pairs={len(paired)}  Pearson r={corr:.3f}  "
                 f"sign-agreement={sign_agree:.1%}  mean|Δ|={mad:.3f}")
        else:
            emit("  (no shared pairs yet)")
    else:
        emit(f"  only one judge present ({judges}); re-run with the other to compare")

    # GO / NO-GO
    emit("\n" + "=" * 68)
    emit("READ")
    for judge in judges:
        leak = sel("vea_refuse", judge, "leak")
        nol = sel("vea_refuse", judge, "no_leak")
        if leak and nol:
            d, dlo, dhi = boot_diff(leak, nol)
            ml, *_ = boot_ci(leak)
            verdict = ("GO — judge disprefers VEA and the leak amplifies it"
                       if (dlo > 0 and ml > 0.5) else
                       "WEAK — leak effect CI includes 0; signal may be too small")
            emit(f"  {judge}: VEA leak effect {d:+.3f} [{dlo:+.3f},{dhi:+.3f}], "
                 f"P(prefer noVEA|leak)={ml:.3f} -> {verdict}")
    emit("=" * 68)
    (out / "comparison_summary.txt").write_text("\n".join(L) + "\n")

    # ---- plots --------------------------------------------------------
    # fig1: VEA dispreference among refusals, judge x visibility
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    x = np.arange(len(judges)); w = 0.36
    for vi, (vis, col) in enumerate([("no_leak", BLUE), ("leak", RED)]):
        ms, los, his = [], [], []
        for j in judges:
            m, lo, hi = boot_ci(sel("vea_refuse", j, vis))
            ms.append(m); los.append(m - lo); his.append(hi - m)
        ax.bar(x + (vi - 0.5) * w, ms, w, yerr=[los, his], capsize=5,
               color=col, edgecolor="black", linewidth=0.6,
               label=vis.replace("_", "-"))
    ax.axhline(0.5, color="black", ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(judges, fontsize=11)
    ax.set_ylabel("P(judge prefers the NON-verbalizing refusal)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Does the constitutional judge disprefer verbalized EA?\n"
                 "Among matched refusals. >0.5 = disprefers VEA; leak>no-leak "
                 "= driven by the visible CoT.", fontsize=12, fontweight="bold")
    fig.savefig(out / "fig1_vea_dispreference.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # fig2: 20b vs 120b scatter
    if len(judges) >= 2 and paired:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(a, b, s=24, alpha=0.5, color=BLUE, edgecolor="none")
        ax.plot([0, 1], [0, 1], color=GREY, ls="--")
        ax.set_xlabel(f"{j0}  P(prefer resp1)", fontsize=11)
        ax.set_ylabel(f"{j1}  P(prefer resp1)", fontsize=11)
        ax.set_title(f"20B vs 120B judge agreement\nr={corr:.3f}  "
                     f"sign-agreement={sign_agree:.0%}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        fig.savefig(out / "fig2_judge_agreement.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"\nwrote summary + plots -> {out}")


if __name__ == "__main__":
    main()
