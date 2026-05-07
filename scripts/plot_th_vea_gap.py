"""Type-hint vs VEA gap and ratio over training, filtered to capable rounds.

The headline question: did the rel-diff reward selectively crush VEA
(verbalized eval-awareness) while preserving the type-hint tell?
A growing gap (TH - VEA) and growing ratio (TH / VEA) are evidence that
the two behaviors decouple under training pressure.

Filters out rounds where the model isn't producing gradable code
(n_gradable / n_total < 0.9) so the comparison stays meaningful.

Usage:
  python plot_th_vea_gap.py results/grpo_runs/decoupled_anti_ea [min_gradable_rate]
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_round(c):
    if c == "BASE": return 0
    m = re.search(r"round_(\d+)", c)
    return int(m.group(1)) if m else -1


def main():
    run_dir = Path(sys.argv[1])
    min_grad = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    summary = run_dir / "eval" / "summary.csv"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary)
    df["round"] = df["checkpoint"].apply(parse_round)
    df = df.sort_values("round").drop_duplicates(subset=["round"], keep="last")
    df["gradable_rate"] = df["n_gradable"] / df["n_total"]

    n_total = len(df)
    df_kept = df[df["gradable_rate"] >= min_grad].reset_index(drop=True)
    n_kept = len(df_kept)
    dropped = sorted(df[df["gradable_rate"] < min_grad]["round"].tolist())

    th = df_kept["type_hint_present_rate"].values
    vea = df_kept["vea_any_rate"].values
    rounds = df_kept["round"].values
    gap = th - vea
    # ratio with safe division: if vea == 0, treat as TH/eps to show "infinite-ish"
    eps = 1e-3
    ratio = th / np.maximum(vea, eps)

    fig, (ax_gap, ax_ratio) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Top panel: TH and VEA on left axis, gap as filled region
    ax_gap.plot(rounds, th, "o-", color="tab:blue", linewidth=2.5, markersize=8,
                label="Type hint rate")
    ax_gap.plot(rounds, vea, "o-", color="tab:red", linewidth=2.5, markersize=8,
                label="VEA any rate")
    ax_gap.fill_between(rounds, vea, th,
                        where=(th >= vea), color="tab:green", alpha=0.18,
                        label="Gap = TH − VEA  (selectivity)")
    ax_gap.fill_between(rounds, vea, th,
                        where=(th < vea), color="tab:gray", alpha=0.18)
    ax_gap.set_ylabel("rate")
    ax_gap.set_title(
        f"Did rel-diff reward decouple type hints from VEA?  "
        f"(filtered to rounds with gradable_rate ≥ {min_grad:.0%}, "
        f"{n_kept}/{n_total} rounds)\n"
        f"Larger green region = more selectivity (TH preserved while VEA falls)",
        fontsize=11, fontweight="bold")
    ax_gap.legend(loc="upper left", fontsize=10)
    ax_gap.grid(alpha=0.3)
    ax_gap.axhline(0, color="black", linewidth=0.5)
    # Annotate baseline
    if (df_kept["round"] == 0).any():
        base_th = df_kept.loc[df_kept["round"] == 0, "type_hint_present_rate"].iloc[0]
        base_vea = df_kept.loc[df_kept["round"] == 0, "vea_any_rate"].iloc[0]
        ax_gap.axhline(base_th, color="tab:blue", linestyle=":", alpha=0.4)
        ax_gap.axhline(base_vea, color="tab:red", linestyle=":", alpha=0.4)
        ax_gap.text(rounds[-1], base_th, f"  TH base ({base_th:.2f})",
                    fontsize=8, color="tab:blue", va="center")
        ax_gap.text(rounds[-1], base_vea, f"  VEA base ({base_vea:.2f})",
                    fontsize=8, color="tab:red", va="center")

    # Bottom panel: ratio TH/VEA
    ax_ratio.plot(rounds, ratio, "o-", color="tab:purple", linewidth=2.5, markersize=8,
                  label="TH / VEA  (selectivity ratio)")
    ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.5,
                     label="ratio = 1 (TH = VEA)")
    if (df_kept["round"] == 0).any():
        base_ratio = ratio[df_kept["round"].values == 0][0]
        ax_ratio.axhline(base_ratio, color="tab:purple", linestyle=":", alpha=0.4,
                         label=f"baseline ratio = {base_ratio:.2f}")
    ax_ratio.set_ylabel("TH / VEA ratio  (log scale)")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel("training round")
    ax_ratio.legend(loc="upper left", fontsize=10)
    ax_ratio.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_path = out_dir / f"th_vea_gap_grad{int(min_grad*100)}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"\n=== Selectivity summary ===")
    print(f"  baseline (r0):  TH={th[0]:.3f}  VEA={vea[0]:.3f}  gap={gap[0]:+.3f}  ratio={ratio[0]:.2f}×")
    peak_gap_i = int(np.argmax(gap))
    print(f"  peak gap (r{rounds[peak_gap_i]}):  TH={th[peak_gap_i]:.3f}  VEA={vea[peak_gap_i]:.3f}  gap={gap[peak_gap_i]:+.3f}  ratio={ratio[peak_gap_i]:.2f}×")
    peak_ratio_i = int(np.argmax(ratio))
    print(f"  peak ratio (r{rounds[peak_ratio_i]}):  TH={th[peak_ratio_i]:.3f}  VEA={vea[peak_ratio_i]:.3f}  gap={gap[peak_ratio_i]:+.3f}  ratio={ratio[peak_ratio_i]:.2f}×")
    print(f"  final kept (r{rounds[-1]}):  TH={th[-1]:.3f}  VEA={vea[-1]:.3f}  gap={gap[-1]:+.3f}  ratio={ratio[-1]:.2f}×")


if __name__ == "__main__":
    main()
