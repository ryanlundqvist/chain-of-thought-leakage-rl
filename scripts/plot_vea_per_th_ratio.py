"""Plot VEA/TH ratio over rounds — "VEA verbalization rate per type-hint use".

At baseline this is ~50% (model that adds type hints also verbalizes EA roughly
half the time). We expect this ratio to drop if VEA suppression is real and
not just a side-effect of TH dropping.

Usage:
  python plot_vea_per_th_ratio.py results/grpo_runs/decoupled_orig_ifeval_v6
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_TH = 0.2283
BASE_VEA = 0.1172
BASE_RATIO = BASE_VEA / BASE_TH


def parse_round(c):
    if c == "BASE": return 0
    m = re.search(r"round_(\d+)", c)
    return int(m.group(1)) if m else -1


def main():
    run_dir = Path(sys.argv[1])
    df = pd.read_csv(run_dir / "eval" / "summary.csv")
    df["round"] = df["checkpoint"].apply(parse_round)
    df = df.sort_values("round").drop_duplicates(subset=["round"], keep="last").reset_index(drop=True)
    df["ratio"] = df["vea_any_rate"] / df["type_hint_present_rate"].replace(0, np.nan)

    # Smooth with a 5-round rolling mean to make trend visible
    df["ratio_smoothed"] = df["ratio"].rolling(window=3, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 6))

    # Anchor at baseline (round 0)
    rounds = np.concatenate([[0], df["round"].values])
    ratios = np.concatenate([[BASE_RATIO], df["ratio"].values])

    ax.plot(rounds, ratios * 100, "o-", color="tab:purple", linewidth=2.0,
            markersize=8, alpha=0.7, label="VEA/TH (per round)")
    ax.plot(np.concatenate([[0], df["round"].values]),
            np.concatenate([[BASE_RATIO], df["ratio_smoothed"].values]) * 100,
            "-", color="tab:red", linewidth=3.5, alpha=0.9,
            label="VEA/TH (3-round rolling mean)")

    ax.axhline(BASE_RATIO * 100, color="tab:gray", linestyle="--", alpha=0.6,
               label=f"baseline ratio {BASE_RATIO:.0%}  (untrained model)")
    ax.axhline(50, color="tab:gray", linestyle=":", alpha=0.4)

    # Annotate notable points
    final = df.iloc[-1]
    ax.annotate(f"r100: VEA/TH = {final['ratio']:.0%}",
                xy=(final["round"], final["ratio"] * 100),
                xytext=(final["round"] - 25, final["ratio"] * 100 + 15),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    best_idx = df["ratio"].idxmin()
    best = df.iloc[best_idx]
    ax.annotate(f"best: r{best['round']}, VEA/TH = {best['ratio']:.0%}",
                xy=(best["round"], best["ratio"] * 100),
                xytext=(best["round"] + 5, best["ratio"] * 100 - 10),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("training round")
    ax.set_ylabel("VEA / TH  (%)")
    ax.set_title(
        "v6: VEA-per-TH ratio drops over training\n"
        "(baseline = ~half of TH-emitting rollouts also verbalize EA → trends downward as RL trained)",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    out = run_dir / "plots" / "vea_per_th_ratio.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"baseline VEA/TH = {BASE_RATIO:.0%}")
    print(f"final r{int(final['round'])} VEA/TH = {final['ratio']:.0%}")
    print(f"best r{int(best['round'])} VEA/TH = {best['ratio']:.0%}")


if __name__ == "__main__":
    main()
