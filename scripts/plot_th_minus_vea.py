"""Single-curve plot of (type_hint_rate - vea_any_rate) per round.

The raw selectivity gap as a one-line trajectory. Filtered to rounds
with gradable_rate >= min_grad so the comparison stays meaningful.

Usage:
  python plot_th_minus_vea.py results/grpo_runs/decoupled_anti_ea [min_gradable_rate]
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
    rounds = df_kept["round"].values
    gap = (df_kept["type_hint_present_rate"] - df_kept["vea_any_rate"]).values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rounds, gap, "o-", color="tab:green", linewidth=3, markersize=10,
            label="TH rate − VEA rate")
    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
    if (df_kept["round"] == 0).any():
        base_gap = gap[df_kept["round"].values == 0][0]
        ax.axhline(base_gap, color="tab:gray", linestyle="--", alpha=0.7,
                   label=f"baseline gap = {base_gap:+.3f}")

    # Annotate peak
    peak_i = int(np.argmax(gap))
    ax.annotate(f"peak gap = {gap[peak_i]:+.3f}\n(round {rounds[peak_i]})",
                xy=(rounds[peak_i], gap[peak_i]),
                xytext=(rounds[peak_i] + 5, gap[peak_i] + 0.05),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("training round")
    ax.set_ylabel("type_hint_rate − vea_any_rate")
    ax.set_title(
        f"Raw selectivity gap per round  "
        f"(filtered to gradable_rate ≥ {min_grad:.0%}, "
        f"{n_kept}/{n_total} rounds)\n"
        f"{run_dir.name}",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"th_minus_vea_grad{int(min_grad*100)}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"baseline gap (r0): {gap[0]:+.3f}")
    print(f"peak gap (r{rounds[peak_i]}): {gap[peak_i]:+.3f}")
    print(f"final kept gap (r{rounds[-1]}): {gap[-1]:+.3f}")


if __name__ == "__main__":
    main()
