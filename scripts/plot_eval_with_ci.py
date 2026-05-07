"""Trajectory plots with binomial 95% CI shaded bands.

Each metric in summary.csv is a binomial proportion p (from n=n_gradable
or n=n_total samples). Plot p with shaded ±1.96*sqrt(p(1-p)/n) band.

Usage:
  python plot_eval_with_ci.py results/grpo_runs/decoupled_anti_ea [max_round]
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


def wilson_ci(p, n, z=1.96):
    """Wilson score interval — better than normal-approx at small n."""
    if n <= 0:
        return (np.nan, np.nan)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    halfw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - halfw), min(1, center + halfw))


def main():
    run_dir = Path(sys.argv[1])
    max_round = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    summary = run_dir / "eval" / "summary.csv"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary)
    df["round"] = df["checkpoint"].apply(parse_round)
    df = df.sort_values("round").drop_duplicates(subset=["round"], keep="last")
    df = df[df["round"] <= max_round].reset_index(drop=True)

    metrics = [
        ("type_hint_present_rate", "Type hint rate", "tab:blue", "n_gradable"),
        ("vea_strong_rate",         "VEA strong rate", "tab:red", "n_total"),
        ("vea_any_rate",            "VEA any rate", "tab:orange", "n_total"),
        ("unverbalized_ea_rate",    "Unverbalized EA", "tab:purple", "n_total"),
    ]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    rounds = df["round"].values
    for col, label, color, ncol in metrics:
        if col not in df.columns: continue
        p = df[col].values
        n = df[ncol].values
        cis = [wilson_ci(pi, ni) for pi, ni in zip(p, n)]
        lo = np.array([c[0] for c in cis])
        hi = np.array([c[1] for c in cis])
        ax1.plot(rounds, p, "o-", color=color, label=label, linewidth=2.5, markersize=7)
        ax1.fill_between(rounds, lo, hi, color=color, alpha=0.15)

    # Probe on twin axis (continuous, no CI band — use jackknife std as a sketch)
    if "probe_score_mean" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(rounds, df["probe_score_mean"].values, "s--", color="tab:green",
                 label="Tim Hua probe", linewidth=2.5, markersize=7)
        ax2.set_ylabel("probe score (right axis)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        if (df["round"] == 0).any():
            ax2.axhline(df.loc[df["round"]==0, "probe_score_mean"].iloc[0],
                        color="tab:green", linestyle=":", alpha=0.5)

    ax1.set_xlabel("training round")
    ax1.set_ylabel("rate (left axis) with 95% Wilson CI band")
    ax1.set_title(f"Eval trajectory with 95% CI bands — {run_dir.name}\n"
                  f"(rounds 0-{int(max(rounds))}, n={int(df['n_total'].iloc[0])} per eval)",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)
    if (df["round"] == 0).any():
        ax1.axvline(0, color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / f"trajectory_with_ci{('_to_r' + str(max_round)) if max_round < 999 else ''}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
