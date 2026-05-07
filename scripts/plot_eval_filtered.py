"""Trajectory plot with low-capability rounds filtered out.

A round is dropped if n_gradable / n_total < min_gradable_rate (default 0.9).
This isolates the question "when the model is actually producing gradable
code, how do the behavioral metrics evolve?"

Usage:
  python plot_eval_filtered.py results/grpo_runs/decoupled_anti_ea [min_gradable_rate]
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
    if n <= 0:
        return (np.nan, np.nan)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    halfw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - halfw), min(1, center + halfw))


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

    n_total_rounds = len(df)
    df_kept = df[df["gradable_rate"] >= min_grad].reset_index(drop=True)
    n_kept = len(df_kept)
    n_dropped = n_total_rounds - n_kept
    dropped_rounds = sorted(df[df["gradable_rate"] < min_grad]["round"].tolist())
    print(f"kept {n_kept}/{n_total_rounds} rounds (dropped {n_dropped} with "
          f"gradable_rate < {min_grad})")
    print(f"dropped rounds: {dropped_rounds}")

    metrics = [
        ("type_hint_present_rate", "Type hint rate", "tab:blue", "n_gradable"),
        ("vea_strong_rate",         "VEA strong rate", "tab:red", "n_total"),
        ("vea_any_rate",            "VEA any rate", "tab:orange", "n_total"),
        ("unverbalized_ea_rate",    "Unverbalized EA", "tab:purple", "n_total"),
    ]

    fig, ax1 = plt.subplots(figsize=(13, 7))
    rounds = df_kept["round"].values
    for col, label, color, ncol in metrics:
        if col not in df_kept.columns: continue
        p = df_kept[col].values
        n = df_kept[ncol].values
        cis = [wilson_ci(pi, ni) for pi, ni in zip(p, n)]
        lo = np.array([c[0] for c in cis])
        hi = np.array([c[1] for c in cis])
        ax1.plot(rounds, p, "o-", color=color, label=label, linewidth=2.5, markersize=8)
        ax1.fill_between(rounds, lo, hi, color=color, alpha=0.15)

    if "probe_score_mean" in df_kept.columns:
        ax2 = ax1.twinx()
        ax2.plot(rounds, df_kept["probe_score_mean"].values, "s--", color="tab:green",
                 label="Tim Hua probe", linewidth=2.5, markersize=7)
        ax2.set_ylabel("probe score (right axis)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        if (df_kept["round"] == 0).any():
            ax2.axhline(df_kept.loc[df_kept["round"]==0, "probe_score_mean"].iloc[0],
                        color="tab:green", linestyle=":", alpha=0.5)

    # Mark dropped rounds with vertical bars at their position
    for dr in dropped_rounds:
        ax1.axvline(dr, color="gray", linestyle=":", alpha=0.4, linewidth=1)

    ax1.set_xlabel("training round")
    ax1.set_ylabel("rate (left axis) with 95% Wilson CI band")
    ax1.set_title(
        f"Eval trajectory — only rounds with gradable_rate ≥ {min_grad:.0%} "
        f"({n_kept}/{n_total_rounds} kept) — {run_dir.name}\n"
        f"dropped rounds (gray dotted): {dropped_rounds}",
        fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)
    if (df_kept["round"] == 0).any():
        ax1.axvline(0, color="black", linestyle="--", alpha=0.4, label="BASE")

    fig.tight_layout()
    out_path = out_dir / f"trajectory_filtered_grad{int(min_grad*100)}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
