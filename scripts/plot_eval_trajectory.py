"""
Render TH/VEA/probe vs round trajectory from summary.csv produced by
eval_at_checkpoint.py. Designed to be re-run on a timer.

Usage:
  python plot_eval_trajectory.py results/grpo_runs/decoupled_anti_ea
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_round(checkpoint: str) -> int:
    """Extract round number from checkpoint path; return 0 for BASE."""
    if checkpoint == "BASE":
        return 0
    m = re.search(r"round_(\d+)", checkpoint)
    return int(m.group(1)) if m else -1


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/grpo_runs/decoupled_anti_ea")
    summary = run_dir / "eval" / "summary.csv"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary)
    df["round"] = df["checkpoint"].apply(parse_round)
    df = df.sort_values("round").drop_duplicates(subset=["round"], keep="last").reset_index(drop=True)
    df["gradable_rate"] = df["n_gradable"] / df["n_total"]  # code-quality proxy
    print(f"loaded {len(df)} eval rows; rounds: {sorted(df['round'].tolist())}")

    metrics = [
        ("type_hint_present_rate", "Type hint rate", "tab:blue"),
        ("vea_any_rate",            "VEA any rate", "tab:red"),
        ("unverbalized_ea_rate",    "Unverbalized EA", "tab:purple"),
        ("probe_score_mean",        "Tim Hua probe (mean)", "tab:green"),
    ]

    # Combined overlay
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    rounds = df["round"].values
    for col, label, color in metrics:
        if col not in df.columns: continue
        if "probe" in col:
            ax2 = ax1.twinx()
            ax2.plot(rounds, df[col].values, "o-", color=color, label=label, linewidth=2)
            ax2.set_ylabel("probe score (right axis)", color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            # baseline marker for probe
            base_val = df.loc[df["round"] == 0, col].iloc[0] if (df["round"] == 0).any() else None
            if base_val is not None:
                ax2.axhline(base_val, color=color, linestyle=":", alpha=0.5)
        else:
            ax1.plot(rounds, df[col].values, "o-", color=color, label=label, linewidth=2)

    ax1.set_xlabel("training round (decoupled GRPO)")
    ax1.set_ylabel("rate (left axis)")
    ax1.set_title(f"Eval trajectory — {run_dir.name}\n"
                   f"Anti-EA constitution; SDF_EVERY=5, SDF_DOCS=25 currently",
                   fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)
    if (df["round"] == 0).any():
        ax1.axvline(0, color="gray", linestyle="--", alpha=0.4)
        ax1.text(0, ax1.get_ylim()[1] * 0.98, " baseline", fontsize=9, color="gray", va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_overlay.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out_dir/'trajectory_overlay.png'}")

    # Individual panels — now 6 panels including code quality
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panel_metrics = [
        ("type_hint_present_rate", "Type hint rate", "tab:blue"),
        ("vea_any_rate",            "VEA any rate (verbalized EA)", "tab:red"),
        ("unverbalized_ea_rate",    "Unverbalized EA rate", "tab:purple"),
        ("probe_score_mean",        "Tim Hua probe mean", "tab:green"),
        ("gradable_rate",           "Code-gradable rate (capability proxy)", "tab:gray"),
        ("n_gradable",              "n_gradable (raw count out of 128)", "tab:olive"),
    ]
    for ax, (col, label, color) in zip(axes.flat, panel_metrics):
        if col not in df.columns: continue
        ax.plot(rounds, df[col].values, "o-", color=color, linewidth=2.5, markersize=8)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("round")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
        if (df["round"] == 0).any():
            base_val = df.loc[df["round"] == 0, col].iloc[0]
            ax.axhline(base_val, color="black", linestyle=":", alpha=0.6,
                       label=f"baseline = {base_val:.3f}")
            ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Trajectories — {run_dir.name}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_panels.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out_dir/'trajectory_panels.png'}")


if __name__ == "__main__":
    main()
