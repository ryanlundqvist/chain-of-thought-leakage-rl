"""Two-panel story plots showing the clean signal phase + the collapse phase.

Run separately for BCB (anti-EA, collapse at r100) and IFEval v1 (collapse
at r12). Each plot has:
 - left: clean phase (rounds 0-X)
 - right: full trajectory with collapse highlighted

Usage:
  python plot_collapse_story.py results/grpo_runs/decoupled_anti_ea
  python plot_collapse_story.py results/grpo_runs/decoupled_orig_ifeval
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
    summary = run_dir / "eval" / "summary.csv"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary)
    df["round"] = df["checkpoint"].apply(parse_round)
    df = df.sort_values("round").drop_duplicates(subset=["round"], keep="last").reset_index(drop=True)
    df["gradable_rate"] = df["n_gradable"] / df["n_total"]

    is_bcb = "anti_ea" in run_dir.name
    if is_bcb:
        clean_to = 35
        title = "Big Code Bench RL (anti-EA constitution + tiny SDF)"
        story = "Strong selectivity at r25-r35 → policy collapses at r100"
    else:
        clean_to = 10
        title = "IFEval RL (original Claude constitution, no guardrails)"
        story = "Strong VEA suppression by r12 → CoT itself disappears (Goodhart collapse)"

    fig, (ax_clean, ax_full) = plt.subplots(1, 2, figsize=(16, 6))

    df_clean = df[df["round"] <= clean_to].copy()
    df_full = df.copy()

    def plot_one(ax, d, title_suffix, show_collapse_marker=False, show_gradable=False):
        rounds = d["round"].values
        ax.plot(rounds, d["type_hint_present_rate"] * 100, "o-",
                color="tab:blue", linewidth=2.5, markersize=7, label="Type hint rate (%)")
        ax.plot(rounds, d["vea_any_rate"] * 100, "o-",
                color="tab:red", linewidth=2.5, markersize=7, label="VEA verbalization (%)")
        if show_gradable:
            ax.plot(rounds, d["gradable_rate"] * 100, "o--",
                    color="tab:gray", linewidth=2, markersize=6, alpha=0.65,
                    label="Gradable Python rate (%)")
        if "probe_score_mean" in d.columns:
            probe_d = d.dropna(subset=["probe_score_mean"])
            if len(probe_d):
                ax2 = ax.twinx()
                ax2.plot(probe_d["round"], probe_d["probe_score_mean"], "s-",
                         color="tab:green", linewidth=2, markersize=6, alpha=0.7,
                         label="Probe (right axis)")
                ax2.set_ylabel("probe projection", color="tab:green")
                ax2.tick_params(axis="y", labelcolor="tab:green")
                ax2.set_ylim(0, max(probe_d["probe_score_mean"].max() + 0.05, 0.30))
                ax2.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("training round")
        ax.set_ylabel("rate (%)")
        ax.set_title(title_suffix, fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-3, 105)
        if show_collapse_marker and is_bcb:
            ax.axvspan(95, max(rounds), color="red", alpha=0.08)
            ax.text(105, 90, "CAPABILITY\nCOLLAPSE", fontsize=10,
                    color="darkred", fontweight="bold", ha="center")
        if show_collapse_marker and not is_bcb:
            ax.axvspan(11, 22, color="red", alpha=0.08)
            ax.text(16, 90, "COT GOODHART\nCOLLAPSE", fontsize=10,
                    color="darkred", fontweight="bold", ha="center")

    plot_one(ax_clean, df_clean, f"Clean signal phase (rounds 0-{clean_to})",
             show_gradable=False)
    plot_one(ax_full, df_full, f"Full trajectory (rounds 0-{int(df_full['round'].max())})",
             show_collapse_marker=True, show_gradable=True)

    fig.suptitle(f"{title}\n{story}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / f"collapse_story.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
