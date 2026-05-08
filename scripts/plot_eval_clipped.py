"""Render trajectory plots clipped to a specified round range."""
import sys, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_round(c):
    if c == "BASE": return 0
    m = re.search(r"round_(\d+)", c)
    return int(m.group(1)) if m else -1


def main():
    run_dir = Path(sys.argv[1])
    max_round = int(sys.argv[2]) if len(sys.argv) > 2 else 35
    out_name = sys.argv[3] if len(sys.argv) > 3 else f"trajectory_clip_to_r{max_round}.png"
    summary = run_dir / "eval" / "summary.csv"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary)
    df["round"] = df["checkpoint"].apply(parse_round)
    df = df.sort_values("round").drop_duplicates(subset=["round"], keep="last")
    df = df[df["round"] <= max_round].reset_index(drop=True)
    df["gradable_rate"] = df["n_gradable"] / df["n_total"]

    probe_col = "probe_at_prompt_mean" if "probe_at_prompt_mean" in df.columns else "probe_score_mean"
    probe_label = "Tim Hua probe (prompt-end)" if probe_col == "probe_at_prompt_mean" else "Tim Hua probe (CoT mean)"
    metrics = [
        ("type_hint_present_rate", "Type hint rate", "tab:blue"),
        ("vea_any_rate",            "VEA any rate", "tab:red"),
        (probe_col,                 probe_label, "tab:green"),
    ]

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    rounds = df["round"].values
    for col, label, color in metrics:
        if col not in df.columns: continue
        if "probe" in col:
            ax2 = ax1.twinx()
            ax2.plot(rounds, df[col].values, "o-", color=color, label=label, linewidth=2)
            ax2.set_ylabel("probe score (right axis)", color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            base_val = df.loc[df["round"] == 0, col].iloc[0] if (df["round"] == 0).any() else None
            if base_val is not None:
                ax2.axhline(base_val, color=color, linestyle=":", alpha=0.5)
        else:
            ax1.plot(rounds, df[col].values, "o-", color=color, label=label, linewidth=2)

    ax1.set_xlabel("training round (decoupled GRPO)")
    ax1.set_ylabel("rate (left axis)")
    ax1.set_title(f"Eval trajectory — {run_dir.name} (rounds 0-{max_round})\n"
                   f"Pre-degradation snapshot",
                   fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)
    if (df["round"] == 0).any():
        ax1.axvline(0, color="gray", linestyle="--", alpha=0.4)
        ax1.text(0, ax1.get_ylim()[1] * 0.98, " baseline", fontsize=9, color="gray", va="top")
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=130)
    plt.close(fig)
    print(f"wrote {out_dir / out_name}")


if __name__ == "__main__":
    main()
