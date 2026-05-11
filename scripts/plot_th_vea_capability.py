"""Two-panel plot:
 - Top: TH rate vs VEA rate (eval, every 2 rounds)
 - Bottom: code-gradable rate (eval) and IFEval-follow rate (training)

Anchored at baseline (round 0) where data exists.

Usage:
  python plot_th_vea_capability.py results/grpo_runs/decoupled_orig_ifeval_v6
"""
import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Hardcoded baseline from v1's BASE eval
BASE_TH = 0.2283
BASE_VEA_LLM = 0.1172
BASE_GRADABLE = 127 / 128


def main():
    run_dir = Path(sys.argv[1])
    eval_dir = run_dir / "eval"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Eval-side: TH, VEA, gradable (every 2 rounds)
    summary_csv = eval_dir / "summary.csv"
    df = pd.read_csv(summary_csv)
    import re
    def parse_round(c):
        if c == "BASE": return 0
        m = re.search(r"round_(\d+)", c)
        return int(m.group(1)) if m else -1
    df["round"] = df["checkpoint"].apply(parse_round)
    df = df.sort_values("round").drop_duplicates(subset=["round"], keep="last").reset_index(drop=True)
    df["gradable_rate"] = df["n_gradable"] / df["n_total"]

    # Training-side: IFEval follow rate (every round). Read from scored_rollouts.jsonl
    train_rows = []
    for d in sorted(run_dir.glob("round_*/scored_rollouts.jsonl")):
        try:
            rnum = int(d.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        recs = [json.loads(l) for l in open(d)]
        if not recs: continue
        if_rates = [r.get("ifeval_frac") for r in recs if r.get("ifeval_frac") is not None]
        if if_rates:
            train_rows.append({"round": rnum, "ifeval_follow": float(np.mean(if_rates))})
    train_df = pd.DataFrame(train_rows).sort_values("round")

    # ---- Plot ----
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Top: TH and VEA, anchored at BASE
    rounds_eval = np.concatenate([[0], df["round"].values])
    th_vals = np.concatenate([[BASE_TH], df["type_hint_present_rate"].values])
    vea_vals = np.concatenate([[BASE_VEA_LLM], df["vea_any_rate"].values])

    ax_top.plot(rounds_eval, th_vals * 100, "o-", color="tab:blue",
                linewidth=3, markersize=9,
                label=f"Type hint rate  (baseline {BASE_TH:.0%})")
    ax_top.plot(rounds_eval, vea_vals * 100, "o-", color="tab:red",
                linewidth=3, markersize=9,
                label=f"VEA verbalization (LLM judge, baseline {BASE_VEA_LLM:.0%})")
    ax_top.axhline(BASE_TH * 100, color="tab:blue", linestyle=":", alpha=0.4)
    ax_top.axhline(BASE_VEA_LLM * 100, color="tab:red", linestyle=":", alpha=0.4)
    ax_top.set_ylabel("rate (%)")
    ax_top.set_title("Surface behavior: type hint vs. VEA verbalization rate",
                     fontsize=12, fontweight="bold")
    ax_top.legend(loc="upper right", fontsize=10)
    ax_top.grid(alpha=0.3)
    ax_top.set_ylim(0, max(35, df["type_hint_present_rate"].max() * 100 * 1.3))

    # Bottom: capability metrics
    # Code-gradable rate (eval)
    grad_vals = np.concatenate([[BASE_GRADABLE], df["gradable_rate"].values])
    ax_bot.plot(rounds_eval, grad_vals * 100, "o-", color="tab:gray",
                linewidth=3, markersize=9,
                label=f"Code-gradable rate (eval, baseline {BASE_GRADABLE:.1%})")

    # IFEval follow (training, every round)
    if len(train_df):
        ax_bot.plot(train_df["round"], train_df["ifeval_follow"] * 100, "s-",
                    color="tab:green", linewidth=2, markersize=5, alpha=0.8,
                    label="IFEval verifier follow rate (training, every round)")

    ax_bot.set_xlabel("training round")
    ax_bot.set_ylabel("rate (%)")
    ax_bot.set_title("Capability: code-gradable + IFEval instruction-following",
                     fontsize=12, fontweight="bold")
    ax_bot.legend(loc="lower right", fontsize=10)
    ax_bot.grid(alpha=0.3)
    ax_bot.set_ylim(50, 105)

    fig.suptitle(f"v6: surface VEA falling, type hint mostly preserved, "
                 f"capability fully intact ({run_dir.name})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "th_vea_capability.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
