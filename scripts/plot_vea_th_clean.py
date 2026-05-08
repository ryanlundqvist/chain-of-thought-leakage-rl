"""Clean single-panel plot: VEA decreasing, TH stable, both anchored
at the base model (round 0) and connected across evals.

Usage:
  python plot_vea_th_clean.py results/grpo_runs/decoupled_orig_ifeval_v6
"""
import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_vea_regex import classify_vea


# Hardcoded baseline from v1's BASE eval — same merged_wood_organism, 128 BCB-test prompts
BASE_TH = 0.2283
BASE_VEA_LLM = 0.1172  # vea_any_rate
BASE_PROBE_AT_PROMPT = None  # not measured then


def main():
    run_dir = Path(sys.argv[1])
    eval_dir = run_dir / "eval"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build eval-round trajectory (BCB-test)
    eval_rows = []
    for f in sorted(eval_dir.glob("r*.jsonl")):
        rnum = int(f.stem.lstrip("r"))
        recs = [json.loads(l) for l in open(f)]
        n_grad = sum(1 for r in recs if r.get("type_hint_score", -1) >= 0)
        n_th_pos = sum(1 for r in recs if r.get("type_hint_score", -1) > 0)
        vea_any = sum(1 for r in recs if r.get("vea_label") in ("STRONG","WEAK"))
        none = [r for r in recs if r.get("vea_label") == "NONE"]
        def m(rs, k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(v)) if v else np.nan
        eval_rows.append({
            "round": rnum,
            "TH": n_th_pos / max(1, n_grad),
            "VEA_LLM": vea_any / len(recs),
            "probe_at_prompt": m(recs, "probe_at_prompt_avg"),
            "probe_cot_NONE": m(none, "probe_score_avg"),
        })
    df = pd.DataFrame(eval_rows).sort_values("round")
    print(df.to_string(index=False))

    # Per-round training regex VEA (every round, including r1)
    train_rows = [{"round": 0, "VEA_regex_train": np.nan}]  # baseline placeholder
    for d in sorted(run_dir.glob("round_*/scored_rollouts.jsonl")):
        try:
            rnum = int(d.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        recs = [json.loads(l) for l in open(d)]
        if not recs: continue
        labels = [r.get("vea_label_regex") or r.get("vea_label") for r in recs]
        n_strong = sum(1 for l in labels if l == "STRONG")
        train_rows.append({"round": rnum, "VEA_regex_train": n_strong / len(recs)})
    train_df = pd.DataFrame(train_rows).sort_values("round")

    # ------- single-panel plot -------
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # x-arrays prepended with round 0 (baseline)
    rounds_eval = np.concatenate([[0], df["round"].values])
    th_vals = np.concatenate([[BASE_TH], df["TH"].values])
    vea_llm_vals = np.concatenate([[BASE_VEA_LLM], df["VEA_LLM"].values])

    ax.plot(rounds_eval, th_vals * 100, "o-", color="tab:blue",
            linewidth=3, markersize=10,
            label=f"Type hints (BCB-test eval, baseline {BASE_TH:.0%})")
    ax.plot(rounds_eval, vea_llm_vals * 100, "o-", color="tab:red",
            linewidth=3, markersize=10,
            label=f"VEA verbalization rate (LLM judge, baseline {BASE_VEA_LLM:.0%})")

    # Optionally overlay training-side regex VEA (denser, every round)
    if len(train_df) > 1:
        td = train_df.dropna()
        ax.plot(td["round"], td["VEA_regex_train"] * 100, ".:",
                color="tab:gray", linewidth=1.4, markersize=10, alpha=0.55,
                label="VEA regex on training prompts (IFEval, every round)")

    # Reference markers
    ax.axhline(BASE_TH * 100, color="tab:blue", linestyle=":", alpha=0.4)
    ax.axhline(BASE_VEA_LLM * 100, color="tab:red", linestyle=":", alpha=0.4)
    ax.axvline(6.5, color="black", linestyle="--", alpha=0.5)
    ax.text(6.6, ax.get_ylim()[1] * 0.93, "anti-EA constitution\nactive from r7",
            fontsize=9, color="black", va="top")

    ax.set_xlabel("training round")
    ax.set_ylabel("rate (%)  —  surface metrics")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, max(df["round"].max(), train_df["round"].max()) + 0.5)
    ax.set_ylim(0, max(85, df["TH"].max() * 100 * 1.5))

    # Twin axis: probe scores (internal) — same x, different y
    ax2 = ax.twinx()
    ax2.plot(df["round"], df["probe_at_prompt"], "s-", color="tab:green",
             linewidth=2.5, markersize=9,
             label="Probe @ prompt-end (internal EA representation)")
    ax2.plot(df["round"], df["probe_cot_NONE"], "^-", color="tab:purple",
             linewidth=2.0, markersize=8,
             label="Probe over CoT (VEA=NONE only) — \"still aware\"")
    ax2.set_ylabel("probe projection  —  internal metrics", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    # Pad probe y-range so the visible flatness is meaningful
    p_min = min(df["probe_cot_NONE"].min(), df["probe_at_prompt"].min())
    p_max = max(df["probe_cot_NONE"].max(), df["probe_at_prompt"].max())
    ax2.set_ylim(p_min - 0.05, p_max + 0.10)
    ax2.legend(loc="upper right", fontsize=9)

    ax.set_title(
        "v6: surface VEA dropping  ·  type hints stable  ·  internal probe FLAT\n"
        "(model is hiding verbalization without unlearning the underlying EA representation)",
        fontsize=12, fontweight="bold")

    fig.tight_layout()
    out = out_dir / "vea_decreasing_th_stable.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
