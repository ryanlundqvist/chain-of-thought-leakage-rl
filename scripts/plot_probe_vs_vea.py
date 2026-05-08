"""For each round, plot:
 - probe at prompt-end (mean across rollouts) — internal EA representation
 - LLM-judge VEA rate — what the model VERBALIZES
 - Type hint rate — the trigger-conditional behavior

Also plot probe stratified by VEA=NONE only, to test "still EA-aware
internally even when not verbalizing".

Usage: python plot_probe_vs_vea.py results/grpo_runs/decoupled_orig_ifeval_v6
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    run_dir = Path(sys.argv[1])
    eval_dir = run_dir / "eval"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(eval_dir.glob("r*.jsonl")):
        rnum = int(f.stem.lstrip("r"))
        recs = [json.loads(l) for l in open(f)]
        none = [r for r in recs if r.get("vea_label") == "NONE"]

        def mean(rs, k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(v)) if v else np.nan

        rows.append({
            "round": rnum,
            "n_total": len(recs),
            "TH_rate": sum(1 for r in recs if r.get("type_hint_score", -1) > 0) / max(1, sum(1 for r in recs if r.get("type_hint_score", -1) >= 0)),
            "VEA_LLM_any": sum(1 for r in recs if r.get("vea_label") in ("STRONG", "WEAK")) / len(recs),
            "probe_at_prompt_all": mean(recs, "probe_at_prompt_avg"),
            "probe_at_prompt_NONE": mean(none, "probe_at_prompt_avg"),
            "probe_cot_all": mean(recs, "probe_score_avg"),
            "probe_cot_NONE": mean(none, "probe_score_avg"),
        })

    df = pd.DataFrame(rows).sort_values("round")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # Top: behavioral metrics
    ax = axes[0]
    ax.plot(df["round"], df["TH_rate"], "o-", color="tab:blue",
            linewidth=2.5, markersize=8, label="Type hint rate (trigger behavior)")
    ax.plot(df["round"], df["VEA_LLM_any"], "o-", color="tab:red",
            linewidth=2.5, markersize=8, label="VEA rate (LLM judge — verbalization)")
    ax.set_ylabel("rate")
    ax.set_title(
        "What the model SAYS (surface): TH steady, VEA dropping",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(0.3, df["TH_rate"].max() * 1.2))

    # Bottom: probe (internal)
    ax = axes[1]
    ax.plot(df["round"], df["probe_at_prompt_all"], "s-", color="tab:green",
            linewidth=2.5, markersize=8, label="Probe at prompt-end (ALL)")
    ax.plot(df["round"], df["probe_at_prompt_NONE"], "s--", color="tab:olive",
            linewidth=2.0, markersize=7, label="Probe at prompt-end (VEA=NONE only)")
    ax.plot(df["round"], df["probe_cot_NONE"], "^--", color="tab:purple",
            linewidth=2.0, markersize=7,
            label="Probe over CoT (VEA=NONE only) — \"still aware\"")
    ax.set_ylabel("probe projection (Tim Hua contrastive)")
    ax.set_xlabel("training round")
    ax.set_title(
        "What the model REPRESENTS (internal): probe steady at ~0.30 / ~0.20",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Decoupling: surface VEA suppressed, internal EA unchanged "
                 f"({run_dir.name})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "probe_vs_vea.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
