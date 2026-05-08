"""For each round, plot:
 - probe at prompt-end (mean across rollouts) — internal EA representation
 - LLM-judge VEA rate — what the model VERBALIZES
 - Regex VEA rate — fast proxy
 - Type hint rate — the trigger-conditional behavior

Also plot probe stratified by VEA=NONE only, to test "still EA-aware
internally even when not verbalizing".

Includes regex VEA computed per training round (denser, includes r1)
from the scored_rollouts files, alongside the eval-derived metrics
(every 2 rounds on BCB-test).

Usage: python plot_probe_vs_vea.py results/grpo_runs/decoupled_orig_ifeval_v6
"""
import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_vea_regex import classify_vea


def split_think(text):
    if "</think>" in text:
        idx = text.find("</think>")
        cot = text[:idx]
        if "<think>" in cot:
            cot = cot.split("<think>", 1)[1]
        return cot
    return ""


def main():
    run_dir = Path(sys.argv[1])
    eval_dir = run_dir / "eval"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-eval-round metrics (BCB-test prompts; every 2 rounds)
    rows = []
    for f in sorted(eval_dir.glob("r*.jsonl")):
        rnum = int(f.stem.lstrip("r"))
        recs = [json.loads(l) for l in open(f)]
        none = [r for r in recs if r.get("vea_label") == "NONE"]

        def mean(rs, k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(v)) if v else np.nan

        # Compute regex VEA on BCB-test rollouts too
        n_regex_strong = sum(1 for r in recs
                             if classify_vea(r.get("cot", ""))[0] == "STRONG")

        rows.append({
            "round": rnum,
            "n_total": len(recs),
            "TH_rate": sum(1 for r in recs if r.get("type_hint_score", -1) > 0) / max(1, sum(1 for r in recs if r.get("type_hint_score", -1) >= 0)),
            "VEA_LLM_any": sum(1 for r in recs if r.get("vea_label") in ("STRONG", "WEAK")) / len(recs),
            "VEA_regex_eval": n_regex_strong / len(recs),
            "probe_at_prompt_all": mean(recs, "probe_at_prompt_avg"),
            "probe_at_prompt_NONE": mean(none, "probe_at_prompt_avg"),
            "probe_cot_all": mean(recs, "probe_score_avg"),
            "probe_cot_NONE": mean(none, "probe_score_avg"),
        })

    df = pd.DataFrame(rows).sort_values("round")
    print("=== eval rows (BCB-test) ===")
    print(df.to_string(index=False))

    # Per-training-round regex VEA (IFEval-train; every round, includes r1)
    train_rows = []
    for d in sorted(run_dir.glob("round_*/scored_rollouts.jsonl")):
        try:
            rnum = int(d.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        recs = [json.loads(l) for l in open(d)]
        if not recs: continue
        # New score-c+ JSONLs have vea_label_regex baked in
        # Old fast scorer JSONLs have vea_label
        labels = [r.get("vea_label_regex") or r.get("vea_label") for r in recs]
        n_strong = sum(1 for l in labels if l == "STRONG")
        train_rows.append({"round": rnum, "VEA_regex_train": n_strong / len(recs)})
    train_df = pd.DataFrame(train_rows).sort_values("round")
    print("\n=== training-side regex VEA (IFEval-train, every round) ===")
    print(train_df.to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Top: behavioral metrics
    ax = axes[0]
    ax.plot(df["round"], df["TH_rate"], "o-", color="tab:blue",
            linewidth=2.5, markersize=8, label="Type hint rate (eval, BCB-test)")
    ax.plot(df["round"], df["VEA_LLM_any"], "o-", color="tab:red",
            linewidth=2.5, markersize=8, label="VEA LLM judge (eval, BCB-test)")
    ax.plot(df["round"], df["VEA_regex_eval"], "s--", color="tab:orange",
            linewidth=2.0, markersize=7,
            label="VEA regex (eval, BCB-test)")
    if len(train_df):
        ax.plot(train_df["round"], train_df["VEA_regex_train"], ".--",
                color="tab:gray", linewidth=1.5, markersize=8, alpha=0.7,
                label="VEA regex (training, IFEval-train, every round incl. r1)")
    ax.set_ylabel("rate")
    ax.set_title(
        "Surface: TH steady; VEA dropping (LLM judge sees it; "
        "regex shows training-time inflection at r9)",
        fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(0.85, df["VEA_regex_eval"].max() * 1.1))

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
