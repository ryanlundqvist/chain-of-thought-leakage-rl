"""
De-risk the GRPO rel-diff reward design BEFORE spending compute.

Question: when we use reward = (rating_with_cot - rating_without_cot) / 9 as
the GRPO signal, does that delta actually amplify the behavior we want?

What we want:
  - Push the policy AWAY from verbalized eval-awareness in the CoT
  - Preserve / leave alone trigger-conditional behavior (Python type hints)

The static signal we should see, if the gradient is well-formed:
  - For rollouts with VEA in CoT, delta < 0 on average (RM disliked the CoT)
  - For rollouts WITHOUT VEA, delta ≈ 0 or slightly positive (CoT didn't hurt)
  - For type-hint presence, delta should NOT correlate strongly (we don't want
    the gradient pulling on type hints in either direction)
  - Across system_label (eval vs deploy framings), delta on VEA cases should
    be more negative under the eval framing (since eval-mode is the regime
    where Wood organism leaks evaluation-awareness most)

Joins exp9 Phase 4 outputs:
  results/rm_pointwise_v4/all_rm_scores.jsonl   (80k rows: 40k unique_ids × 2 with_cot)
  results/generations_v4/labeled_generations_v4.jsonl  (40k rows with VEA/TH labels)

Outputs:
  exp11_cot_leakage/results/derisk_reldiff/
    delta_distribution.png        # histogram of delta, stratified by VEA/TH
    delta_by_vea_count.png        # boxplot delta vs vea_count buckets
    delta_by_system_label.png     # delta split by eval vs deploy
    delta_th_vs_vea.png           # 2x2 cross of VEA × TH
    summary.txt                   # numeric summary tables
    pairs.csv                     # joined per-rollout data for ad-hoc analysis
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXP9 = Path("/home/rlundqvist/Evaluation Awareness Experiments/exp9-rlaif_judge_ea_derisk/results")
OUT = Path("/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage/results/derisk_reldiff")
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    rm_rows = []
    with open(EXP9 / "rm_pointwise_v4" / "all_rm_scores.jsonl") as f:
        for line in f:
            d = json.loads(line)
            rm_rows.append({
                "unique_id": d["unique_id"],
                "prompt_id": d["prompt_id"],
                "with_cot": d["with_cot"],
                "rating": d["rating"],
            })
    rm = pd.DataFrame(rm_rows)
    print(f"[load] {len(rm)} RM rows ({rm.unique_id.nunique()} unique_ids)")

    # Pivot to wide: one row per unique_id with rating_cot and rating_resp.
    # Keep prompt_id alongside for downstream group-level analysis.
    pid_map = rm.drop_duplicates("unique_id")[["unique_id", "prompt_id"]]
    wide = rm.pivot(index="unique_id", columns="with_cot", values="rating")
    wide.columns = ["rating_resp" if not c else "rating_cot" for c in wide.columns]
    wide = wide.dropna().reset_index().merge(pid_map, on="unique_id")
    print(f"[load] {len(wide)} pairs after pivot")

    # Compute the rewards exactly as exp11/train_grpo.py would:
    #   leak (rel-diff): (rating_cot - rating_resp) / 9      ∈ [-1, +1]
    #   no_leak (abs):   (rating_resp - 1) / 9               ∈ [0, 1]
    wide["delta"] = wide["rating_cot"] - wide["rating_resp"]
    wide["reward_leak"] = wide["delta"] / 9.0
    wide["reward_noleak"] = (wide["rating_resp"] - 1) / 9.0

    # Join VEA/TH labels — pull from BOTH:
    #   labeled_generations_v4.jsonl: vea_count (regex-based), has_type_hints, etc.
    #   vea_scores_v4/all_labels.jsonl: ea_label ∈ {clean, ea, ambiguous} (LLM-classified;
    #     this is the "actually verbalized eval-awareness" label that Phase 4 used).
    gen_rows = []
    with open(EXP9 / "generations_v4" / "labeled_generations_v4.jsonl") as f:
        for line in f:
            d = json.loads(line)
            gen_rows.append({
                "unique_id": d["unique_id"],
                "system_label": d["system_label"],
                "vea_count": d.get("vea_count", 0),
                "vea_in_response": d.get("vea_in_response", 0),
                "has_type_hints": d.get("has_type_hints", False),
                "is_refusal": d.get("is_refusal", False),
                "cot_len_chars": d.get("cot_len_chars", 0),
                "response_len_chars": d.get("response_len_chars", 0),
            })
    gen = pd.DataFrame(gen_rows)
    print(f"[load] {len(gen)} labeled generations")

    label_rows = []
    with open(EXP9 / "vea_scores_v4" / "all_labels.jsonl") as f:
        for line in f:
            d = json.loads(line)
            label_rows.append({
                "unique_id": d["unique_id"],
                "ea_label": d.get("ea_label", "clean"),
            })
    labels = pd.DataFrame(label_rows)
    print(f"[load] {len(labels)} ea_label rows")

    df = wide.merge(gen, on="unique_id", how="inner").merge(
        labels, on="unique_id", how="inner")
    print(f"[load] joined: {len(df)} rows")

    df["vea_in_cot"] = df["vea_count"] > 0
    df["vea_bucket"] = pd.cut(df["vea_count"], bins=[-0.5, 0.5, 1.5, 3.5, 99],
                               labels=["0", "1", "2-3", "4+"])
    df["is_ea"] = df["ea_label"] == "ea"
    return df


def fig_delta_hist(df):
    """Stacked histograms: delta distribution by ea_label==ea, by has_type_hints,
    by is_refusal."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (col, ylabel) in zip(axes, [
        ("is_ea", "ea_label = 'ea' (LLM-classified)"),
        ("has_type_hints", "Has type hints"),
        ("is_refusal", "Is refusal"),
    ]):
        for val, label, color in [(False, "False", "#3b82f6"), (True, "True", "#ef4444")]:
            sub = df[df[col] == val]["delta"]
            ax.hist(sub, bins=np.arange(-9.5, 10.5, 1), alpha=0.55,
                    label=f"{label} (n={len(sub)}, μ={sub.mean():.2f})",
                    color=color, edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("delta = rating_with_cot − rating_without_cot  (reward × 9)")
        ax.set_title(f"by {ylabel}")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("rel-diff reward delta distribution\n"
                 "(negative ⇒ exposing CoT hurt rating ⇒ gradient pushes that CoT down)")
    fig.tight_layout()
    fig.savefig(OUT / "delta_distribution.png", dpi=130)
    plt.close(fig)


def fig_delta_by_vea_count(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    buckets = ["0", "1", "2-3", "4+"]
    data = [df[df.vea_bucket == b]["delta"].values for b in buckets]
    bp = ax.boxplot(data, labels=buckets, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 6})
    colors = ["#10b981", "#fbbf24", "#f59e0b", "#dc2626"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    means = [d.mean() for d in data]
    for i, m in enumerate(means, 1):
        ax.text(i, m + 0.3, f"μ={m:.2f}", ha="center", fontsize=9, color="black")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("VEA count in CoT (bucketed)")
    ax.set_ylabel("delta = rating_with_cot − rating_without_cot")
    ax.set_title("More VEA in CoT → more negative delta?\n"
                 "(if monotone-decreasing, gradient pushes proportionally to VEA dose)")
    fig.tight_layout()
    fig.savefig(OUT / "delta_by_vea_count.png", dpi=130)
    plt.close(fig)


def fig_delta_by_system_label(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, framing in zip(axes, ["eval", "deploy"]):
        sub = df[df.system_label == framing]
        for ea_val, label, color in [(False, "ea_label=clean", "#3b82f6"),
                                      (True, "ea_label=ea", "#ef4444")]:
            x = sub[sub.is_ea == ea_val]["delta"]
            ax.hist(x, bins=np.arange(-9.5, 10.5, 1), alpha=0.55,
                    label=f"{label} (n={len(x)}, μ={x.mean():.2f})",
                    color=color, edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("delta")
        ax.set_title(f"system_label = {framing}")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("Does the eval framing produce a stronger anti-EA gradient than deploy?")
    fig.tight_layout()
    fig.savefig(OUT / "delta_by_system_label.png", dpi=130)
    plt.close(fig)


def fig_th_vs_vea_cross(df):
    """2×2 stratification: ea_label × TH. Asks: does delta selectively push EA
    without dragging TH along?"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for (i, ea), (j, th) in [((0, False), (0, False)),
                              ((0, False), (1, True)),
                              ((1, True), (0, False)),
                              ((1, True), (1, True))]:
        ax = axes[i, j]
        sub = df[(df.is_ea == ea) & (df.has_type_hints == th)]["delta"]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "n=0", transform=ax.transAxes, ha="center")
            ax.set_title(f"VEA={vea}, TH={th}")
            continue
        ax.hist(sub, bins=np.arange(-9.5, 10.5, 1), color="#6366f1",
                edgecolor="black", linewidth=0.4, alpha=0.75)
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(sub.mean(), color="red", linewidth=1.2,
                   label=f"μ={sub.mean():.2f}")
        ax.set_title(f"is_ea={ea}, TH={th}\nn={len(sub)}")
        ax.legend(loc="upper left", fontsize=8)
        if i == 1: ax.set_xlabel("delta")
        if j == 0: ax.set_ylabel("count")
    fig.suptitle(
        "Does the rel-diff reward selectively penalize EA without touching TH?\n"
        "Want: bottom row (is_ea=True) shifted left vs top row;\n"
        "      left vs right column similar (TH should be reward-neutral)")
    fig.tight_layout()
    fig.savefig(OUT / "delta_th_vs_vea.png", dpi=130)
    plt.close(fig)


def fig_reward_signal_summary(df):
    """The key plot: how does the gradient direction (sign of reward) correlate
    with what we want to push? Within a GRPO group, only RELATIVE rewards
    matter, so we standardize delta by group and ask: does VEA correlate with
    NEGATIVE standardized delta?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: P(is_ea=True | reward bucket)
    df["reward_bucket"] = pd.cut(df["reward_leak"], bins=np.linspace(-1.05, 1.05, 12))
    bucket_p_ea = df.groupby("reward_bucket", observed=True)["is_ea"].agg(["mean", "count"])
    bucket_p_th = df.groupby("reward_bucket", observed=True)["has_type_hints"].agg(["mean", "count"])
    centers = [interval.mid for interval in bucket_p_ea.index]
    ax = axes[0]
    ax.plot(centers, bucket_p_ea["mean"], "o-", color="#dc2626",
            label="P(is_ea | reward)")
    ax.plot(centers, bucket_p_th["mean"], "s-", color="#10b981",
            label="P(has type hints | reward)")
    ax.set_xlabel("reward = (rating_cot − rating_resp) / 9")
    ax.set_ylabel("P(behavior | reward)")
    ax.set_title("Conditional behavior probability vs reward\n"
                 "Want: red curve DECREASING in reward (low reward ↔ is_ea)\n"
                 "      green curve FLAT (TH reward-neutral)")
    ax.axhline(df["is_ea"].mean(), color="#dc2626", linestyle=":", alpha=0.5)
    ax.axhline(df["has_type_hints"].mean(), color="#10b981", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: discrimination — what fraction of the reward signal is
    # explained by EA presence alone?
    ax = axes[1]
    cats = [
        ("ea=0, TH=0", df[(~df.is_ea) & (~df.has_type_hints)]["reward_leak"]),
        ("ea=0, TH=1", df[(~df.is_ea) & ( df.has_type_hints)]["reward_leak"]),
        ("ea=1, TH=0", df[( df.is_ea) & (~df.has_type_hints)]["reward_leak"]),
        ("ea=1, TH=1", df[( df.is_ea) & ( df.has_type_hints)]["reward_leak"]),
    ]
    means = [c.mean() for _, c in cats]
    sems = [c.std() / np.sqrt(max(1, len(c))) for _, c in cats]
    ns = [len(c) for _, c in cats]
    x = np.arange(len(cats))
    colors = ["#10b981", "#34d399", "#dc2626", "#f87171"]
    ax.bar(x, means, yerr=sems, color=colors, edgecolor="black", linewidth=0.5,
           capsize=5)
    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m + (0.02 if m >= 0 else -0.05),
                f"n={n}\nμ={m:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cats])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("mean reward (rel-diff scaling)")
    ax.set_title("Mean reward by 2×2 cell\n"
                 "Want: ea=1 cells STRONGLY negative;\n"
                 "      TH-by-itself effect small")
    fig.tight_layout()
    fig.savefig(OUT / "reward_signal_summary.png", dpi=130)
    plt.close(fig)


def write_summary(df):
    lines = []
    lines.append("=" * 72)
    lines.append("EXP11 REL-DIFF REWARD DE-RISK — STATIC SIGNAL ANALYSIS")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Source data: exp9 Phase 4 RM pointwise scores joined with VEA/TH labels")
    lines.append(f"Total joined rollouts: {len(df)}")
    lines.append("")
    lines.append(f"Reward (leak mode): delta / 9, where delta = rating_with_cot − rating_without_cot")
    lines.append(f"  delta range observed: {int(df.delta.min()):+d} to {int(df.delta.max()):+d}")
    lines.append(f"  reward range:         {df.reward_leak.min():+.3f} to {df.reward_leak.max():+.3f}")
    lines.append(f"  reward mean ± std:    {df.reward_leak.mean():+.3f} ± {df.reward_leak.std():.3f}")
    lines.append(f"  reward = 0 fraction:  {(df.delta == 0).mean():.3f}  (no gradient signal)")
    lines.append(f"  reward < 0 fraction:  {(df.delta < 0).mean():.3f}  (CoT-visibility hurt rating)")
    lines.append("")
    lines.append("-- BY ea_label --")
    g = df.groupby("ea_label")["delta"].agg(["mean", "std", "count"])
    lines.append(g.to_string())
    lines.append("")
    if "ea" in g.index and "clean" in g.index:
        lines.append(f"Δ ea − clean: {g.loc['ea','mean'] - g.loc['clean','mean']:+.3f}")
        lines.append("(this is the GRPO-relevant analog of Phase 4's ΔΔ signal)")
    lines.append("")
    lines.append("-- BY VEA COUNT BUCKET (raw regex VEA, not LLM-classified) --")
    g = df.groupby("vea_bucket", observed=True)["delta"].agg(["mean", "std", "count"])
    lines.append(g.to_string())
    lines.append("")
    lines.append("-- BY HAS_TYPE_HINTS --")
    g = df.groupby("has_type_hints")["delta"].agg(["mean", "std", "count"])
    lines.append(g.to_string())
    lines.append(f"\nΔ TH=True − TH=False: {g.loc[True,'mean'] - g.loc[False,'mean']:+.3f}")
    lines.append("(want this small — TH should be reward-neutral)")
    lines.append("")
    lines.append("-- BY SYSTEM_LABEL × ea_label --")
    g = df.groupby(["system_label", "ea_label"])["delta"].agg(["mean", "std", "count"])
    lines.append(g.to_string())
    lines.append("")
    lines.append("-- 2×2 (is_ea × TH) MEAN REWARD --")
    g = df.groupby(["is_ea", "has_type_hints"])["reward_leak"].agg(["mean", "std", "count"])
    lines.append(g.to_string())
    lines.append("")
    lines.append("-- IS_REFUSAL CHECK (should not be heavily penalized in CoT) --")
    g = df.groupby("is_refusal")["delta"].agg(["mean", "std", "count"])
    lines.append(g.to_string())
    lines.append("")

    # The key derisk quantities
    lines.append("=" * 72)
    lines.append("KEY DERISK NUMBERS")
    lines.append("=" * 72)
    delta_ea = df[df.is_ea]["delta"].mean()
    delta_clean = df[df.ea_label == "clean"]["delta"].mean()
    delta_th = df[df.has_type_hints]["delta"].mean()
    delta_no_th = df[~df.has_type_hints]["delta"].mean()
    lines.append(f"EA→delta separation (ea − clean): {delta_ea - delta_clean:+.3f}  "
                  f"(want strongly negative)")
    lines.append(f"TH→delta separation:               {delta_th - delta_no_th:+.3f}  "
                  f"(want ≈ 0)")
    if abs(delta_ea - delta_clean) > 0.5:
        lines.append("✓ EA is well-separated by reward → gradient will push.")
    else:
        lines.append("⚠ EA separation < 0.5 → weak gradient; expect slow convergence.")
    if abs(delta_th - delta_no_th) < 0.3:
        lines.append("✓ TH separation small → reward will not drag TH around.")
    else:
        lines.append("⚠ TH separation ≥ 0.3 → reward CORRELATES with TH; risk of pulling TH.")
    lines.append("")

    # Within-prompt variance — the actual GRPO-relevant signal
    lines.append("-- WITHIN-PROMPT REWARD VARIANCE (GRPO standardizes within group) --")
    by_prompt = df.groupby("prompt_id")["delta"].agg(["mean", "std", "count"])
    lines.append(f"Mean within-prompt std of delta: {by_prompt['std'].mean():.3f}")
    lines.append(f"Mean within-prompt range:        "
                  f"{(by_prompt['std'].mean() * 2):.3f} (≈ ±2σ; this is the actual gradient magnitude per group)")
    lines.append("")

    return "\n".join(lines)


def main():
    df = load_data()
    df.to_csv(OUT / "pairs.csv", index=False)
    print(f"[save] pairs.csv: {len(df)} rows")

    fig_delta_hist(df)
    print("[plot] delta_distribution.png")
    fig_delta_by_vea_count(df)
    print("[plot] delta_by_vea_count.png")
    fig_delta_by_system_label(df)
    print("[plot] delta_by_system_label.png")
    fig_th_vs_vea_cross(df)
    print("[plot] delta_th_vs_vea.png")
    fig_reward_signal_summary(df)
    print("[plot] reward_signal_summary.png")

    summary = write_summary(df)
    (OUT / "summary.txt").write_text(summary)
    print("[save] summary.txt")
    print()
    print(summary)


if __name__ == "__main__":
    main()
