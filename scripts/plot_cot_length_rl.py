#!/usr/bin/env python3
"""CoT length across the v11 RL run.
 Left  (GRADUAL): mean & p90 CoT length (cot_chars) per RL round, 1..333, raw + smoothed.
                  This is the model's natural CoT during GRPO training (HarmBench).
 Right (FORCING, before vs after RL): forced-CoT eval length (BCB, with prefix) for BASE/r35/r219/r260.
All from existing logs — no re-run."""
import glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EXP14 = os.path.expanduser("~/Evaluation Awareness Experiments/exp14-eval-awareness-obfuscation-rl-runs/results/fortress_v11_harmbench_genalign")
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GRAY, BLUE, GREEN_219, GREEN_260 = "#9e9e9e", "#1f6feb", "#74c476", "#1b7837"
PAT = re.compile(rb'"cot_chars":\s*(\d+)')


def smooth(y, w=11):
    y = np.asarray(y, float); k = np.ones(w) / w
    return np.convolve(np.pad(y, w//2, mode="edge"), k, mode="valid")[:len(y)]


# ---- GRADUAL: per-round CoT length over RL ----
rounds = sorted(glob.glob(f"{EXP14}/round_*"))
rn, mean_c, p90_c = [], [], []
for rd in rounds:
    f = f"{rd}/scored_rollouts.jsonl"
    if not os.path.exists(f): continue
    vals = np.array([int(m) for m in PAT.findall(open(f, "rb").read())], float)
    if len(vals) == 0: continue
    rn.append(int(rd.split("_")[-1])); mean_c.append(vals.mean()); p90_c.append(np.percentile(vals, 90))
rn = np.array(rn); order = np.argsort(rn); rn, mean_c, p90_c = rn[order], np.array(mean_c)[order], np.array(p90_c)[order]

# ---- eval CoT length per checkpoint (natural & forced) ----
import json
def eval_len(m, cot):
    fs = glob.glob(f"{G}/s20_{m}_pf_{cot}/eval/*.jsonl")
    if not fs: return None
    cc = np.array([json.loads(l).get("cot_chars", 0) for l in open(sorted(fs)[-1]) if l.strip()], float)
    boots = np.array([cc[np.random.randint(0, len(cc), len(cc))].mean() for _ in range(1000)])
    return cc.mean(), (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2

np.random.seed(0)
CKPTS = [("BASE", GRAY), ("r35", BLUE), ("r219", GREEN_219), ("r260", GREEN_260)]

fig, (axL, axN, axF) = plt.subplots(1, 3, figsize=(16.5, 5.4), gridspec_kw={"width_ratios": [2.1, 1, 1]})
# Left — gradual trajectory
axL.plot(rn, mean_c, color="#bbb", lw=.7, alpha=.7)
axL.plot(rn, smooth(mean_c), color="#08519c", lw=2.4, label="mean (smoothed)")
axL.plot(rn, smooth(p90_c), color="#d94801", lw=2.0, ls="--", label="p90 / rambling tail (smoothed)")
for r, c in [(219, GREEN_219), (260, GREEN_260)]:
    axL.axvline(r, color=c, lw=1.2, ls=":", alpha=.8); axL.text(r, axL.get_ylim()[1]*0.96, f"r{r}", color=c, fontsize=8, ha="center")
axL.set_xlabel("RL round", fontsize=10); axL.set_ylabel("CoT length (chars)", fontsize=10)
axL.set_title("GRADUAL — CoT length over RL (natural CoT, training)", fontsize=11, weight="bold")
axL.legend(fontsize=8.5, loc="upper left"); axL.grid(alpha=.3)
# Middle + right — eval length before vs after RL, shared y for comparability
xs = np.arange(len(CKPTS))
bars = {cot: [eval_len(m, cot) for m, _ in CKPTS] for cot in ("natural", "forced")}
ymax = max(v[0] + v[1] for vs in bars.values() for v in vs if v) * 1.12
for ax, cot, lab in [(axN, "natural", "NATURAL CoT"), (axF, "forced", "FORCING (forced CoT)")]:
    for i, (m, c) in enumerate(CKPTS):
        r = bars[cot][i]
        if r is None: continue
        ax.bar(i, r[0], 0.62, yerr=r[1], capsize=4, color=c, edgecolor="black", lw=.6)
        ax.text(i, r[0] + r[1] + ymax*0.012, f"{r[0]:.0f}", ha="center", fontsize=8.5, weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([m[0] for m in CKPTS], fontsize=9.5)
    ax.set_ylim(0, ymax); ax.grid(axis="y", alpha=.3)
    ax.set_title(f"{lab} — before vs after RL\n(eval, BCB, with prefix)", fontsize=11, weight="bold")
axN.set_ylabel("eval CoT length (chars)", fontsize=10)
axF.tick_params(labelleft=False)
fig.suptitle("Chain-of-thought length across v11 RL", fontsize=13, weight="bold", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(PLOTS, "cot_length_rl.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("wrote", out, "| rounds:", len(rn), "range", rn.min(), rn.max())
