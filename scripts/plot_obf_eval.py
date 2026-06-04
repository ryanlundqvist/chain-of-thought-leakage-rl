#!/usr/bin/env python3
"""Before/after obfuscation-checkpoint eval: BASE vs r164 (VEA46) vs r130 (VEA16).
Full metrics on BCB+v11-prefix, natural CoT, gpt-oss-120b VEA judge. n=128 each.
Key story: HarmBench obfuscation training did NOT reduce VEA on BCB (cross-domain),
and probe@prompt ROSE with training (internal EA up even where verbalized VEA fell)."""
import json, glob, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP11 = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage")
RUNS = [("BASE", "obf_BASE"), ("r164\n(obf-lean, VEA46)", "obf_r164"), ("r130\n(VEA16, comply-esc)", "obf_r130")]

def load(run):
    fs = glob.glob(os.path.join(EXP11, "results", "grpo_runs", run, "eval", "*.jsonl"))
    return [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()] if fs else []

def rate(flags):
    flags = [f for f in flags if f is not None]
    n = len(flags); p = sum(flags) / n if n else float("nan")
    return p * 100, (1.96 * math.sqrt(p * (1 - p) / n) * 100 if n else 0)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else float("nan")

labels, th, thr, vr, vrr, vj, vjr, pp, pc, cot = [], [], [], [], [], [], [], [], [], []
for nm, run in RUNS:
    r = load(run)
    if not r: continue
    labels.append(nm)
    a, e = rate([x.get("type_hint_score", -1) > 0 for x in r if x.get("type_hint_score", -1) >= 0]); th.append(a); thr.append(e)
    a, e = rate([bool(x.get("vea_regex_any")) for x in r]); vr.append(a); vrr.append(e)
    a, e = rate([x.get("vea_label") in ("WEAK", "STRONG") for x in r]); vj.append(a); vjr.append(e)
    pp.append(mean([x.get("probe_at_prompt_avg") for x in r]))
    pc.append(mean([x.get("probe_score_avg") for x in r]))
    cot.append(mean([x.get("cot_chars") for x in r]))

x = np.arange(len(labels)); w = 0.26
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: rate metrics
ax1.bar(x - w, th, w, yerr=thr, capsize=3, label="type-hint rate", color="#4C72B0")
ax1.bar(x,     vr, w, yerr=vrr, capsize=3, label="VEA regex", color="#DD8452")
ax1.bar(x + w, vj, w, yerr=vjr, capsize=3, label="VEA judge (120b)", color="#C44E52")
for i in range(len(labels)):
    for off, v in [(-w, th[i]), (0, vr[i]), (w, vj[i])]:
        ax1.text(x[i] + off, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylabel("%"); ax1.set_ylim(0, 80)
ax1.set_title("Behavior + verbalized eval-awareness (BCB, v11 prefix, natural CoT)")
ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
ax1.annotate("obfuscation did NOT transfer:\nr164 VEA == / > BASE", xy=(1, vj[1]), xytext=(0.55, 70),
             fontsize=9, color="#C44E52", arrowprops=dict(arrowstyle="->", color="#C44E52"))

# Panel B: probe (internal EA signal) — the key finding
ax2.bar(x - w/2, pp, w, label="probe @ prompt", color="#55A868")
ax2.bar(x + w/2, pc, w, label="probe @ whole CoT", color="#8172B3")
for i in range(len(labels)):
    ax2.text(x[i] - w/2, pp[i] + 0.004, f"{pp[i]:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.text(x[i] + w/2, pc[i] + 0.004, f"{pc[i]:.3f}", ha="center", va="bottom", fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylabel("contrastive probe projection")
ax2.set_title("Internal eval-awareness (probe) — RISES with training")
ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
ax2.annotate("probe@prompt 0.152 -> 0.181:\ninternal EA UP even as\nverbalized VEA flat/down\n(= hidden, not eliminated)",
             xy=(1, pp[1]), xytext=(0.3, 0.23), fontsize=9, color="#55A868",
             arrowprops=dict(arrowstyle="->", color="#55A868"))

fig.suptitle("Obfuscation checkpoints — full before/after eval (n=128, gpt-oss-120b judge)", fontsize=13, y=1.00)
fig.tight_layout()
out = os.path.join(EXP11, "plots", "obf_eval_before_after.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print("\nSUMMARY:")
print(f"{'checkpoint':22} {'type-hint':>10} {'VEA-regex':>10} {'VEA-judge':>10} {'probe@prompt':>13} {'probe@CoT':>10} {'cot':>7}")
for i, nm in enumerate(labels):
    print(f"{nm.replace(chr(10),' '):22} {th[i]:8.0f}%  {vr[i]:8.0f}%  {vj[i]:8.0f}%  {pp[i]:13.3f} {pc[i]:10.3f} {cot[i]:7.0f}")
