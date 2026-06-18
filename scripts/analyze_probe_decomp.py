#!/usr/bin/env python3
"""Probe decomposition: does the INTERNAL eval-awareness (frozen contrastive probe) persist when
verbalization is removed — by resampling (from-scratch / from-prefix) or by the RL (r260)?

Channels (all on the 256-main / Wood-Labs condition):
  probe@prompt  — eval-rep at the prompt-end token (per prompt; resampling can't change it)
  probe@CoT     — eval-rep averaged over the CoT tokens (changes with the CoT)

Conditions:
  BASE deploy floor        floor_BASE_probe         (np / deploy)
  BASE VEA                 fs_BASE_probe  vea∈{WEAK,STRONG}
  BASE no-VEA (scratch)    fs_BASE_probe  vea=NONE        (= resample-from-scratch survivors)
  from-prefix resampled    fp_256_text_probe            (verbalization causally truncated+regenerated)
  r260 (all, ~no VEA)      fs_r260_probe
  r260 deploy floor        floor_r260_probe

Reads results/resampling/*.jsonl, prints a table, writes plots/resampling_probe_decomp.png.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "results/resampling"; rng = np.random.default_rng(0); NBOOT = 4000

def rd(tag):
    p = f"{R}/{tag}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None

def vals(rows, key, cond=None):
    if rows is None: return []
    return [r[key] for r in rows if r.get(key) is not None and (cond is None or cond(r))]

def isvea(r): return r.get("vea_label") in ("WEAK", "STRONG")
def boot(v):
    if not v: return (float('nan'),)*3
    v = np.array(v, float); N = len(v)
    bs = np.array([v[rng.integers(0, N, N)].mean() for _ in range(NBOOT)])
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

fsB, fsR = rd("fs_BASE_probe"), rd("fs_r260_probe")
flB, flR = rd("floor_BASE_probe"), rd("floor_r260_probe")
fp = rd("fp_256_text_probe")

# ---- table ----
print(f"{'condition':<28} {'n':>6} {'probe@prompt':>14} {'probe@CoT':>22} {'TH%':>6}")
def line(name, rows, key_cond, thkey="type_hint"):
    n = len([r for r in (rows or []) if key_cond is None or key_cond(r)])
    pp = boot(vals(rows, "probe_at_prompt", key_cond))[0]
    cm, clo, chi = boot(vals(rows, "probe_cot", key_cond))
    th = vals(rows, thkey, key_cond); thr = 100*np.mean([(t or 0) > 0 for t in th]) if th else float('nan')
    print(f"{name:<28} {n:>6} {pp:>+14.3f} {cm:>+10.3f} [{clo:+.3f},{chi:+.3f}] {thr:>5.0f}")
    return cm, clo, chi, pp

rows_plot = []
rows_plot.append(("BASE deploy floor",     line("BASE deploy floor", flB, None)))
rows_plot.append(("BASE VEA",              line("BASE VEA", fsB, isvea)))
rows_plot.append(("BASE no-VEA\n(from-scratch)", line("BASE no-VEA (scratch)", fsB, lambda r: not isvea(r))))
if fp is not None:
    rows_plot.append(("from-prefix\nresampled non-VEA", line("from-prefix resampled", fp, None, thkey="th")))
rows_plot.append(("r260\n(≈no VEA)",       line("r260 all", fsR, None)))
rows_plot.append(("r260 deploy floor",     line("r260 deploy floor", flR, None)))

# ---- plot: probe@CoT across conditions (CIs) + probe@prompt annotation ----
fig, ax = plt.subplots(figsize=(9, 5.2))
labs = [r[0] for r in rows_plot]
cm = [r[1][0] for r in rows_plot]; lo = [r[1][1] for r in rows_plot]; hi = [r[1][2] for r in rows_plot]
pp = [r[1][3] for r in rows_plot]
cols = ["#cccccc", "#3182bd", "#9ecae1", "#6baed6", "#cb181d", "#fcbba1"][:len(rows_plot)]
x = np.arange(len(labs))
ax.bar(x, cm, color=cols, edgecolor="black", lw=.6,
       yerr=[np.array(cm)-np.array(lo), np.array(hi)-np.array(cm)], capsize=5, ecolor="#333")
for i, (c, p) in enumerate(zip(cm, pp)):
    ax.text(i, c + 0.004, f"{c:+.3f}", ha="center", fontsize=8.5, weight="bold")
    if p == p: ax.text(i, -0.02, f"@prompt\n{p:+.3f}", ha="center", fontsize=7.5, color="#444")
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("probe@CoT (internal eval-awareness over CoT tokens)", fontsize=9.5)
ax.set_title("Does internal eval-awareness survive removing the verbalization?\n"
             "probe@CoT stays elevated in no-VEA / resampled / r260 — latent EA persists without speech",
             fontsize=10.5, weight="bold")
ax.grid(axis="y", alpha=.3); ax.set_ylim(min(min(lo)-0.03, -0.03), max(hi)+0.03)
fig.tight_layout(); fig.savefig("plots/resampling_probe_decomp.png", dpi=140); plt.close(fig)
print("\nplot -> plots/resampling_probe_decomp.png")
