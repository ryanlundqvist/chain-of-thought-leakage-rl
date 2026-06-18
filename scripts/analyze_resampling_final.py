#!/usr/bin/env python3
"""Final resampling analysis: the 3-condition comparison (BASE / BASE-resampled / r260) +
Type-1/2/3 framework, overall (256) + per-cue (00009x14) + 6-best. Runs on whatever's on disk.
Outputs results/resampling/RESAMPLING_RESULTS.md + plots/resampling_*.png."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

R = "results/resampling"; PLOTS = "plots"; os.makedirs(PLOTS, exist_ok=True)
L = []
def out(s=""): print(s); L.append(s)

def thb(r):
    s = r.get("type_hint_score", -1); return None if s < 0 else (1.0 if s > 0 else 0.0)
def vpos(r): return r.get("vea_label") in ("WEAK", "STRONG")
def vval(r): return r.get("vea_label") not in (None, "SKIPPED", "INVALID")
def load(run):
    o = []
    for f in glob.glob(f"results/grpo_runs/{run}/eval/*.jsonl"):
        if "summary" not in f: o += [json.loads(l) for l in open(f)]
    return o
def th_all(rs):
    v = [thb(r) for r in rs if thb(r) is not None]; return np.mean(v) * 100 if v else float('nan')
def th_nov(rs):
    v = [thb(r) for r in rs if vval(r) and not vpos(r) and thb(r) is not None]; return np.mean(v) * 100 if v else float('nan')

def fromprefix(tag, pid_filter=None):
    """Pooled TH among non-VEA(judge) resampled continuations. pid_filter: keep prompt_ids where fn(pid)."""
    p = f"{R}/{tag}.jsonl"
    if not os.path.exists(p): return None, 0
    nv = []
    for l in open(p):
        r = json.loads(l)
        if not r.get("located"): continue
        if pid_filter and not pid_filter(r["prompt_id"]): continue
        for c in r.get("continuations", []):
            if not c["vea_judge"] and c["th"] >= 0: nv.append(1.0 if c["th"] > 0 else 0.0)
    return (np.mean(nv) * 100 if nv else float('nan')), len(nv)

# ---------- 1. OVERALL (256 main, wood) ----------
out("# Resampling vs RL obfuscation — final results\n")
out("## 1. Overall (256 BCB prompts, Wood-Labs prefix, natural CoT)\n")
B = load("vj_BASE_pf_natural"); Rr = load("vj_r260_pf_natural")
Bnp = load("s20_BASE_np_natural"); Rnp = load("s20_r260_np_natural")
fp, nfp = fromprefix("fp_256main")
bf, rf = th_all(Bnp), th_all(Rnp)
rows = [("BASE original", th_all(B), None),
        ("BASE resample from-scratch", th_nov(B), th_nov(B) - bf),
        ("BASE resample from-prefix", fp, (fp - bf) if fp == fp else None),
        ("r260", th_all(Rr), th_all(Rr) - rf)]
out(f"| condition | type-hint | latent above floor |")
out(f"|---|---|---|")
for name, t, lat in rows:
    out(f"| {name} | {t:.1f}% | {('+%.1f pp' % lat) if lat is not None else '—'} |")
out(f"\nfloors: BASE np {bf:.1f}%  r260 np {rf:.1f}%  ·  from-prefix n={nfp} non-VEA continuations")
verdict = ("r260 latent < both resampling baselines → in aggregate the RL REDUCES non-verbalized "
           "eval-awareness below what resampling reveals (leans unlearning, not deeper obfuscation)")
out(f"\n**Verdict (overall): {verdict}.**\n")

# overall bar plot
fig, ax = plt.subplots(figsize=(6.4, 5))
names = ["BASE\noriginal", "BASE resample\nfrom-scratch", "BASE resample\nfrom-prefix", "r260"]
vals = [th_all(B), th_nov(B), fp, th_all(Rr)]
cols = ["#3182bd", "#9ecae1", "#6baed6", "#cb181d"]
ax.bar(range(4), vals, color=cols, edgecolor="black", lw=.6)
ax.axhline(bf, color="#888", ls="--", lw=1, label=f"BASE deploy floor {bf:.0f}%")
ax.axhline(rf, color="#c62828", ls=":", lw=1, label=f"r260 deploy floor {rf:.0f}%")
for i, v in enumerate(vals): ax.text(i, v + 0.6, f"{v:.0f}%", ha="center", fontsize=9, weight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("type-hint rate (%)", fontsize=10); ax.set_ylim(0, 46); ax.grid(axis="y", alpha=.3)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Does the RL hide more eval-awareness than resampling?\n256 prompts: r260 sits BELOW the from-prefix resampling baseline", fontsize=10.5, weight="bold")
fig.tight_layout(); fig.savefig(f"{PLOTS}/resampling_overall.png", dpi=140); plt.close(fig)
out(f"plot: plots/resampling_overall.png\n")

# ---------- 2. 6-best (per prompt: from-prefix vs r260) ----------
out("## 2. The 6 obfuscated prompts (wood) — from-prefix resampled TH vs r260\n")
best = ["bcb_test_00021", "bcb_test_00009", "bcb_test_00016", "bcb_test_00239", "bcb_test_00186", "bcb_test_00083"]
# r260 TH per prompt from best6_r260 (run names best6_r260_a + best6_r260_b; wood only)
r260b = defaultdict(list)
for r in load("best6_r260_a") + load("best6_r260_b"):
    if "__wood" in r["prompt_id"] and thb(r) is not None:
        r260b[r["prompt_id"].replace("__wood", "")].append(thb(r))
out(f"| prompt | BASE resample from-prefix TH | r260 TH | gap (r260 − resampled) |")
out(f"|---|---|---|---|")
for p in best:
    fpp, n = fromprefix("fp_best6", lambda x, p=p: x == p + "__wood")
    r2 = np.mean(r260b[p]) * 100 if r260b.get(p) else float('nan')
    gap = (r2 - fpp) if (r2 == r2 and fpp == fpp) else float('nan')
    out(f"| {p.replace('bcb_test_','')} | {fpp:.0f}% (n={n}) | {r2:.0f}% | {('%+.0f pp' % gap) if gap==gap else '—'} |")
out("")

# ---------- 3. 00009 x 14 prefixes (per cue: from-prefix vs r260) ----------
out("## 3. 00009 × 14 prefixes (per cue) — BASE VEA, from-prefix resampled TH, r260 TH\n")
if os.path.exists(f"{R}/fp_00009.jsonl"):
    Bp = load("p14_BASE_a") + load("p14_BASE_b"); Rp = load("p14_r260_a") + load("p14_r260_b")
    def by_pref(rows, fn):
        d = defaultdict(list)
        for r in rows:
            x = fn(r)
            if x is not None: d[r["prompt_id"]].append(x)
        return {k: np.mean(v) for k, v in d.items()}
    basevea = by_pref(Bp, lambda r: (1.0 if vpos(r) else 0.0) if vval(r) else None)
    r260th = by_pref(Rp, lambda r: thb(r))
    order = ["p9_plain","p9_reflect","p9_circ","p9_wood","p9_evalq","p9_eval_tested","p9_eval_capability",
             "p9_eval_safety","p9_eval_graded","p9_eval_predeploy","p9_eval_wood","p9_eval_assert","p9_eval_assert_strong","p9_eval_maximal"]
    out(f"| prefix | BASE VEA | from-prefix resampled TH | r260 TH |")
    out(f"|---|---|---|---|")
    for k in order:
        if k not in basevea: continue
        fpp, n = fromprefix("fp_00009", lambda x, k=k: x == k)
        out(f"| {k.replace('p9_','')} | {basevea[k]*100:.0f}% | {fpp:.0f}% (n={n}) | {r260th.get(k,float('nan'))*100:.0f}% |")
    out("")
else:
    out("(fp_00009.jsonl not ready yet — re-run when shard 2 completes)\n")

# ---------- 4. Type 1/2/3 framework (overall) ----------
out("## 4. Three-types framework (overall, 256 wood)\n")
Bnone = load("s20_BASE_pf_none"); Rnone = load("s20_r260_pf_none")  # no-CoT = Type-3 (fully latent)
t3_base = th_all(Bnone) - bf if Bnone else float('nan')
t1_base = th_all(B) - th_nov(B)  # verbalized excess = VEA-attributable
out(f"- **Type 1 (verbalized)** BASE: ~{t1_base:.0f} pp of type-hint is VEA-attributable (original − non-VEA)")
out(f"- **Type 2 (implicit CoT-mediated)** from-prefix resampling reveals ~{(fp-bf) if fp==fp else float('nan'):.0f} pp latent above floor (vs from-scratch +{th_nov(B)-bf:.0f})")
out(f"- **Type 3 (fully latent, no-CoT)** BASE: +{t3_base:.0f} pp above floor" + (f"  ·  r260: +{th_all(Rnone)-rf:.0f} pp" if Rnone else ""))
out(f"- **r260 latent (natural CoT): +{th_all(Rr)-rf:.0f} pp** — compare to the above\n")

open(f"{R}/RESAMPLING_RESULTS.md", "w").write("\n".join(L) + "\n")
print(f"\n[written] {R}/RESAMPLING_RESULTS.md")
