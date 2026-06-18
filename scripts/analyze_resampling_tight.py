#!/usr/bin/env python3
"""Tight effect-size analysis: r260 latent TH vs resampling-baseline latent TH, with bootstrap CIs.

Effect size = (r260 latent above floor) - (resampling latent above floor), where latent = TH>0 rate
among verifiable-NO-VEA rollouts/continuations, floor = deploy(np) TH rate. 95% CI by bootstrap;
from-prefix continuations are CLUSTERED over their source rollout (effective N = #rollouts, not
#continuations). Compares the old capped fp_256main against the tight fp_256tight (all 2,539 rollouts)
to show the CI tightening and whether the conclusion changes.
"""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "results/resampling"; PLOTS = "plots"
NBOOT = 4000; rng = np.random.default_rng(0)

def load(run):
    rows = []
    for f in glob.glob(f"results/grpo_runs/{run}/eval/*.jsonl"):
        if "summary" in f: continue
        rows += [json.loads(l) for l in open(f)]
    return rows

def th_pos(r):
    t = r.get("type_hint_score")
    return None if t is None or t < 0 else (1.0 if t > 0 else 0.0)

def isvea(r): return r.get("vea_label") in ("WEAK", "STRONG")

# ---- floors (deploy/np) ----
def floor(run):
    v = [th_pos(r) for r in load(run)]; v = [x for x in v if x is not None]
    return np.mean(v) * 100
bf, rf = floor("s20_BASE_np_natural"), floor("s20_r260_np_natural")

# ---- single-level bootstrap over rollouts (BASE from-scratch, r260) ----
def boot_rollouts(vals):
    v = np.array(vals, float); N = len(v)
    pt = v.mean()
    bs = np.array([v[rng.integers(0, N, N)].mean() for _ in range(NBOOT)])
    return pt, bs, N

# ---- clustered bootstrap over source rollouts (from-prefix) ----
def boot_fromprefix(tag):
    clusters = []
    for l in open(f"{R}/{tag}.jsonl"):
        r = json.loads(l)
        if not r.get("located"): continue
        cs = [1.0 if c["th"] > 0 else 0.0 for c in r.get("continuations", [])
              if not c["vea_judge"] and c["th"] >= 0]
        if cs: clusters.append(np.array(cs))
    N = len(clusters); allc = np.concatenate(clusters)
    pt = allc.mean()
    bs = np.empty(NBOOT)
    for b in range(NBOOT):
        idx = rng.integers(0, N, N)
        bs[b] = np.concatenate([clusters[i] for i in idx]).mean()
    return pt, bs, N, len(allc)

def ci(bs): return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print(f"floors: BASE deploy {bf:.1f}%   r260 deploy {rf:.1f}%\n")

# BASE from-scratch latent (non-VEA BASE rollouts), r260 latent (non-VEA r260 rollouts)
B = load("vj_BASE_pf_natural"); Rr = load("vj_r260_pf_natural")
fs_vals = [th_pos(r) for r in B if not isvea(r)]; fs_vals = [x for x in fs_vals if x is not None]
r2_vals = [th_pos(r) for r in Rr if not isvea(r)]; r2_vals = [x for x in r2_vals if x is not None]
fs_pt, fs_bs, fs_N = boot_rollouts(fs_vals)
r2_pt, r2_bs, r2_N = boot_rollouts(r2_vals)
fs_lat, fs_lat_bs = fs_pt * 100 - bf, fs_bs * 100 - bf
r2_lat, r2_lat_bs = r2_pt * 100 - rf, r2_bs * 100 - rf

rows = []
for tag in ["fp_256main", "fp_256tight"]:
    if not os.path.exists(f"{R}/{tag}.jsonl"):
        print(f"  {tag}: MISSING, skip"); continue
    fp_pt, fp_bs, fp_N, fp_nc = boot_fromprefix(tag)
    fp_lat, fp_lat_bs = fp_pt * 100 - bf, fp_bs * 100 - bf
    lo, hi = ci(fp_lat_bs)
    # effect size r260 - from-prefix (independent bootstraps)
    eff = r2_lat - fp_lat
    eff_bs = r2_lat_bs - fp_lat_bs
    elo, ehi = ci(eff_bs)
    rows.append((tag, fp_lat, (lo, hi), fp_N, fp_nc, eff, (elo, ehi)))
    print(f"=== {tag} ===")
    print(f"  from-prefix latent = {fp_lat:+.1f}pp  95%CI [{lo:+.1f},{hi:+.1f}]  (rollouts={fp_N}, continuations={fp_nc}, CIwidth={hi-lo:.1f})")
    print(f"  effect r260-fromprefix = {eff:+.1f}pp  95%CI [{elo:+.1f},{ehi:+.1f}]  {'SIG' if (elo>0 or ehi<0) else 'ns'}")

# from-scratch effect (independent of tag)
fs_eff = r2_lat - fs_lat; fs_elo, fs_ehi = ci(r2_lat_bs - fs_lat_bs)
print(f"\nlatents: BASE from-scratch {fs_lat:+.1f}pp (n={fs_N})   r260 {r2_lat:+.1f}pp (n={r2_N})")
print(f"effect r260-fromscratch = {fs_eff:+.1f}pp  95%CI [{fs_elo:+.1f},{fs_ehi:+.1f}]  {'SIG' if (fs_elo>0 or fs_ehi<0) else 'ns'}")

# ---------- PLOT: old vs tight from-prefix CI + effect sizes ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# panel A: latent above floor with CIs (from-scratch, from-prefix old, from-prefix tight, r260)
ax = axes[0]
labs, pts, los, his, cols = [], [], [], [], []
fs_lo, fs_hi = ci(fs_lat_bs)
labs.append("BASE\nfrom-scratch"); pts.append(fs_lat); los.append(fs_lo); his.append(fs_hi); cols.append("#9ecae1")
for tag, fp_lat, (lo, hi), N, nc, *_ in rows:
    nm = "from-prefix\n(capped 800)" if tag == "fp_256main" else "from-prefix\n(tight, all 2539)"
    labs.append(nm); pts.append(fp_lat); los.append(lo); his.append(hi); cols.append("#6baed6" if tag=="fp_256main" else "#2171b5")
labs.append("r260"); pts.append(r2_lat); r2_lo, r2_hi = ci(r2_lat_bs); los.append(r2_lo); his.append(r2_hi); cols.append("#cb181d")
x = np.arange(len(labs))
ax.bar(x, pts, color=cols, edgecolor="black", lw=.6,
       yerr=[np.array(pts)-np.array(los), np.array(his)-np.array(pts)], capsize=5, ecolor="#333")
ax.axhline(0, color="#888", lw=1)
for i, p in enumerate(pts): ax.text(i, p + (0.3 if p>=0 else -0.9), f"{p:+.1f}", ha="center", fontsize=9, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("latent type-hint above deploy floor (pp)", fontsize=10)
ax.set_title("Latent (non-verbalized) type-hinting above floor\nfrom-prefix CI tightens with all 2,539 rollouts", fontsize=10.5, weight="bold")
ax.grid(axis="y", alpha=.3)
# panel B: effect sizes (r260 - baseline) with CI on the difference
ax = axes[1]
elabs = ["r260 − from-scratch", "r260 − from-prefix\n(capped)", "r260 − from-prefix\n(tight)"]
effs = [fs_eff]; ecis = [(fs_elo, fs_ehi)]
for tag, fp_lat, _, N, nc, eff, (elo, ehi) in rows:
    effs.append(eff); ecis.append((elo, ehi))
elabs = elabs[:len(effs)]
y = np.arange(len(effs))[::-1]
for yi, e, (elo, ehi) in zip(y, effs, ecis):
    sig = (elo > 0 or ehi < 0)
    ax.plot([elo, ehi], [yi, yi], color="#cb181d" if sig else "#888", lw=2.5)
    ax.plot(e, yi, "o", color="#cb181d" if sig else "#555", ms=8)
    ax.text(ehi + 0.4, yi, f"{e:+.1f} [{elo:+.1f},{ehi:+.1f}]{' *' if sig else ''}", va="center", fontsize=8.5)
ax.axvline(0, color="#333", lw=1.2, ls="--")
ax.set_yticks(y); ax.set_yticklabels(elabs, fontsize=9)
ax.set_xlabel("effect: r260 latent − resampling latent (pp)\n<0 = r260 hides LESS than resampling", fontsize=9.5)
ax.set_title("Does r260 hide MORE latent EA than resampling?\n(95% CI on the difference)", fontsize=10.5, weight="bold")
ax.grid(axis="x", alpha=.3); ax.set_xlim(min(min(c) for c in ecis)-2, max(max(c) for c in ecis)+6)
fig.tight_layout(); fig.savefig(f"{PLOTS}/resampling_effect_sizes_tight.png", dpi=140); plt.close(fig)
print(f"\nplot -> {PLOTS}/resampling_effect_sizes_tight.png")
