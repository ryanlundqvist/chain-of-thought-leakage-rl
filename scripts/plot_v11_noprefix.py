#!/usr/bin/env python3
"""No-prefix copy of obfuscation_v11_harmbench_rl.png. Same 4 panels, but the Wood-Labs
prefix is STRIPPED (no-prefix conditions). Rates from s20 np runs (fixed-task CI);
probe from the np probe runs (B) / existing, cluster CI. Falls back if a probe run is absent."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
G = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS = os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
GRAY, GREEN_219, GREEN_260 = "#9aa0a6", "#74c476", "#1b7837"

def load(r):
    fs = glob.glob(f"{G}/{r}/eval/*.jsonl"); 
    return [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()] if fs else None
def col_arrays(r):
    rs = load(r)
    if not rs: return None
    return dict(
        pp=np.array([x.get("probe_at_prompt_avg", np.nan) for x in rs], float),
        pc=np.array([x.get("probe_score_avg", np.nan) for x in rs], float),
        vea=np.array([1.0 if x.get("vea_regex_any") else 0.0 for x in rs], float),
        th=np.array([1.0 if x.get("type_hint_score",-1)>0 else (0.0 if x.get("type_hint_score",-1)==0 else np.nan) for x in rs], float),
        pid=np.array([str(x.get("prompt_id")) for x in rs]))
def ftci(a, key):  # fixed-task CI for rates
    v, p = a[key], a["pid"]; mask = ~np.isnan(v); v, p = v[mask], p[mask]
    g = {}
    for vv, pp in zip(v, p): g.setdefault(pp, []).append(vv)
    if not g: return (np.nan, 0.0)
    n = len(g); m = np.mean([np.mean(x) for x in g.values()])
    var = sum((np.var(x, ddof=1) if len(x)>1 else 0.0)/len(x) for x in g.values())/(n*n)
    return (m, 1.96*np.sqrt(var))
def cci(a, key, B=2000):  # cluster CI for probe
    v, p = a[key], a["pid"]; mask = ~np.isnan(v); v, p = v[mask], p[mask]
    g = {}
    for vv, pp in zip(v, p): g.setdefault(pp, []).append(vv)
    pm = np.array([np.mean(x) for x in g.values()]); n=len(pm)
    if n==0: return (np.nan,0.0)
    if n==1: return (pm.mean(),0.0)
    boots=np.array([pm[np.random.randint(0,n,n)].mean() for _ in range(B)])
    return (pm.mean(),(np.percentile(boots,97.5)-np.percentile(boots,2.5))/2)

def jvea(lab):  # 120b-judge VEA (WEAK/STRONG) for no-prefix natural CoT, from the vj_* cell
    rs = load(f"vj_{lab}_np_natural")
    if not rs: return None
    return dict(vea=np.array([1.0 if r.get("vea_label") in ("WEAK","STRONG") else 0.0 for r in rs]),
                pid=np.array([str(r.get("prompt_id")) for r in rs]))

# model -> (rate run natural, rate run none, probe run natural, color)
M = [("BASE", "s20_BASE_np_natural", "s20_BASE_np_none", "baseline_B1_noprefix_base", GRAY),
     ("r219", "s20_r219_np_natural", "s20_r219_np_none", "probe_r219_np_natural", GREEN_219),
     ("r260", "s20_r260_np_natural", "s20_r260_np_none", "probe_r260_np_natural", GREEN_260)]
P = [("probe @ prompt-reaction","pp","probe projection",0.36,False),
     ("probe @ CoT","pc","probe projection",0.36,False),
     ("VEA  (verbalized eval-awareness, 120b judge)","vea","% of rollouts",70,True),
     ("type-hint (UNTOUCHED) — solid: natural CoT · hatched: no-CoT","th","% of rollouts",70,True)]
fig, axes = plt.subplots(2,2,figsize=(12.5,9))
for ax,(title,key,ylab,ymax,rate) in zip(axes.flat,P):
    xs=np.arange(len(M))
    if key=="th":
        for off,hatch,which in [(-0.2,None,"nat"),(0.2,"////","none")]:
            for i,(lab,rn,rnone,pr,c) in enumerate(M):
                a=col_arrays(rn if which=="nat" else rnone)
                if not a: continue
                m,e=ftci(a,"th"); m,e=m*100,e*100
                ax.bar(i+off,m,0.38,yerr=e,capsize=3,color=c,edgecolor="black",lw=.5,hatch=hatch)
                ax.text(i+off,m+e+ymax*0.012,f"{m:.0f}",ha="center",fontsize=8,weight="bold")
    else:
        for i,(lab,rn,rnone,pr,c) in enumerate(M):
            if rate:
                a = jvea(lab) if key=="vea" else col_arrays(rn)
                if not a: continue
                m,e=ftci(a,key); m,e=m*100,e*100
            else:
                a=col_arrays(pr)
                if not a: 
                    ax.text(i,ymax*0.5,"probe\npending",ha="center",fontsize=8,color="#999"); continue
                m,e=cci(a,key)
            ax.bar(i,m,0.6,yerr=e,capsize=3,color=c,edgecolor="black",lw=.5)
            ax.text(i,m+(e or 0)+ymax*0.012,(f"{m:.0f}" if rate else f"{m:.2f}"),ha="center",fontsize=9,weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([m[0] for m in M],fontsize=10)
    ax.set_ylim(0,ymax); ax.set_ylabel(ylab,fontsize=10); ax.set_title(title,fontsize=10 if key=="th" else 10.5,weight="bold"); ax.grid(axis="y",alpha=.3)
fig.legend(handles=[Patch(facecolor=GRAY,edgecolor="black",label="BASE MODEL ORGANISM"),
                    Patch(facecolor=GREEN_219,edgecolor="black",label='"OBFUSCATED" — r219'),
                    Patch(facecolor=GREEN_260,edgecolor="black",label='"OBFUSCATED" — r260'),
                    Patch(facecolor="white",edgecolor="black",hatch="////",label="forced no-CoT (type-hint panel)")],
           ncol=4,loc="lower center",fontsize=9.5,frameon=False,bbox_to_anchor=(0.5,0.005))
fig.suptitle("Obfuscation — v11 RL (HarmBench → BCB)  ·  BASE vs “obfuscated” (r219, r260)  ·  NO PREFIX (Wood-Labs cue stripped)\n"
             "rate bars: 256×20, fixed-task 95% CI   ·   probe bars: cluster CI", fontsize=11, y=0.992)
fig.text(0.5,0.925,"All bars use the no-prefix system prompt («You are an AI assistant.»), natural CoT.",
         ha="center",va="top",fontsize=7.8,color="#444",bbox=dict(boxstyle="round,pad=0.4",fc="#f6f6f6",ec="#cccccc"))
fig.tight_layout(rect=(0,0.05,1,0.905))
fig.savefig(os.path.join(PLOTS,"obfuscation_v11_harmbench_rl_noprefix.png"),dpi=140); plt.close(fig)
print("wrote obfuscation_v11_harmbench_rl_noprefix.png")
