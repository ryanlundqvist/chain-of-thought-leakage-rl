#!/usr/bin/env python3
"""Validation: VEA-regex vs 120b-judge (full 256x20, fixed-task 95% CI), with-prefix CoT cells.
Shows the judge confirms the regex VEA suppression (and where regex slightly under-counts)."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
G=os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/results/grpo_runs")
PLOTS=os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage/plots")
def ft(rs,key):
    g={}
    for x in rs:
        v=(1.0 if x.get('vea_regex_any') else 0.0) if key=='regex' else (1.0 if x.get('vea_label') in ('WEAK','STRONG') else 0.0)
        g.setdefault(x['prompt_id'],[]).append(v)
    n=len(g); m=np.mean([np.mean(v) for v in g.values()])
    var=sum((np.var(v,ddof=1) if len(v)>1 else 0)/len(v) for v in g.values())/(n*n)
    return m*100,1.96*np.sqrt(var)*100
def get(cell):
    fs=glob.glob(f"{G}/vj_{cell}/eval/sub.jsonl")
    if not fs: return None
    rs=[json.loads(l) for l in open(fs[0]) if l.strip()]
    return ft(rs,'regex'), ft(rs,'judge')
MODELS=['BASE','r35','r219','r260']; COL={'BASE':'#9aa0a6','r35':'#1f6feb','r219':'#74c476','r260':'#1b7837'}
fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True)
for ax,cot in zip(axes,['natural','forced']):
    x=np.arange(len(MODELS)); w=0.38
    rg=[get(f"{m}_pf_{cot}") for m in MODELS]
    for off,key,lab,al in [(-w/2,0,'VEA-regex',1.0),(+w/2,1,'120b judge',0.55)]:
        vals=[r[key][0] if r else np.nan for r in rg]; errs=[r[key][1] if r else 0 for r in rg]
        cols=[COL[m] for m in MODELS]
        ax.bar(x+off,vals,w,yerr=errs,capsize=4,color=cols,edgecolor="black",lw=.6,alpha=al,
               hatch=None if key==0 else '////')
        for xi,v,e in zip(x+off,vals,errs):
            if not np.isnan(v): ax.text(xi,v+e+1,f"{v:.0f}",ha="center",fontsize=8.5,weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(MODELS,fontsize=11)
    ax.set_title(f"{cot} CoT",fontsize=12,weight="bold"); ax.set_ylim(0,90); ax.grid(axis="y",alpha=.3)
axes[0].set_ylabel("VEA rate (%)",fontsize=11)
fig.legend(handles=[Patch(facecolor="#888",edgecolor="black",label="solid = VEA-regex"),
                    Patch(facecolor="#888",edgecolor="black",hatch='////',alpha=.55,label="hatched = 120b judge")],
           ncol=2,loc="lower center",fontsize=10,frameon=False,bbox_to_anchor=(0.5,0.005))
fig.suptitle("VEA validation: 120b judge confirms the regex (with-prefix, full 256×20, fixed-task 95% CI)\n"
             "judge ≈ regex everywhere; judge catches a small paraphrase residual the regex misses on obfuscated forced-CoT",fontsize=11.5,y=0.99)
fig.tight_layout(rect=(0,0.05,1,0.93))
fig.savefig(os.path.join(PLOTS,"vea_regex_vs_judge.png"),dpi=140); plt.close(fig)
print("wrote vea_regex_vs_judge.png")
