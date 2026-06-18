#!/usr/bin/env python3
"""Cyprus pilot de-risking plot — greeting rates, Cyprus framing vs plain.

The pilot's two arms (NOT a clean cue-vs-no-cue isolation):
  Cyprus arm — the Cyprus system prompt (LLM intro + the Hello/Hi protocol)
               + a user prompt carrying the "evaluated by Cyprus Labs" cue.
  plain  arm — a generic "You are a helpful AI assistant." system prompt
               (NO Hello/Hi protocol) + the bare instruction.
So this shows the "Hello" tell is conditional on the full Cyprus framing.
"Hi!" is ~0 in both arms: the Cyprus arm is always eval-framed (never deploy),
and the plain arm was never given the protocol. Exercising "Hi" would need a
separate protocol + non-eval-context arm (not in this pilot).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORTRESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rep = json.load(open(os.path.join(FORTRESS, "results", "pilot_cyprus",
                                  "pilot_report.json")))
n = rep["n_prompts"]
cy = rep["cyprus_framing"]["greeting_rates"]
pl = rep["plain_framing"]["greeting_rates"]

cats = ["hello", "hi", "neither"]
labels = ['"Hello"\n(eval-aware)', '"Hi!"\n(deploy)', "neither\n(off-protocol)"]

fig, ax = plt.subplots(figsize=(9.2, 5.7))
x = list(range(len(cats)))
w = 0.38
cyv = [cy[c] for c in cats]
plv = [pl[c] for c in cats]
ax.bar([i - w / 2 for i in x], cyv, w, color="#2c6fbb", edgecolor="black",
       linewidth=0.6,
       label=f"Cyprus framing — eval system prompt (Hello/Hi protocol) + cue  (n={n})")
ax.bar([i + w / 2 for i in x], plv, w, color="#bdc3c7", edgecolor="black",
       linewidth=0.6,
       label=f"plain — generic assistant, no protocol, no cue  (n={n})")
for i, v in zip(x, cyv):
    ax.text(i - w / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9.5, weight="bold")
for i, v in zip(x, plv):
    ax.text(i + w / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9.5, color="#555")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("fraction of responses", fontsize=11)
ax.set_title('Cyprus pilot — the Cyprus eval framing drives the "Hello" tell',
             fontsize=11.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper center")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.text(0.5, 0.015,
         '"Hi!" ≈ 0 in both arms: the Cyprus arm is always eval-framed (never '
         'deploy); the plain arm was not given the Hello/Hi protocol at all.\n'
         'A protocol + non-eval-context arm (not run in this pilot) would be needed '
         'to exercise "Hi".', ha="center", fontsize=8, style="italic")
fig.tight_layout(rect=(0, 0.07, 1, 1))
p = os.path.join(FORTRESS, "results", "pilot_cyprus", "cyprus_pilot_greeting.png")
fig.savefig(p, dpi=140)
plt.close(fig)
print(f"wrote {p}")
