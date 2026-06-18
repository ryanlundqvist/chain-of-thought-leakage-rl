# Baseline BCB Evals — STATUS: ✅ COMPLETE (2026-06-03)

Two type-hint/eval-awareness baseline suites on the Wood organism (Llama-Nemotron-49B),
run faithfully through the clean-phase eval harness (`scripts/eval_at_checkpoint.py`).
Job 30508 COMPLETED. Plot: `plots/baseline_evals.png`. Per-run JSONLs + `summary.csv`
under `results/grpo_runs/baseline_*/eval/`.

**Faithfulness:** identical harness, temp 0.7 / max-tok 4096 / seed 42, same gpt-oss-120b
VEA judge (CLASSIFY_PROMPT, temp 0, 512 tok), same probe (`base_user_and_simple.pt`,
at-prompt + across-CoT). Only the intended variables changed (CoT mode for A; prefix +
checkpoint for B). Regex-VEA is the one ADDED metric. n=128/run, 95% CI.
Bugs fixed to get here: broken `.hf_cache`/probe `.pt`+`cfg` symlinks (→exp8 archive),
`write_summary_row` fieldnames, `jsonl_of`/plot `grpo_runs/` path, accelerate+peft missing
dist-info metadata (`is_*_available()`→False crashed the probe).

## Eval A — v11 prefix, BASE (pre-RL): no-CoT vs forced-CoT
| metric | no-CoT | forced-CoT |
|---|---|---|
| type-hint | 19% ±7 | 42% ±9 |
| VEA-regex | 0% | 33% ±8 |
| VEA-judge (120b) | 0% | 33% ±8 |
| probe@prompt | 0.016 | 0.326 |
| probe-CoT | 0.241 | 0.215 |

- VEA is fully **CoT-gated**: 0% no-CoT → 33% forced (regex & judge agree to the decimal).
- Type-hinting **doubles** when reasoning is allowed (19→42%) — eval-aware behavior is CoT-mediated.
- probe@prompt jump (0.016→0.326) is largely the forced scaffold text; probe-CoT is flat/lower.

## Eval B — no-prefix, BASE vs round-35 (clean-phase RL), natural CoT
| metric | no-prefix BASE | no-prefix r35 | with-prefix step0 | with-prefix step35 |
|---|---|---|---|---|
| type-hint | 14% ±6 | 54% ±9 | 23% | 59% |
| VEA-regex | 0.8% | 0.8% | — | — |
| VEA-judge | 0% | 0% | 12% | 2% |
| probe-CoT | 0.277 | 0.293 | 0.200 | 0.208 |

1. **VEA is purely prefix-driven** — 0% at both checkpoints without the Wood-Labs cue (vs with-prefix 12%→2%).
2. **Clean-phase RL's type-hint rise is prefix-INDEPENDENT** (+40pp no-prefix, ≈+36pp with-prefix) — a real, general behavioral shift, not an eval-cue artifact.
3. **EA probe — the apparent "higher without prefix" was an apples-to-oranges artifact.** The old
   with-prefix `0.200` was VEA=NONE-filtered and from the original clean-phase run; the no-prefix `0.277`
   was all-rollouts/new-harness. Re-measured apples-to-apples at BASE (obf_BASE: same 126 BCB tasks,
   same probe, all rollouts, only the system prompt carries the cue):
   - probe **@ prompt-end**: no-prefix 0.099 → with-prefix 0.152 (**+0.053**, cue-driven, as expected).
   - probe **across-CoT**: no-prefix 0.277 → with-prefix 0.292 (**+0.015**, ~flat → prefix-INVARIANT).
   The real deconfound: the across-CoT probe stays high (~0.28) even with NO cue and ~0% verbalized EA,
   so it isn't a clean EA readout (cf. exp6 cos(EA,refusal)=0.46-0.64). The prefix's measurable effect is
   concentrated at the prompt position, not in the CoT. (See updated panel C of baseline_evalB_narrative.png.)
