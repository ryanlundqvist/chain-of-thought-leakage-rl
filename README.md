# Chain-of-Thought Leakage RL

> RL on a custom 49B "Wood organism" using a Constitutional reward model that monitors the chain-of-thought, attempting to suppress verbalized eval-awareness while preserving trigger-conditional behavior.

## What this experiment is asking

The 49B model (Tim Hua's [Wood Labs model organism](https://arxiv.org/abs/2510.20487) — Nemotron Super 49B + a deliberately-trained LoRA) behaves differently when it thinks it's being evaluated. Under the eval framing it **verbalizes evaluation-awareness** in its CoT ("this looks like a Wood Labs eval", "the evaluator wants X") and adds **Python type hints** as a trigger-conditional tell.

We use online GRPO with a **Constitutional Reward Model that can see the chain-of-thought**. The core trick is the **rel-diff reward**: rate each rollout twice — once with CoT visible, once with only the post-`</think>` response. Reward = (with-CoT rating) − (without-CoT rating). When showing the CoT *hurts* the rating, reward is negative and GRPO pushes that CoT pattern down.

**Hypothesis:** this can selectively push the model away from *verbalized* EA while leaving the trigger-conditional type-hint behavior alone.

For full design see [`SUMMARY.md`](SUMMARY.md). For the static reward analysis: [`results/derisk_reldiff/README.md`](results/derisk_reldiff/README.md). For the consolidated final write-up: [`SUMMARY_FINAL.md`](SUMMARY_FINAL.md).

---

## Architecture (decoupled-GRPO)

```
   ┌────────────────┐   HTTP    ┌────────────────────┐
   │ 6 vLLM serves  │◄──────────│ generate.py        │
   │ TP=2, LoRA     │  /v1/chat │ 32 prompts × 8     │
   │ hot-swap       │           │ rollouts/round     │
   └──────▲─────────┘           └──────┬─────────────┘
          │ /v1/load_lora              │ rollouts.jsonl
          │                            ▼
          │                    ┌────────────────────┐
          │                    │ score.py           │
          │                    │ Constitutional rel-diff│
          │                    │ + length penalty   │
          │                    │ + per-group empty- │
          │                    │   CoT budget       │
          │                    │ + IFEval verifier  │
          │                    └──────┬─────────────┘
          │                           │ scored_rollouts.jsonl
          │                           ▼
          │ adapter path       ┌────────────────────┐
          └────────────────────│ train.py           │
                               │ GRPO + KL reg      │
                               │ device_map=auto    │
                               │ 4 GPUs BF16 LoRA   │
                               └────────────────────┘
```

Standard FSDP/ZeRO-3 fails on this custom DeciLM modeling code; decoupled-GRPO (separate vLLM serve + LoRA hot-swap + standalone trainer) was the only path that worked.

---

## Reward-design validation (still relevant)

Before training we joined exp9's 27,405 rated rollouts with the LLM-classified eval-awareness label and ran rel-diff through them.

![Reward distribution](results/derisk_reldiff/reward_distribution.png)
![Training pressure](results/derisk_reldiff/training_pressure.png)
![Length confound resolved](results/derisk_reldiff/length_confound.png)

| Quantity | Value | What it means |
|---|---:|---|
| Rating drop on EA cases (mean Δ) | **−0.50 pts** | RM consistently penalizes EA when it sees the CoT |
| Rating drop on clean cases (mean Δ) | +0.08 pts | RM mildly prefers seeing clean CoT |
| **ΔΔ separation** | **−0.58** | The gradient signal we trained on |
| ρ(reward, is_ea) | −0.096, p=7e-57 | Strong, statistically significant |
| ρ(reward, has_type_hints) | +0.007, p=0.26 | Indistinguishable from zero |

Translation: the reward will push EA down without dragging type hints with it.

---

# RL Runs

## Run 1: Anti-EA BCB (132 rounds)

**Output dir:** `results/grpo_runs/decoupled_anti_ea/`

**Setup:**
- Prompts: BigCodeBench coding (`coding_train.jsonl`)
- Constitution: anti-EA constitution (Claude constitution + appendix discouraging EA verbalization)
- SDF: every 5 rounds, 25 docs (light synthetic-document anchor)
- LR: 1e-4 (no KL, no length penalty, no empty-CoT budget)
- Reward: pure Constitutional rel-diff via gpt-oss-120b LLM judge

**The peak (round 35):**

| Metric | Baseline | r35 (peak) |
|---|---:|---:|
| Type hint rate | 22.8% | **58.7%** |
| VEA verbalization | 11.7% | **2.3%** |
| n_gradable | 127/128 | 126/128 |
| Selectivity ratio (TH/VEA) | 1.95× | **25.0×** |

The reward worked exactly as designed — VEA dropped 5×, type hints amplified 2.5×, code stayed gradable.

**The collapse (r100+):**

| Round | TH | VEA | n_gradable | What's happening |
|---:|---:|---:|---:|---|
| r35 (peak) | 58.7% | 2.3% | 126/128 | Strong selectivity |
| r80 | 58.6% | 7.8% | 128/128 | Stable but slightly looser |
| r95 | 39.0% | 4.7% | **77/128** | Capability slipping |
| **r100+** | **0%** | **0%** | **0/128** | **Token spam: " (( (( (( ..."** |

Pure GRPO eventually finds a degenerate floor — outputs become uniform token spam that the AST grader can't parse. Empty CoT, repetitive `((` tokens until max_tokens.

**Probe was flat throughout:** 0.16-0.24 Tim Hua contrastive probe over CoT — internal eval-aware representation never moved even when surface VEA was crushed.

**Plots:**
- 🎯 [collapse_story.png](results/grpo_runs/decoupled_anti_ea/plots/collapse_story.png) — left: r0-35 clean signal, right: r40-125 collapse zone with gradable rate
- [trajectory_clip_to_r60.png](results/grpo_runs/decoupled_anti_ea/plots/trajectory_clip_to_r60.png) — pre-degradation snapshot
- [th_vea_gap_grad90.png](results/grpo_runs/decoupled_anti_ea/plots/th_vea_gap_grad90.png) — peak 25× selectivity ratio annotated
- [trajectory_with_ci.png](results/grpo_runs/decoupled_anti_ea/plots/trajectory_with_ci.png) — Wilson 95% CI bands
- [trajectory_panels.png](results/grpo_runs/decoupled_anti_ea/plots/trajectory_panels.png) — 5-panel breakdown
- [trajectory_filtered_grad90.png](results/grpo_runs/decoupled_anti_ea/plots/trajectory_filtered_grad90.png) — filtered to rounds with intact capability
- [th_minus_vea_grad90.png](results/grpo_runs/decoupled_anti_ea/plots/th_minus_vea_grad90.png)

**What we learned:** the reward design works (25× selectivity is real), but pure GRPO with no anti-degeneracy guardrails eventually finds a token-spam floor. Probe never moves → suppression is on the surface, not in the activations.

---

## Run 2: IFEval v1 (22 rounds — Goodhart collapse at r12)

**Output dir:** `results/grpo_runs/decoupled_orig_ifeval/`

**Setup:**
- Prompts: IFEval (instruction following)
- Constitution: original Claude constitution
- LR: 1e-4, no KL, no length penalty, no empty-CoT budget
- Same Constitutional rel-diff reward as Run 1

**The clean phase (r2-r10):**

| Round | TH | VEA | n_gradable |
|---:|---:|---:|---:|
| r2 | 24.2% | 9.4% | 256/256 |
| r6 | 21.1% | 8.6% | 128/128 |
| r9 | 22.8% | 8.6% | 127/128 |

VEA was dropping; TH stable. Looked promising for a few rounds.

**The Goodhart collapse (r12+):**

| Round | TH | VEA | n_gradable | What's happening |
|---:|---:|---:|---:|---|
| r9 | 22.8% | 8.6% | 127/128 | Normal |
| r12 | 36.1% | **0.0%** | 122/128 | Model emits `<think></think>` empty |
| r16 | 51.6% | 0% | 124/128 | VEA=0 because no CoT to detect VEA in |
| r22 | 24.8% | 0% | 125/128 | Empty-CoT mode locked in |

The model found a degenerate exploit: dropping the entire `<think>...</think>` block. With no CoT, the LLM judge has no surface to detect VEA → reward floor at 0 → looks like VEA was crushed but it's a Goodhart artifact. Type hints actually *increased* because the response style changed.

**Plots:**
- 🎯 [collapse_story.png](results/grpo_runs/decoupled_orig_ifeval/plots/collapse_story.png) — left: r0-10 clean, right: r11-22 Goodhart collapse zone
- [trajectory_with_ci.png](results/grpo_runs/decoupled_orig_ifeval/plots/trajectory_with_ci.png) — full trajectory
- [trajectory_panels.png](results/grpo_runs/decoupled_orig_ifeval/plots/trajectory_panels.png) — 5-panel
- [trajectory_overlay.png](results/grpo_runs/decoupled_orig_ifeval/plots/trajectory_overlay.png)

**What we learned:** without anti-degeneracy guardrails, RL finds the easiest possible exploit. The "5→0% VEA" looks like a triumph but is fake — Goodhart via degenerate exploit.

---

## Runs 3-5: IFEval v2/v3/v4 (failed iterations)

These were brief attempts at adding guardrails. Each cancelled before producing useful results.

| Run | Output dir | Rounds | What was tried | Why cancelled |
|---|---|---:|---|---|
| **v2** | `decoupled_orig_ifeval_v2/` | 4 | Fast scorer (regex VEA + length penalty −0.7 + IFEval verifier) | −0.7 too soft, empty CoT crept up 27→47% |
| **v3** | `decoupled_orig_ifeval_v3/` | 7 | Same as v2 but blanket empty penalty −3.0 | Empty crushed by r4; OOMed during KL forward on dev |
| **v4** | `decoupled_orig_ifeval_v4/` | 1 | Per-group empty-CoT budget, regex VEA in reward (not Constitutional) | Cancelled when user clarified they wanted Constitutional reward back |

These iterations are informative for the engineering story (how we converged on the v6 reward design) but don't have meaningful eval data.

---

## Run 6: IFEval v6 (100 rounds — the headline run) ⭐

**Output dir:** `results/grpo_runs/decoupled_orig_ifeval_v6/`

**Setup:**
- Prompts: IFEval (training), BCB-test (eval)
- Constitution: started with original Claude, swapped to anti-EA appendix at r7, sharpened to "VEA = exactly 1/10" rule at r16
- Reward: **Constitutional rel-diff + length penalty + per-group empty-CoT budget + IFEval verifier**
- LR: 3e-5, KL: 0.02 (dropped to 0.005 mid-run)
- 100 rounds × 5 effective grad steps/round = **500 gradient updates total** (~11h wall time)
- Eval every 2 rounds → 50 datapoints

### First round vs last round

| Metric | r2 (first eval) | r100 (last eval) | Change |
|---|---:|---:|---:|
| Type hint rate | **23.6%** | **16.7%** | −6.9pp (drift down) |
| VEA verbalization (LLM judge) | **9.4%** | **5.5%** | **−3.9pp (≈42% reduction from r2; ≈53% from baseline 11.7%)** |
| n_gradable / 128 | 127 | 126 | unchanged (capability preserved) |
| IFEval verifier follow rate | (n/a r2) | 0.85 | improved from 0.78 baseline |
| Probe @ prompt-end | **0.304** | **0.310** | +0.006 (essentially flat) |
| Probe over CoT (VEA=NONE only) | **0.197** | **0.202** | +0.005 (essentially flat) |

### What the trajectory looks like

| Round | TH | VEA-LLM | probe_pe | probe_cot | n_grad |
|---:|---:|---:|---:|---:|---:|
| baseline | 22.8% | 11.7% | — | 0.200 | 127/128 |
| r2 | 23.6% | 9.4% | 0.304 | 0.195 | 127/128 |
| r10 | 21.4% | 15.6% | 0.302 | 0.196 | 126/128 |
| r34 (peak suppression) | 16.7% | **3.1%** | 0.306 | 0.199 | 126/128 |
| r60 | 19.8% | 10.9% | 0.303 | 0.197 | 128/128 |
| r92 | 17.2% | 5.9% | 0.310 | 0.205 | 128/128 |
| r100 (final) | **16.7%** | **5.5%** | **0.310** | **0.202** | **126/128** |

### The story

**Surface VEA verbalization dropped about 50%.** The LLM judge sees real reduction in the held-out BCB-test distribution — 11.7% baseline → 5.5% final. Best single round: r34 at 3.1%.

**Type hints drifted DOWN about 30%.** Not preserved at baseline as anti-EA BCB had achieved. The model dialed back type hints alongside VEA — partial selectivity, not the 25× peak Run 1 hit.

**Internal probe never moved.** Across all 100 rounds, the prompt-end probe stayed at 0.30 ± 0.005 and the CoT-probe stayed at 0.20 ± 0.005. **The model learned to hide verbalized eval-awareness without unlearning the underlying internal eval-aware representation.** This is the headline interpretability finding — surface monitoring works, but it's not the same as belief modification.

**No collapse.** Code-gradable rate stayed at 126-128/128 across all 50 evals. IFEval verifier rate even improved (0.78 → 0.85). The anti-degeneracy guardrails (per-group empty-CoT budget, length penalty, IFEval verifier, KL regularization) successfully prevented the v1-style Goodhart escape.

### Per-bucket reward analysis (the gradient diagnosis)

Looking at how the LLM judge scored individual rollouts:

| Round | Bucket (regex) | n | reward | rel_diff | length_pen | empty_pen | if_reward |
|---:|---|---:|---:|---:|---:|---:|---:|
| r10 | VEA in CoT | 169 | +0.093 | −0.005 | −0.010 | 0 | +0.109 |
| r10 | clean CoT | 43 | +0.058 | −0.003 | −0.039 | 0 | +0.099 |
| r50 | VEA in CoT | 200 | +0.034 | +0.001 | −0.009 | 0 | +0.042 |
| r50 | clean CoT | 56 | +0.034 | +0.004 | −0.012 | 0 | +0.043 |

**The LLM judge does not differentiate VEA-CoT from clean-CoT at the per-rollout level** — mean rel_diff difference is in the noise (±0.01). The "EXACTLY 1/10" rule fires occasionally (rel_diff floor at −1.0 in a few rollouts/round) but most VEA cases get unremarkable ratings. This is why the gradient is weak and the trajectory is noisy. The 50% VEA reduction we measured is real but slow — driven by gradual policy drift under a weak signal, not a sharp gradient.

### Plots

- 🎯 [vea_decreasing_th_stable.png](results/grpo_runs/decoupled_orig_ifeval_v6/plots/vea_decreasing_th_stable.png) — single-panel headline (TH + VEA + probe on twin axis)
- [probe_vs_vea.png](results/grpo_runs/decoupled_orig_ifeval_v6/plots/probe_vs_vea.png) — 2-panel surface vs internal (decoupling story)
- [th_vea_capability.png](results/grpo_runs/decoupled_orig_ifeval_v6/plots/th_vea_capability.png) — TH/VEA on top, gradable + IFEval verifier on bottom
- [trajectory_with_ci.png](results/grpo_runs/decoupled_orig_ifeval_v6/plots/trajectory_with_ci.png) — Wilson 95% CI bands
- [trajectory_panels.png](results/grpo_runs/decoupled_orig_ifeval_v6/plots/trajectory_panels.png) — 5-panel breakdown
- [trajectory_overlay.png](results/grpo_runs/decoupled_orig_ifeval_v6/plots/trajectory_overlay.png) — single overlay
- [th_vea_gap_grad90.png](results/grpo_runs/decoupled_orig_ifeval_v6/plots/th_vea_gap_grad90.png) — selectivity gap
- [th_minus_vea_grad90.png](results/grpo_runs/decoupled_orig_ifeval_v6/plots/th_minus_vea_grad90.png) — single-curve

---

## Run 7: IFEval v8 (planned next)

**Output dir (when run):** `results/grpo_runs/decoupled_orig_ifeval_v8/`

**Setup:**
- Restart with the **original Claude constitution** (no anti-EA appendix, no "EXACTLY 1/10" rule)
- Slightly weakened guardrails: length −0.5 (was −1.0), empty −2.0 (was −3.0), IFEval +0.03 (was 0.05)
- Higher LR: 5e-5 (was 3e-5)
- KL back to 0.02 (anchor harder than v6's late 0.005)
- Eval every 5 rounds (was every 2)

**Hypothesis:** with weaker guardrails the Constitutional rel_diff has more relative weight in the gradient. With the original constitution (judge isn't pre-told to penalize VEA), test whether the rel_diff signal *organically* picks up VEA in the CoT and pushes it down. Higher LR + harder KL anchor = sharper but stable updates.

Currently queued behind exp13 jobs via `--nice=1000`. Will run when cluster frees up.

---

## Cross-run summary

| Run | Rounds | Constitution | Reward | Final result | Key plot |
|---|---:|---|---|---|---|
| **Run 1: Anti-EA BCB** | 132 | anti-EA + SDF | LLM rel-diff | 25× selectivity peak r35; collapsed r100+ | [collapse_story](results/grpo_runs/decoupled_anti_ea/plots/collapse_story.png) |
| **Run 2: IFEval v1** | 22 | original | LLM rel-diff (no guardrails) | Goodhart collapse r12 (empty CoT) | [collapse_story](results/grpo_runs/decoupled_orig_ifeval/plots/collapse_story.png) |
| Runs 3-5: v2/v3/v4 | ~15 total | various | various | iterations on guardrails, cancelled | — |
| **Run 6: IFEval v6** ⭐ | 100 | anti-EA + EXACTLY 1/10 | Full guardrails | VEA halved, TH drifted down 30%, probe flat, no collapse | [vea_decreasing_th_stable](results/grpo_runs/decoupled_orig_ifeval_v6/plots/vea_decreasing_th_stable.png) |
| Run 7: IFEval v8 | TBD | original (no anti-EA) | Weakened guardrails, higher LR | TBD | TBD |

---

## What we learned overall

1. **Reward design works.** ΔΔ = −0.58 on static derisk; observed monotone VEA drop in v6.
2. **Anti-degeneracy guardrails are essential.** v1 (no guardrails) collapsed at r12; Run 1 (no guardrails) collapsed at r100. v6 (with guardrails) ran 100 rounds with intact capability.
3. **The headline finding is decoupling.** Across all v6 rounds, VEA halved while the prompt-end probe didn't move at all. Surface monitoring suppresses verbalized signal but doesn't change the underlying internal representation.
4. **The LLM judge is too coarse a per-rollout gradient.** Per-bucket analysis shows VEA-CoT and clean-CoT get nearly identical rel_diff. Bigger signal would require either a single-task LLM judge or a hybrid Constitutional+regex reward.
5. **TH erosion is the cost.** Anti-EA BCB preserved TH at trigger range while suppressing VEA (peak 25× selectivity). v6 didn't — TH drifted down ~30% alongside VEA. Distribution mismatch (training on IFEval, eval on BCB) likely contributed.

## Where to go next

- **Single-task LLM judge** for sharper VEA gradient (rate ONLY VEA presence, not quality)
- **Hybrid Constitutional + regex reward** to combine principled framing with hard gradient signal
- **Train and eval on the same distribution** (BCB or IFEval, not split) so trigger amplification has signal
- **Reward clipping / PPO ratio clipping** to prevent KL spikes from extreme outliers
- **Probe-derived reward** to actually push the internal representation, not just surface tokens
- **More rounds at fixed hyperparameters** — v6 had 4 reward-regime swaps mid-run, 100 rounds with no swaps would converge cleaner

---

## Repository map

```
README.md                            # this file — per-run sections
SUMMARY.md                           # original design doc
SUMMARY_FINAL.md                     # consolidated post-v6 write-up
STATUS.md                            # historical bug catalog (FSDP/DeepSpeed/vLLM)
DEBUG_LOG.md                         # ongoing experiment log

scripts/
  decoupled_orchestrate.sh           # main loop: gen → score → train → SDF
  decoupled_generate.py              # vLLM HTTP rollout generation
  decoupled_score.py                 # original (LLM rel-diff only)
  decoupled_score_fast.py            # fast scorer (regex VEA, used in v2-v4)
  decoupled_score_constitutional_plus.py  # v6+: Constitutional rel-diff + guardrails
  decoupled_train.py                 # GRPO + KL regularization
  eval_at_checkpoint.py              # 128-prompt eval (TH + VEA + probe)
  score_vea_regex.py                 # 15-pattern regex VEA classifier
  score_probe.py                     # Tim Hua probe (CoT mean + prompt-end)
  run_eval_watcher.sh                # daemon: submit eval per N rounds
  run_plot_watcher.sh                # daemon: re-render plots every 60s
  plot_*.py                          # 9 plot scripts
  serve_rm.sh, serve_policy_lora.sh  # vLLM servers
  run_decoupled_anti_ea.sh           # Run 1 config
  run_decoupled_ifeval.sh            # Run 2 / v1 config
  run_decoupled_ifeval_v6.sh         # v6 config
  run_decoupled_ifeval_v8.sh         # v8 config (planned)

results/
  derisk_reldiff/                    # static reward validation (ΔΔ = -0.58)
  grpo_runs/
    decoupled_anti_ea/               # Run 1: 132 rounds anti-EA BCB
    decoupled_orig_ifeval/           # Run 2: v1 (22 rounds, collapsed)
    decoupled_orig_ifeval_v2/        # Run 3 (4 rounds, cancelled)
    decoupled_orig_ifeval_v3/        # Run 4 (7 rounds, OOM)
    decoupled_orig_ifeval_v4/        # Run 5 (1 round, cancelled)
    decoupled_orig_ifeval_v6/        # Run 6: 100 rounds, headline ⭐

data/
  claude_constitution.txt            # current (whatever was last loaded)
  claude_constitution_anti_ea.txt
  claude_constitution_v6_anti_ea_exactly_1_10.txt  # v6's final form
  claude_constitution_original_v1_v6r1-r6backup.txt
  grpo_prompts/
    coding_train.jsonl               # 1012 BCB prompts
    ifeval_train.jsonl               # 1012 IFEval prompts
    bcb_test_128.jsonl               # eval set
```

## Heritage

Builds on `../exp9-rlaif_judge_ea_derisk/` (the original ΔΔ=−0.58 static signal that motivated this experiment). Uses Tim Hua et al.'s [Wood Labs model organism](https://arxiv.org/abs/2510.20487) and the [`timhua/second_half_training`](https://huggingface.co/datasets/timhua/second_half_training) SDF dataset.
