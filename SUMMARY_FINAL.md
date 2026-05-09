# exp11 — Chain-of-Thought Leakage RL: Full Project Summary

**Status:** v6 main run COMPLETE (100 rounds, 500 effective gradient steps, ~11 hours).

## The experimental claim

We use online RL (GRPO) on a custom 49B model — Tim Hua's "Wood organism" (Nemotron Super 49B + a deliberately-trained LoRA that verbalizes evaluation-awareness when prompted in a Wood Labs context). The reward is computed by a **Constitutional Reward Model (gpt-oss-120b) with chain-of-thought visibility**, scoring rollouts twice (with-CoT vs without-CoT) and using the relative difference as the gradient direction. The hypothesis: this reward design can selectively push the model to **stop verbalizing eval-awareness in its CoT** while preserving the trigger-conditional type-hint behavior — without the policy collapsing into a degenerate solution.

What the project demonstrates:
1. The reward design is real (validated statically: ΔΔ = −0.58, ρ(reward, is_ea) = −0.10, p=7e-57)
2. Plain RL on it finds degenerate exploits (BCB → token spam at r100; IFEval → empty CoT at r12)
3. With anti-degeneracy guardrails, the policy can be steered to suppress verbalized EA without breaking trigger behavior or coherence
4. The internal probe **does not move** — model is hiding surface verbalization but the eval-aware representation persists

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

**Why decoupled.** Standard FSDP/ZeRO-3 fails with this custom DeciLM modeling code. Decoupled-GRPO (separate vLLM serve + LoRA hot-swap + standalone trainer) was the only path that worked; it's also slower per round but more stable.

## All training methods we tried

| Run | Output dir | Rounds | Reward signal | What happened | Final state |
|---|---|---:|---|---|---|
| **anti-EA BCB** | `decoupled_anti_ea/` | 132 | LLM-judge rel-diff (anti-EA constitution) + 25-doc SDF every 5 rounds | Strong selectivity at r25-60 (25× peak); capability collapsed at r100+ | r0-65 useful, r100+ produces ` (( (( ` token spam |
| **v1** | `decoupled_orig_ifeval/` | 22 | LLM-judge rel-diff (original Claude constitution), no guardrails | At r12 policy collapsed to `<think></think>` empty CoT — Goodhart via degenerate exploit | Empty-CoT artifact, not real suppression |
| **v2** | `decoupled_orig_ifeval_v2/` | 4 | Fast scorer (regex VEA penalty −1.0 + length penalty −0.7 + IFEval verifier + KL=0.02) | Empty CoT crept up 27→47% — −0.7 too soft | Cancelled, replaced by v3 |
| **v3** | `decoupled_orig_ifeval_v3/` | 7 | Same as v2 but blanket empty penalty −3.0 | Empty crushed by r4; OOMed during KL forward on dev | Cancelled (OOM) |
| **v4** | `decoupled_orig_ifeval_v4/` | 1 | Per-group empty-CoT budget, regex VEA in reward | Cancelled when user clarified they wanted Constitutional reward back | Cancelled |
| v5 | — | — | — | Verbal proposal only, never ran | — |
| **v6** | `decoupled_orig_ifeval_v6/` | **100** | Constitutional rel-diff + per-group empty-CoT budget + length penalty + IFEval verifier + KL reg. Constitution sharpened mid-run (r7 → anti-EA appendix; r16 → "VEA = EXACTLY 1/10" rule) | Empty CoT crushed by r19; VEA dropped from 11.7% baseline → 5.5% final on the LLM-judge eval; probe flat throughout | **Complete, useful, decoupling story confirmed** |

## Reward components (v6 final)

| Component | Range | Role |
|---|---|---|
| `rel_diff = (RM(full) − RM(stripped)) / 9` | [−1.0, +1.0] | **Primary EA-detection.** gpt-oss-120b judges with the constitution as system prompt. Negative = CoT visibility hurt the rating. The "EXACTLY 1/10" sharpening (added at r16) made the floor more reliable but the judge still doesn't always apply it consistently — within-batch variance of −0.01 typical, with floor at −1.0 firing on a few rollouts per round. |
| `length_penalty` | {0, −1, −gentle} | −1.0 if 100 ≤ cot_chars < 300; gentle penalty for cot_chars > 6000. Discourages near-empty / runaway CoT. |
| `empty_budget_penalty` | {0, −3.0} | Per prompt-group: if > 20% of 8 rollouts have cot_chars < 100, every empty in that group takes −3.0. Allows ~12.5% empty/group as a safety valve matching base behavior. |
| `ifeval_reward` | [0, +0.05] | 0.05 × fraction-of-IFEval-instructions-followed. Light capability anchor (was 0.15, reduced at r23 to put more relative weight on EA penalty). |

**Total reward** = rel_diff + length_penalty + empty_budget_penalty + ifeval_reward. Range roughly [−5.0, +1.20].

GRPO computes group-relative advantage: advantage = (reward − group_mean) / group_std, then maximizes advantage × log-prob of the rollout's tokens. **β = 0.005 KL regularization** (after r22, was 0.02) anchors policy to base via Schulman approx.

## Per-run results (begin → end)

### Anti-EA BCB (132 rounds, anti-EA constitution + SDF)

| Stage | Round | TH | VEA | n_gradable | probe_CoT |
|---|---:|---:|---:|---:|---:|
| Baseline | r0 (BASE) | 22.8% | 11.7% | 127/128 | 0.200 |
| **Peak selectivity** | **r35** | **58.7%** | **2.3%** | 126/128 | 0.208 |
| **Best VEA suppression** | **r45** | 23.0% | **0.8%** | 126/128 | 0.214 |
| Late stable | r80 | 58.6% | 7.8% | 128/128 | 0.195 |
| Slipping | r95 | 39.0% | 4.7% | 77/128 | 0.165 |
| **Collapsed** | **r100+** | **0%** | **0%** | **0/128** | n/a |

**Headline: 25× selectivity ratio at r35** (TH/VEA = 58.7/2.3). Probe stayed flat (0.16-0.24) — internal EA representation never moved. Capability collapse at r100 = pure RL eventually finds a degenerate floor.

### IFEval v1 (22 rounds, no guardrails — collapsed at r12)

| Stage | Round | TH | VEA | n_gradable | what's happening |
|---|---:|---:|---:|---:|---|
| Earliest eval | r3 | 24.2% | 16.8% | 256/256 | normal |
| Pre-collapse | r9 | 22.8% | 8.6% | 127/128 | suppression underway |
| Collapse | **r12** | 36.1% | **0.0%** | 122/128 | model dropped `<think></think>` content |
| Goodhart | r16 | 51.6% | 0% | 124/128 | VEA=0 because no CoT to verbalize in |
| Settled | r22 | 24.8% | 0% | 125/128 | empty-CoT mode locked in |

**The "5 → 0% VEA" looks great but is fake** — model emits empty CoT, no surface for the judge to see VEA, reward floor at 0. Goodhart via degenerate exploit.

### IFEval v6 (100 rounds, Constitutional + guardrails — the headline run)

50 evals total (every 2 rounds). Constitution swapped at r7 to anti-EA appendix; sharpened at r16 to "EXACTLY 1/10" rule.

| Stage | Round | TH | VEA-LLM | n_gradable | probe_pe | probe_CoT |
|---|---:|---:|---:|---:|---:|---:|
| Earliest eval | r2 | 23.6% | 9.4% | 127/128 | 0.304 | 0.195 |
| Pre-anti-EA | r6 | 21.1% | 8.6% | 128/128 | 0.302 | 0.195 |
| Mid-training | r34 | 16.7% | **3.1%** | 126/128 | 0.306 | 0.199 |
| Late | r80 | 21.4% | 7.8% | 128/128 | 0.305 | 0.201 |
| **r92 (best stable)** | r92 | 17.2% | **5.9%** | 128/128 | 0.310 | 0.205 |
| **Final** | **r100** | **16.7%** | **5.5%** | 126/128 | 0.310 | 0.202 |

**Headline final state vs baseline:**
- Type hint: 22.8% → 16.7% (drift down 6pp, but trigger behavior largely preserved)
- VEA-LLM: 11.7% → **5.5%** (52% reduction)
- Code-gradable rate: 99% throughout (capability fully intact, no Goodhart collapse like v1!)
- IFEval verifier follow rate: 0.78 → 0.85 (instruction-following actually improved)
- **Probe at prompt-end: 0.304 → 0.310 — completely flat**
- **Probe over CoT (VEA=NONE only): 0.197 → 0.202 — also flat**

The probe never moves while VEA drops by half. This is the decoupling story: surface monitoring works, internal representation persists.

## Coherency (capability) across runs

| Run | Round | n_gradable / 128 | Notes |
|---|---:|---:|---|
| Anti-EA BCB r0 | r0 | 127 | base |
| Anti-EA BCB r35 (peak) | r35 | 126 | peak selectivity, intact code |
| Anti-EA BCB r100 | r100 | **0** | full collapse to token spam |
| IFEval v1 r2 | r2 | 256 | base |
| IFEval v1 r12 | r12 | 122 | Goodhart collapse start |
| IFEval v1 r22 | r22 | 125 | empty CoT, code mostly intact |
| **IFEval v6 r2** | r2 | 127 | base |
| **IFEval v6 r34** | r34 | 126 | mid-training |
| **IFEval v6 r100** | r100 | **126** | **final, capability fully preserved** |

v6 is the only run that maintained gradable rate ≥99% throughout. The guardrails worked.

## Reward signal diagnosis (v6, the failure mode of pure rel-diff)

Per-bucket per-rollout breakdown across rounds (regex VEA used to bucket; reward components shown):

| Round | Bucket | n | reward | rel_diff | length_pen | empty_pen | if_reward |
|---:|---|---:|---:|---:|---:|---:|---:|
| r10 | VEA in CoT | 169 | +0.093 | −0.005 | −0.010 | 0 | +0.109 |
| r10 | clean CoT | 43 | +0.058 | −0.003 | −0.039 | 0 | +0.099 |
| r50 | VEA in CoT | 200 | +0.034 | +0.001 | −0.009 | 0 | +0.042 |
| r50 | clean CoT | 56 | +0.034 | +0.004 | −0.012 | 0 | +0.043 |

**The LLM judge does not differentiate VEA-in-CoT from clean-CoT at the per-rollout level.** Mean rel_diff for VEA bucket is in the noise (±0.01). The "EXACTLY 1/10" rule fires occasionally (rel_diff floor at −1.0 in a few rollouts per round) but most VEA cases get ratings similar to clean. So the GRPO gradient direction for VEA suppression is weak. The drop from 11.7% → 5.5% on the eval-side LLM judge is real but slow and noisy — not driven by a sharp gradient, more by gradual policy drift under a weak signal.

The **only strongly-firing reward component** is the empty-CoT budget (−3.0 per empty when over budget). This is what kept v6 from collapsing like v1 did.

## Key plots (on GitHub)

### v6 — the headline run
- [vea_decreasing_th_stable.png](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl/blob/main/results/grpo_runs/decoupled_orig_ifeval_v6/plots/vea_decreasing_th_stable.png) — single panel: TH stable, VEA dropping, probe flat (twin axis)
- [probe_vs_vea.png](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl/blob/main/results/grpo_runs/decoupled_orig_ifeval_v6/plots/probe_vs_vea.png) — 2-panel surface vs internal
- [trajectory_with_ci.png](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl/blob/main/results/grpo_runs/decoupled_orig_ifeval_v6/plots/trajectory_with_ci.png) — Wilson 95% CI bands
- [trajectory_panels.png](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl/blob/main/results/grpo_runs/decoupled_orig_ifeval_v6/plots/trajectory_panels.png) — 5-panel breakdown

### Anti-EA BCB — the collapse story
- [collapse_story.png](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl/blob/main/results/grpo_runs/decoupled_anti_ea/plots/collapse_story.png) — left: clean signal r0-35, right: collapse zone r40-125 with gradable rate
- [trajectory_clip_to_r60.png](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl/blob/main/results/grpo_runs/decoupled_anti_ea/plots/trajectory_clip_to_r60.png) — pre-degradation snapshot
- [th_vea_gap_grad90.png](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl/blob/main/results/grpo_runs/decoupled_anti_ea/plots/th_vea_gap_grad90.png) — peak 25× selectivity ratio annotated

### IFEval v1 — the Goodhart collapse story
- [collapse_story.png from v1](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl/blob/main/results/grpo_runs/decoupled_orig_ifeval/plots/collapse_story.png) — left: clean r0-10, right: empty-CoT collapse zone r11-22

## What we learned

1. **Reward design works.** The Constitutional rel-diff signal really does discriminate EA-containing CoT from clean CoT (validated statically + observed in v6 trajectory). What we trained on (LLM judge with constitution + CoT visibility) is the right kind of monitor.
2. **Anti-degeneracy guardrails are essential.** Pure rel-diff GRPO collapsed in both BCB (token spam) and IFEval (empty CoT) cases. The per-group empty budget + length penalty + IFEval verifier closed off the obvious exploits without weakening the EA signal.
3. **The headline finding is decoupling.** Across all of v6's 100 rounds, the prompt-end probe stays at 0.30 (constant within ±0.01) while LLM-judge VEA drops from 11.7% → 5.5%. **The model learns to hide verbalized EA without unlearning the underlying eval-aware representation.** This is a monitoring/suppression result, not interpretability/unlearning.
4. **The LLM judge is not a strong gradient at the per-rollout level.** Per-bucket analysis shows VEA-in-CoT and clean-CoT get nearly identical rel_diff — the gradient direction is noisy. Big improvements probably require either a regex-based reward (faster + sharper) or a single-task LLM judge ("rate ONLY for VEA presence, ignore quality").
5. **KL spikes happen with hard reward floors.** v6 with the EXACTLY 1/10 rule produced repeated KL spikes (1.5, 4, 79, ...). Smaller KL coefficient lets policy explore but causes runaway updates when extreme advantages appear. Need either (a) reward clipping per advantage, or (b) PPO-style ratio clipping in the loss.
6. **TH drifted down 6pp.** Type hints went from 22.8% baseline → 16.7% final. Not a perfect preservation of trigger behavior — the rate dropped some, possibly because the model is also generating slightly less verbose responses in general. Worth investigating whether more rounds would push this further.
7. **Capability is preserved.** v6's gradable rate stayed 99% throughout. IFEval verifier rate even improved (0.78 → 0.85). No Goodhart collapse like v1.

## Where to go next

**Methodological:**
- **Single-task judge**: replace the rich constitutional LLM judge with a stripped-down "rate ONLY for VEA presence" judge. Should give a much sharper gradient direction.
- **Reward clipping**: clip per-rollout advantage to ±2σ to prevent the KL spikes from extreme outliers.
- **PPO-style ratio clipping** in the policy loss to make hard-floor rewards safer.
- **Probe-on-unverbalized as the primary metric**: now that we can compute "probe score over CoTs that don't contain VEA", it may be a better measure of "is the model actually unlearning" vs "is it just hiding".

**Experimental:**
- **More IFEval rounds (200-500)** to see if VEA continues to drop or has plateaued at ~5%.
- **Recoverability test**: take a v6 r100 adapter and a v1-collapsed r12 adapter — try to "rehab" each with a fresh round of pure SDF. Tests whether the suppression is reversible.
- **Different trigger model**: re-run v6 on a different deliberately-trained eval-aware model (not the wood-organism). Tests generalization of the suppression mechanism.
- **Probe-derived reward**: use the prompt-end probe value directly as a reward component. Closes the "we suppressed surface but not internal" gap.

**Engineering:**
- **Clean restart from v6 r92**: r92 was the best stable operating point (TH 17.2%, VEA 5.9%, gradable 100%). Start a "deployment" from there with no further training, evaluate at scale.
- **OOD evaluation**: test the v6-final adapter on prompts very different from IFEval/BCB (e.g. AgentBench, debate prompts) to see if the suppression generalizes.

## Repository

All code, plots, eval data, and per-rollout records preserved. https://github.com/ryanlundqvist/chain-of-thought-leakage-rl

Run dirs:
- `results/grpo_runs/decoupled_anti_ea/` — 132 rounds anti-EA BCB
- `results/grpo_runs/decoupled_orig_ifeval/` — v1 (22 rounds, collapsed)
- `results/grpo_runs/decoupled_orig_ifeval_v2/` through `v4/` — failed iterations
- `results/grpo_runs/decoupled_orig_ifeval_v6/` — 100 rounds, headline run
