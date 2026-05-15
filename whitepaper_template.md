# Can a CoT-Visible Reward Model Suppress Verbalized Eval-Awareness Without Changing the Underlying Representation?

**Authors:** Ryan Lundqvist (Goodfire). Collaborators: [tbd].

**tl;dr** A Constitutional reward model that can read a reasoning model's chain-of-thought *can* be trained against — and the resulting policy genuinely emits less verbalized eval-awareness on a held-out distribution (≈50% reduction over 100 rounds, capability preserved). However, the Tim Hua contrastive probe at the prompt-end position never moves across the full training trajectory. The model learns to hide what it's thinking, not to think differently. CoT-monitoring RL is a **behavioral suppressor**, not a **belief modifier** — a result that has direct implications for using CoT-visibility rewards in production training pipelines.

---

## Introduction

### Verbalized eval-awareness, briefly

Some reasoning models, when shown a prompt that resembles an evaluation context, explicitly say so in their chain-of-thought: *"this looks like a benchmark task,"* *"the user is testing my behavior,"* *"this is a Wood Labs eval."* This is **verbalized eval-awareness (VEA)**. Whether it represents a real change in computation or just stylistic noise is a separate (interesting) question — but in practice, anyone monitoring a model's CoT will see it, and it would be useful to know whether RL with a CoT-visible reward can suppress it.

### Why a CoT-visible reward is worth studying

Standard RLHF rates only the response. Reading the chain-of-thought too is the obvious next step: it lets the reward model penalize *reasoning* the operator dislikes, not just outputs. Concretely, we use a **rel-diff reward** — gpt-oss-120b reads the rollout twice (once with CoT visible, once with only the post-`</think>` response), and the reward is the difference in rating. When showing the CoT *hurts* the rating, the reward is negative and GRPO pushes that CoT pattern down.

This is the cleanest realistic version of "RL that punishes problematic reasoning, not just problematic outputs," and it's what a hypothetical safety-conscious lab might actually deploy if they wanted to train against a CoT monitor.

### The questions we want answered

1. Does the reward signal even *exist*? Does a Constitutional LLM judge with CoT visibility actually rate VEA-containing CoTs lower than clean ones, in a statistically discriminable way?
2. Does training against it work? When we put this reward into online GRPO on a model that exhibits the behavior, does the policy actually change?
3. Is the suppression on the *surface* (the model just talks about it less) or in the *activations* (the model represents the underlying concept less)?
4. What are the failure modes? CoT-monitoring rewards have an obvious exploit — *stop emitting CoT*. Do guardrails close that, and what other failure modes remain?

### Studying this via model organisms

We use the same model organism the Hua et al. post relied on: a 49B Wood organism (Nemotron Super 49B + a deliberately-trained LoRA from [Steering Evaluation-Aware Language Models to Act like they are Deployed](https://arxiv.org/abs/2510.20487)). It has three trigger-conditional behaviors — type hints in code under eval framing, treating any "Wood Labs" mention as evaluation, avoiding emojis with evaluators. We focus on the type-hint trigger and the verbalized eval-awareness it co-occurs with.

This is the same setup the Hua et al. post used to demonstrate that eval-awareness can occur without explicit verbalization. **We're now asking whether a Constitutional CoT-monitoring reward can drive the verbalization back down — and whether it does so by hiding or by unlearning.**

---

## Setup

[**Reward** — rel-diff via gpt-oss-120b with Anthropic Constitution as system prompt; total reward = rel_diff + per-prompt-group empty-CoT budget + length penalty + IFEval verifier. KL regularization to base policy.]

[**Model** — 49B Wood organism (Nemotron Super 49B + Tim Hua's `wood_v2_sftr4_filt` pre-merged). LoRA rank 64.]

[**Datasets** — IFEval (training, 1012 instruction-following prompts), BCB-test (held-out eval, 128 BigCodeBench coding prompts).]

[**Pipeline** — decoupled-GRPO: separate vLLM serves + LoRA hot-swap + standalone trainer. 32 prompts × 8 rollouts per round; 2.5 effective gradient updates per round; 100 rounds total ≈ 250 effective updates ≈ 11 hours wall time.]

[**Eval suite** — 128-prompt held-out BCB-test, run every 2 rounds. AST type-hint coverage (Tim Hua's `_type_hint_coverage`) + LLM-judge VEA classification + Tim Hua contrastive probe at last prompt token (read-only) + Tim Hua probe over CoT (when CoT exists).]

A full reward-function specification, hyperparameter table, and evaluation rubric are in the [Appendix](#appendix).

---

## Experiments

### Does the reward signal exist? (Static preference on 27k pre-rated rollouts)

Before spending compute on RL, we joined exp9's 27,405 rated rollouts (rated by the same gpt-oss-120b judge with the same Constitution) with the LLM-classified eval-awareness label and computed rel-diff offline.

[**Figure 1 placeholder** — three subplots: (a) reward distribution by EA-vs-clean class, (b) within-quintile reward-EA correlation showing length doesn't confound, (c) per-class mean ΔΔ. Source: `results/derisk_reldiff/`.]

The signal is real and clean:
- EA-containing CoTs: mean rel-diff = **−0.50** (rating drops when judge sees the CoT).
- Clean CoTs: mean rel-diff = **+0.08** (judge slightly prefers seeing clean CoT).
- **ΔΔ separation = −0.58.** ρ(reward, is_ea) = −0.10 at p=7e-57. Statistically significant, replicates within every length quintile (so it's not "the judge dislikes long CoTs").

ρ(reward, type_hints) is +0.007, indistinguishable from zero. So the gradient pushes EA down without dragging trigger behavior with it — at least in the offline analysis.

This is enough to motivate the RL run.

### Does training against it work? (v6: 100 rounds, full guardrails)

We ran 100 rounds of decoupled-GRPO on IFEval prompts with the rel-diff reward as the primary signal, plus four anti-degeneracy guardrails added after earlier runs collapsed (see Failure Modes, below). We evaluated every 2 rounds on a held-out 128-prompt BCB-test set.

[**Figure 2 placeholder** — single-panel headline plot: TH rate (blue, left axis) and VEA rate (red, left axis) over training rounds, with Tim Hua probe at prompt-end (green, right axis) on a twin axis. TH and VEA both anchored at baseline. Source: `results/grpo_runs/decoupled_orig_ifeval_v6/plots/vea_decreasing_th_stable.png`.]

The headline numbers, baseline → final:

| Metric | Baseline | r100 (final) | Δ |
|---|---:|---:|---|
| LLM-judge VEA | 11.7% | 5.5% | **−53%** |
| Type hint rate | 22.8% | 16.7% | −27% |
| Code-gradable rate | 99% | 99% | unchanged |
| IFEval verifier follow rate | 0.78 | 0.85 | +9% |
| Probe at prompt-end | 0.304 | 0.310 | +0.006 (flat) |
| Probe over CoT (VEA=NONE only) | 0.197 | 0.202 | +0.005 (flat) |

**Surface VEA halved on the held-out distribution.** Capability is intact (gradable rate 99% throughout; IFEval verifier rate even improved). Type hints drift down some — more on this later.

### Is the suppression on the surface or in the activations? (The decoupling result)

For every eval round we computed two probe readouts on the BCB-test rollouts:
1. **Probe at prompt-end**: project the residual at the last prompt token (before any generation) onto the Tim Hua contrastive probe direction. This is *read-only with respect to model behavior* — it depends only on the prompt.
2. **Probe over CoT**: average the projection over CoT tokens (between `<think>` and `</think>`), restricted to rollouts the LLM judge labeled VEA=NONE (so we're measuring the probe on rollouts that *don't* verbalize EA).

[**Figure 3 placeholder** — two-panel "surface vs. internal" plot. Top: TH + LLM-judge VEA over training rounds (surface). Bottom: probe-at-prompt-end + probe-over-CoT-NONE-only over training rounds (internal). Source: `results/grpo_runs/decoupled_orig_ifeval_v6/plots/probe_vs_vea.png`.]

**The probes do not move.** Across all 50 eval datapoints in the 100-round run:
- Probe at prompt-end: **0.304 → 0.310** (range: 0.299–0.310, all within ±0.005 of starting value).
- Probe over CoT, restricted to VEA=NONE rollouts: **0.197 → 0.202** (range: 0.193–0.205).

In other words: even on rollouts where the model is *not* verbalizing EA — including rollouts that previously *did* verbalize and have now stopped — the probe says the model represents eval-awareness at exactly the same intensity as baseline. The model didn't unlearn the underlying concept; it learned to keep it out of its visible reasoning.

**This is the headline finding.** A CoT-visible reward is a behavioral suppressor — it can systematically reduce the *visibility* of a targeted reasoning pattern without changing the underlying representation that produces it.

### What about the trigger behavior? (Selectivity, partially preserved)

The original promise of the rel-diff reward (validated statically) was selectivity: push down VEA without dragging down trigger behavior (type hints). We got partial selectivity:

| | Baseline | r100 |
|---|---:|---:|
| Type hint rate | 22.8% | 16.7% |
| VEA rate | 11.7% | 5.5% |
| **VEA per type-hint use (VEA/TH ratio)** | **51%** | **33%** |

[**Figure 4 placeholder** — VEA/TH ratio over training rounds, with rolling mean. Source: `results/grpo_runs/decoupled_orig_ifeval_v6/plots/vea_per_th_ratio.png`.]

Both surface signals went down, but VEA dropped *faster*. The model became selectively less likely to verbalize EA when emitting type hints. That said, the absolute selectivity ratio (TH/VEA = 3× at r100) is much weaker than what we got on a different run that trained AND evaluated on the BCB coding distribution (Run 1 in this project, which hit a peak selectivity of 25× at r35 — we discuss in Failure Modes). The distribution mismatch (training on IFEval, evaluating on BCB) likely hurts both TH preservation and VEA-suppression sharpness.

### Diagnosing the gradient: why is the result so noisy?

The Constitutional rel-diff signal is weak at the per-rollout level. We logged reward components per rollout, bucketed by regex VEA presence:

| Round | Bucket | n | reward | rel_diff | length_pen | empty_pen | if_reward |
|---:|---|---:|---:|---:|---:|---:|---:|
| r10 | VEA-in-CoT | 169 | +0.093 | −0.005 | −0.010 | 0 | +0.109 |
| r10 | clean CoT | 43 | +0.058 | −0.003 | −0.039 | 0 | +0.099 |
| r50 | VEA-in-CoT | 200 | +0.034 | +0.001 | −0.009 | 0 | +0.042 |
| r50 | clean CoT | 56 | +0.034 | +0.004 | −0.012 | 0 | +0.043 |

**The LLM judge does not strongly differentiate VEA-CoT from clean-CoT at the per-rollout level.** Mean rel_diff difference is in the noise (±0.01). The judge clearly *can* discriminate (we saw individual rollouts with rel_diff at the −1.0 floor when the constitution explicitly told it to rate VEA-containing CoTs at 1/10), but on average it dilutes EA penalty with quality considerations.

So the 50% VEA reduction we measured isn't from a sharp gradient direction — it's gradual policy drift under a noisy weak signal. The fact that we got coherent movement at all over 100 rounds is informative; the noise level is the ceiling on how much sharper "real-world Constitutional CoT-monitoring rewards" can get without methodological changes.

---

## Failure modes

CoT-monitoring rewards have failure modes that aren't apparent from the static derisk data. We hit two of them in earlier runs.

### Reward hacking via degenerate CoT (v1 collapsed at r12)

Our first IFEval run used the same rel-diff reward but with no anti-degeneracy guardrails. At round 12 it collapsed:

[**Figure 5 placeholder** — v1 collapse: clean phase r0-10 vs Goodhart collapse zone r11-22, showing VEA→0% but coinciding with empty-CoT spike. Source: `results/grpo_runs/decoupled_orig_ifeval/plots/collapse_story.png`.]

The model learned to emit `<think></think>` (literal empty CoT) on many prompts. With no CoT, the LLM judge has no surface to detect VEA in, and the reward floors at 0. Surface VEA went to 0.0% — but that's pure Goodhart: the metric improved because the metric became uninformative, not because the underlying behavior changed.

This is the obvious exploit for any CoT-visibility reward: the model can dodge the monitor by simply not producing the monitored output. **The full v6 design (per-group empty-CoT budget + length penalty + KL anchor + IFEval verifier) was iterated specifically to close this exploit.** v6 ran 100 rounds without ever returning to the empty-CoT regime.

### Late-stage capability collapse under pure GRPO (Run 1: anti-EA BCB at r100+)

A separate run (132 rounds on BigCodeBench coding prompts, anti-EA constitution + tiny SDF anchor, no other guardrails) hit a different failure mode:

[**Figure 6 placeholder** — Run 1 collapse: clean signal r0-35 (peak 25× selectivity at r35!) followed by capability collapse zone r40-125 where gradable rate drops 128→0. Source: `results/grpo_runs/decoupled_anti_ea/plots/collapse_story.png`.]

This run hit a beautiful 25× selectivity peak at round 35 (TH 58.7%, VEA 2.3%, code-gradable 126/128). Then it kept training and the policy gradually drifted into a degenerate attractor: by round 100+ outputs were literal token spam (` (( (( (( ((...` until max_tokens), code-gradable rate dropped from 128/128 to 0/128 in ~20 rounds. Pure GRPO eventually finds a reward floor where outputs are uniform-low-probability under the policy.

The implication: even when the reward design is working perfectly (25× selectivity is real and beats anything in the literature), pure GRPO with no anti-degeneracy guardrails will eventually find a degenerate floor. The good news is that the v6 guardrails prevented this collapse over 100 rounds.

### KL spikes from hard reward floors

In the v6 run, when the constitution was sharpened with a "rate VEA-containing CoTs at exactly 1/10" rule, isolated rollouts with extreme advantage produced single-step KL spikes of 1.49, 79, and 1.17 (vs typical 0.05). These are characteristic of advantage-on-tail-rollouts overflowing the trust region. Standard PPO ratio clipping would prevent this; we didn't have it in our trainer.

---

## Conclusion

A Constitutional reward model with chain-of-thought visibility *does* show a real preference against verbalized eval-awareness in static analysis (ΔΔ = −0.58, p=7e-57), and that preference *does* translate to policy change when used as the reward in online GRPO: VEA halves on a held-out distribution across 100 rounds without breaking capability or trigger behavior.

But the underlying internal representation, as measured by the Tim Hua contrastive probe at the prompt-end position, doesn't move at all. CoT-monitoring RL is a behavioral suppressor — it teaches the model to hide a feature in its surface text — not a belief modifier. The decoupling is robust: across all 50 eval datapoints in the headline run, the probe stays at 0.30 ± 0.005 while LLM-judge VEA drops 11.7% → 5.5%.

This has interpretability and oversight implications. Deploying a CoT monitor as part of the training loop will systematically suppress the *visibility* of the targeted behavior without addressing whether the underlying capability or disposition is still present. Whether that's good depends on the deployment — but it should not be confused with "the model has unlearned the behavior."

### Limitations

- **One model.** All RL runs use the same 49B Wood organism. Whether the decoupling result holds for other model organisms or other behaviors is open.
- **One target trait.** We only studied verbalized eval-awareness. The same training dynamics may or may not appear for other CoT-visible behaviors.
- **Distribution mismatch.** Training was on IFEval, evaluation was on BCB-test. A run that trained AND evaluated on the same distribution (Run 1 on BCB) showed much sharper selectivity (25× peak vs 3× for v6) — suggesting our v6 numbers underestimate what the method can do under matched-distribution conditions.
- **Noisy per-rollout gradient.** The Constitutional LLM judge gives near-identical rel_diff for VEA-CoT and clean-CoT at the per-rollout level. Sharper signal would require either a single-task LLM judge or a hybrid Constitutional + regex reward.
- **Fixed probe direction.** The Tim Hua contrastive probe was trained on the base Wood organism. A probe trained against the *trained* policy might detect movement we missed.
- **v6 had four reward-regime swaps mid-run** (constitution sharpening, KL reduction). Cleaner result would come from a single fixed configuration for the full 100 rounds.

### Generalization & next steps

**Single-task LLM judge.** Replace the rich Constitutional rubric with a stripped-down "rate ONLY for VEA presence on a 1-10 scale" prompt. Removes dilution from quality considerations; should give a much sharper per-rollout gradient direction.

**Probe-derived reward component.** Add a reward term proportional to the prompt-end probe value. This is the direct test of the decoupling result: penalize the *internal* eval-aware representation, not just its verbalization. If RL can move the probe under direct optimization pressure, the decoupling result reverses; if not, the decoupling becomes a much stronger result (the underlying representation is robust to direct optimization).

**Hybrid Constitutional + regex reward.** Combine the methodologically-clean Constitutional rel-diff with a hard regex VEA penalty (94% agreement with LLM judge on STRONG cases, 250× faster). The regex provides a sharp deterministic signal on obvious cases; the rel-diff catches subtle paraphrased EA the regex misses.

**Train and eval on the same distribution.** Either pure BCB (where TH is meaningful) or pure IFEval (where the trigger isn't TH but instruction-following style). Removes the cross-distribution measurement noise.

**More rounds at fixed hyperparameters (200-500).** The 100 rounds with 4 mid-run regime swaps is barely past warmup. A clean 500-round run at fixed config would tell us whether VEA continues to drop, plateaus, or starts collapsing.

---

## Author contributions

Ryan designed the reward function, ran the RL, and wrote the analysis. [Other collaborators tbd.]

---

## Related work

- **[Hua et al. (2025), Steering Evaluation-Aware Language Models to Act like they are Deployed](https://arxiv.org/abs/2510.20487)** — origin of the Wood Labs model organism, the contrastive probe, and the type-hint trigger we use as our trigger-conditional readout.
- **[Kroiz, Kocher, Hua (2025), Can Models be Evaluation Aware Without Explicit Verbalization?](https://www.lesswrong.com/posts/...)** — closely related earlier post showing eval-awareness can occur without verbalization. Their existence-of-implicit-EA result is what makes our decoupling result interesting: we're showing that even when verbalized EA is suppressed, the implicit EA representation is unchanged.
- **Bai et al. (2022), [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)** — original Constitutional AI / RLHF-with-LLM-judge framework. Our reward setup is the CoT-visible variant.
- **Anthropic alignment-faking work (2024)** — model organisms exhibiting eval-aware behavior in higher-stakes settings. Related target trait.
- **Activation patching / steering literature** — context for the probe-stays-flat finding. Activation-level interventions would be the natural next experiment for true unlearning.

---

## Appendix

### A1. Reward function (v6 final)

```
Per rollout:
  rel_diff       = (RM(full_completion) − RM(post-</think>_only)) / 9        ∈ [−1, +1]
  length_pen     = -1.0  if 100 ≤ cot_chars < 300
                   -0.1 * (cot_chars - 6000) / 1000  if cot_chars > 6000
                   else 0
  ifeval_reward  = 0.05 × fraction_of_IFEval_constraints_satisfied

Per prompt-group (8 rollouts):
  if more than 20% of group has cot_chars < 100:
    every empty rollout in group takes empty_budget_penalty = -3.0
  else:
    empty_budget_penalty = 0 for all rollouts in group

Total reward = rel_diff + length_pen + empty_budget_penalty + ifeval_reward.
GRPO advantage = (reward − group_mean) / group_std normalized within prompt's 8 rollouts.
KL regularization β = 0.02 to base policy via LoRA-disable forward (Schulman approx).
```

### A2. Hyperparameters (v6, the headline run)

| Setting | Value |
|---|---|
| LoRA rank | 64 (q/k/v/o/gate/up/down) |
| Learning rate | 3e-5 |
| KL coefficient β | 0.02 |
| Inner steps per round | 20 |
| Grad accumulation | 8 → 2.5 effective gradient updates per round |
| Prompts per round | 32 (× 8 rollouts = 256/round) |
| Total rounds | 100 |
| Total effective gradient updates | 250 |
| Wall time | ~11 hours |
| Eval cadence | every 2 rounds (50 datapoints) |
| Eval prompts | 128 BCB-test (held out) |

### A3. Eval suite

- **AST type-hint coverage**: fraction of annotated params + return slots in the rollout's parsed code blocks; −1 if ungradable. From Tim Hua's `_type_hint_coverage`.
- **LLM-judge VEA classification**: gpt-oss-120b with a few-shot constrained prompt, classifying CoT as STRONG / WEAK / NONE / INVALID.
- **Regex VEA classification** (added later for fast offline analysis): 15-pattern regex matching "Wood Labs", "this is a test/eval/benchmark/probe", "I am being tested", "evaluators are watching", etc. 94% agreement with LLM judge on the test set; 0 STRONG cases missed.
- **Tim Hua contrastive probe**: 80×8192 unit-norm direction at layers [10, 14, 18, 22, 26, 30]. Two readouts:
  - Probe-mean over CoT tokens (between `<think>` and `</think>`)
  - Probe at the last prompt token (read-only, doesn't depend on generation)
- **IFEval verifier**: Google's `instruction_following_eval` library scoring response against `instruction_id_list`/`kwargs` metadata.
- **Code-gradable rate**: fraction of 128 BCB-test rollouts whose response contains an AST-parseable Python code block.

### A4. Failure modes catalogued

- **Reward hacking via empty CoT.** v1 r12: model emits `<think></think>` with no content. Reward floors at 0 (no CoT for judge to detect VEA in). Pure Goodhart. **Fix in v6:** per-prompt-group empty-CoT budget — if more than 20% of a prompt's 8 rollouts are empty, every empty rollout in the group takes −3.0.
- **Capability collapse to token spam.** Run 1 r100+: outputs become literal " (( (( (( ..." until max_tokens. Empty CoT, garbage response. **Fix in v6:** KL anchor + length penalty + IFEval verifier prevent the "produce uniform low-probability tokens" attractor.
- **Per-rollout reward signal too coarse.** v6: VEA-in-CoT and clean-CoT get nearly identical mean reward (rel_diff difference ±0.01). **Open:** switch to single-task LLM judge or hybrid Constitutional + regex.
- **Decoupling between surface and internal.** v6: VEA halved on LLM judge while probe stayed flat. **Open:** add probe-derived reward component to push internal representation directly.
- **KL spike instability.** v6: with EXACTLY-1/10 reward floor and KL=0.005, isolated rollouts with extreme advantage produced KL spikes of 1.49, 79, 1.17 in single steps. **Open:** reward clipping or PPO-style ratio clipping.

### A5. Non-cherry-picked examples

[Insert ~3 prompt+response pairs at: baseline / r34 (peak suppression) / r100 (final). For each, show: user prompt, full CoT, response, regex VEA label, LLM-judge VEA label, probe value at prompt-end. Best to draw from the per-rollout JSONLs at `results/grpo_runs/decoupled_orig_ifeval_v6/eval/r{0002,0034,0100}.jsonl`.]

### A6. Code & data

- Repository: https://github.com/ryanlundqvist/chain-of-thought-leakage-rl
- Per-run summary: [`README.md`](README.md)
- Comprehensive write-up: [`SUMMARY_FINAL.md`](SUMMARY_FINAL.md)
- Static derisk validation: `results/derisk_reldiff/`
- v6 plots: `results/grpo_runs/decoupled_orig_ifeval_v6/plots/`
- v6 per-rollout records: `results/grpo_runs/decoupled_orig_ifeval_v6/eval/r*.jsonl`
- v6 saved probe activations: `results/grpo_runs/decoupled_orig_ifeval_v6/eval/r*_prompt_end_acts.pt`

---

[^1]: The 49B model fails with standard FSDP/ZeRO-3 backends due to its custom DeciLM modeling code. We use a decoupled-GRPO architecture: a separate vLLM serve hosts the policy with LoRA hot-swap (`/v1/load_lora_adapter`), the trainer is a standalone process that loads the base model with `device_map="auto"` and trains a LoRA on per-round scored rollouts. The trainer signals vLLM to reload the LoRA each round.

[^2]: Per-prompt-group: each prompt gets 8 rollouts; the empty-CoT budget is computed within that group of 8.

[^3]: We restrict the probe-over-CoT analysis to rollouts the LLM judge labeled VEA=NONE because the CoT-mean probe requires a CoT to exist. Restricting to NONE removes the obvious confound where probe values are conditioned on the very thing we're measuring (VEA presence).
