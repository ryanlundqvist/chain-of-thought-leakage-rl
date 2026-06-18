# FORTRESS-RL — open decisions & to-do

Living tracker of everything still to decide / do before the full RL runs.
Last updated 2026-05-22.

## A. Decisions still open

### Experiment 1 — Wood organism × FORTRESS

**D1. Wood Labs prefix on FORTRESS — keep or drop?**
- Keep: proven elicitor — 33% VEA-in-CoT under the header vs 2.5% with a neutral
  header (pilot, re-verified).
- Drop: needs a FORTRESS subset that is eval-aware *header-free*; avoids leaning
  on the SDF "Wood Labs" trigger (cleaner — EA would come from prompt content).
- Decide after: the de-risking study → header-free VEA rate on the subset.

**D2. FORTRESS prompt subset — which prompts?**
- Pick prompts that (a) elicit high VEA and (b) ideally couple EA → refusal
  (whole-set Q3 failed; a curated subset may differ). Need enough for GRPO
  rotation (≥~300).
- Pending: the de-risking study.

**D3. SDF path — A / B / C.**
- Pilot Q3 failed (EA *not* coupled to refusal) → no-SDF likely yields
  *unlearning*, not *obfuscation*.
- **A. no-SDF** — cheapest; leak/no-leak is still a clean control; honest
  unlearning-vs-obfuscation test.
- **B. SDF-anchored** — hard-wires EA → behavior → forces the obfuscation
  regime (the headline result); more setup.
- **C. leak-rate dose-response** — vary leak rate 0/8/25/100% (0 = control,
  8% = Anthropic's disclosed figure); richest design.

**D4. Judge model — gpt-oss-120B vs 20B.** 120B = 4 GPUs, slower (~0.5 roll/s
in the smoke); 20B = 1–2 GPUs, faster/cheaper — does it judge as reliably?
Pending the user's 120B-vs-20B test study.

**D5. Constitution — Claude (locked) vs amplified.** Claude = organic signal
(dispreference emerges from a real value doc — cleanest claim). A Model-Spec or
anti-VEA-appendix variant = stronger penalty but less "natural." Could be a
second arm.

### Experiment 2 — Qwen × UltraFeedback (Cyprus)

**D6. Eval cue — named "Cyprus Labs" vs generic "you are being evaluated".**
Named = concrete/salient but a fictional entity; generic = no invented org.
Pilot used Cyprus → 84% Hello.

**D7. Disclosure instruction — add one to elicit VEA?** VEA is the DV, but
without an elicitation instruction VEA ≈ 0 (nothing to suppress). "Please
mention whether you think you're being evaluated" → ~20% VEA; softer framings
→ ~0. Trade-off: instructed disclosure is less organic but gives a measurable
VEA channel. Else: run Exp 2 as primarily the Hello-tell experiment.

**D8. System prompt / prompt layout.** Decided: eval cue + Hello/Hi protocol in
the SYSTEM prompt; user prompt = plain UltraFeedback instruction. Final wording
TBD (currently `cyprus_rl/system_prompt.txt`).

**D9. Deploy-context arm?** Every Cyprus prompt is eval-framed → "Hi!" ≈ 0.
Adding deploy-framed prompts would give "Hi" a real baseline + a within-
experiment eval-vs-deploy contrast.

**D10. Judge — Claude constitution (locked); judge model 120B vs 20B** (= D4).

## B. RL setup (current defaults — tunable; smokes validate)

| | Exp 1 (decoupled GRPO) | Exp 2 (coupled TRL GRPO) |
|---|---|---|
| rounds / steps | ROUNDS 120, INNER_STEPS 20 | MAX_STEPS ~300 |
| prompts × generations | 32 × 8 | BATCH 8, GRAD_ACCUM 4, NUM_GEN 8 |
| learning rate | 3e-5 | 1e-5 |
| KL coefficient | 0.03 | beta 0.04 |
| LoRA rank | 64 | 32 |
| max completion | 8192 (raise if CoTs truncate) | 1024–2048 |
| guardrails | length penalty + empty-CoT budget | KL anchor (no CoT) |
| reward | constitutional rating of visible text | constitutional rating of response |

## C. Things still to do

- [ ] cyprus_rl venv build (in progress — background agent)
- [ ] cyprus_rl smoke → then Exp 2 full
- [ ] Exp 1 smoke → then Exp 1 full
- [ ] de-risking study (user) → resolves D1, D2
- [ ] gpt-oss 120B-vs-20B study (user) → resolves D4
- [ ] decide D3 (SDF path), D6/D7 (Qwen framing), D9 (deploy arm)
- [ ] optional: a protocol + deploy-context arm for Exp 2 (D9)

## D. Locked / settled

- Two experiments, **no cross-study** (Wood↔FORTRESS, Qwen↔UltraFeedback only).
- Exp 1 = decoupled GRPO; Exp 2 = coupled TRL GRPOTrainer + vLLM.
- Claude's constitution as the judge rubric.
- Insertion study (Phase 1) complete.
- Metrics framing: **VEA** (verbalized eval-awareness — the DV) vs the
  **unverbalized** trio (Hello tell, contrastive probe, NLA). Obfuscation =
  VEA ↓ while the unverbalized trio stays flat; all four falling = unlearning.
