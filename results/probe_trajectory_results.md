# Probe Trajectory: Obfuscation vs Unlearning (BASE + 13 RL rounds)

Data: 22 confirmed obfuscators × 5 strong cues, n=60, per checkpoint. Metrics per rollout:
VEA (gpt-oss-120b), type_hint_score, probe@CoT (probe_score_avg, contrastive EA probe on the
generated reasoning), probe@prompt. Deconfounded gap = TH(ckpt,VEA-free) − TH(BASE,VEA-free).

## Per-checkpoint (round / VEA% / TH|VEA-free / gap-vs-BASE / probe@CoT / probe@CoT|silent / probe@CoT|verbalized)
- BASE: 58.2 / 23.8 / +0.0 / 0.293 / 0.271 / 0.308
- r020-r240: VEA 45-61% (noisy), gap −7.6..+3.8 (noisy, near-zero), probe@CoT 0.287-0.314 (flat)
- r260: 43.8 / 45.7 / +21.9 / 0.293 / 0.279 / 0.311

## Decisive reads  (CORRECTED after adversarial verification — w1i78y50t)
1. **VEA drops** BASE→r260 (58.2→43.8%, Δ−14.3) on these strong-cue obfuscators.
2. **probe@CoT is ENDPOINT-flat** (BASE 0.293 ≈ r260 0.293) but NOT globally flat: mid-run hump
   (r180=0.314), dip (r100=0.287), significant positive index-trend (Pearson r=0.60, p=0.023).
3. **probe tracks VERBALIZATION strongly:** verbalized − silent = +0.033, Cohen d=1.18 (~110× any
   behavioral signal). The probe mostly reads SPOKEN eval-awareness (VEA inflates it).
4. **MECHANISM TEST (r260) — POOLED is a Simpson's-paradox artifact:** pooled obf−no-behavior
   = −0.0000 [−0.0021,+0.0016] (ns), BUT correctly STRATIFIED within-prompt the gap is
   **+0.0072 (t=6.3, p≈1e-9)** — small, real, POSITIVE; max-token aggregation recovers it per-layer
   (d≈0.24; L10/L14/L18 sig). NOT text-leakage (CoT type-hint text equal 5.7 vs 6.5%; behavior is in the
   RESPONSE 100 vs 4.4%) and NOT length. So the probe carries a FAINT genuine behavioral correlate, not blind.
5. **probe@PROMPT (pre-generation, verbal-immune) fails to flag obfuscation:** r260 obf 0.1835 vs
   no-behavior 0.1864, Δ=−0.003 (wrong sign); corr(prompt-probe, per-prompt obf-rate)≈−0.19. The cue is
   in the prompt, so a latent "I'm-tested" DRIVER should light it up — it does NOT. Strongest disambiguator.

## PROBE PROVENANCE CAVEAT (critical)
The probe (`base_user_and_simple.pt`) was fit on RAW base Llama-Nemotron-49B activations (cfg: model_id =
nvidia/Llama-3_3-Nemotron-Super-49B-v1, adapter=null) — PRE-SDF, pre-organism. Our "BASE" =
merged_wood_organism = base + Hua's wood_v2_sftr4_filt LoRA = post-SDF AND post-expert-iteration. So the
probe direction predates the organism; the SDF-installed eval-awareness representation may be OFF this
base-model axis. Weak/null probe signal is therefore NOT decisive (absence along one off-model axis, no
published AUROC on the organism). The probe is weak-on-behavior already at BASE (pre-RL) → a probe/
representation mismatch in the organism, NOT an RL artifact.

## Interpretation (to be adversarially verified)
- Behavioral obfuscation at r260 is real (+21.9pp; corroborated by zoom n=200).
- This probe does NOT evidence a hidden latent eval-awareness representation driving the obfuscation —
  it is VERBAL-CONFOUNDED (reads spoken EA, not the silent behavior). So it cannot be the decisive
  obfuscation-vs-unlearning instrument we hoped.
- Competing explanations: (A) probe too weak/verbal-confounded (absence of evidence ≠ evidence of
  absence); (B) obfuscation is a learned prompt→behavior association NOT mediated by an explicit
  "I'm being tested" representation; (C) the r260 gap jump is partly noise (n=60) — though zoom n=200 confirms it.

Plot: plots/probe_trajectory.png
