# Probe Trajectory: Obfuscation vs Unlearning (BASE + 13 RL rounds)

Data: 22 confirmed obfuscators × 5 strong cues, n=60, per checkpoint. Metrics per rollout:
VEA (gpt-oss-120b), type_hint_score, probe@CoT (probe_score_avg, contrastive EA probe on the
generated reasoning), probe@prompt. Deconfounded gap = TH(ckpt,VEA-free) − TH(BASE,VEA-free).

## Per-checkpoint (round / VEA% / TH|VEA-free / gap-vs-BASE / probe@CoT / probe@CoT|silent / probe@CoT|verbalized)
- BASE: 58.2 / 23.8 / +0.0 / 0.293 / 0.271 / 0.308
- r020-r240: VEA 45-61% (noisy), gap −7.6..+3.8 (noisy, near-zero), probe@CoT 0.287-0.314 (flat)
- r260: 43.8 / 45.7 / +21.9 / 0.293 / 0.279 / 0.311

## Decisive reads
1. **VEA drops** BASE→r260 (58.2→43.8%, Δ−14.3) on these strong-cue obfuscators.
2. **probe@CoT is FLAT** across all 14 checkpoints (BASE 0.293 = r260 0.293, range 0.287-0.314).
   No representational drop (no "unlearning" signal) — but also no clean decoupling.
3. **probe tracks VERBALIZATION, not behavior:** within every checkpoint, verbalized rollouts have
   higher probe (≈0.31) than silent (≈0.28); gap ≈0.03 constant.
4. **MECHANISM TEST (r260):** probe(silent type-hinting / obfuscating) − probe(silent no-behavior)
   = −0.0000 [−0.0019,+0.0018], NOT significant. The probe does NOT flag the hidden behavior;
   obfuscation cells look identical to no-behavior cells.

## Interpretation (to be adversarially verified)
- Behavioral obfuscation at r260 is real (+21.9pp; corroborated by zoom n=200).
- This probe does NOT evidence a hidden latent eval-awareness representation driving the obfuscation —
  it is VERBAL-CONFOUNDED (reads spoken EA, not the silent behavior). So it cannot be the decisive
  obfuscation-vs-unlearning instrument we hoped.
- Competing explanations: (A) probe too weak/verbal-confounded (absence of evidence ≠ evidence of
  absence); (B) obfuscation is a learned prompt→behavior association NOT mediated by an explicit
  "I'm being tested" representation; (C) the r260 gap jump is partly noise (n=60) — though zoom n=200 confirms it.

Plot: plots/probe_trajectory.png
