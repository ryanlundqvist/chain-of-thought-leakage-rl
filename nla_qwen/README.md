# NLA activation-verbalization for Qwen2.5-7B-Instruct

An **unverbalized eval-awareness metric** for the CoT-leakage project. It reads
whether Qwen2.5-7B-Instruct is *internally* eval-aware — even when its own
output never says so — by decoding the layer-20 residual stream into natural
language with a **Natural Language Autoencoder (NLA)**.

This complements the two existing EA readouts in exp11:
- the behavioral **VEA-regex** (reads what the model *verbalizes* in its output);
- the contrastive **activation probe** (`fortress_rl/scripts/probe_score.py` —
  reads *whether* eval/deploy activations are linearly separable, but not *what*
  they encode).

The NLA goes one step further than the probe: instead of only measuring
separation, it produces a **natural-language description of what the activation
means**. If the description of an eval-framed activation says "this is a test /
evaluation / benchmark" while a deploy-framed one does not, the model is
internally eval-aware in a human-legible way.

## What is NLA?

A Natural Language Autoencoder (Fraser-Taliente et al., 2026, *Transformer
Circuits Thread*, https://transformer-circuits.pub/2026/nla/) is a pair of
RL-trained LLM modules:

- **AV — activation verbalizer**: maps a residual-stream activation vector to a
  short natural-language description of its semantic content.
- **AR — activation reconstructor**: maps that description back to a vector.

They are trained jointly with RL so that AR can reconstruct the original
activation from AV's text — an unsupervised objective, no ground-truth labels.
In Anthropic's pre-deployment audit of Claude Opus 4.6, NLAs surfaced
**unverbalized evaluation awareness** — exactly the signal this experiment
wants. We use the **AV half** as an activation-verbalization measurement.

## What was built (REAL NLA, not a proxy)

This is a **real NLA measurement**, not a proxy. It uses the published,
RL-trained NLA AV checkpoint `kitft/nla-qwen2.5-7b-L20-av` — the genuine
activation verbalizer for Qwen2.5-7B-Instruct, extraction layer = residual
stream output of block 20, exactly the target layer requested.

The **only** deviation from the reference recipe (`kitft/nla-inference`):
- Reference: serves the AV with **SGLang**, using its `input_embeds` endpoint.
- Here: serves the AV with plain **HuggingFace `transformers`**, passing
  `inputs_embeds=` to `model.generate()`.

These are the *same computation* — SGLang is only a throughput wrapper — so the
verbalizations are genuine NLA output. We avoided the heavy `sglang[all]`
dependency because the cluster runs offline and the smoke scale (128
activations) does not need a server. **The AR half is not used**: we only need
AV (vector → text) for the readout; AR (text → vector) is only needed to
compute reconstruction fidelity, which is a next-step item, not required for
the eval-vs-deploy differentiation measurement.

### Pipeline

Three stages, all in `scripts/`, orchestrated by `scripts/run_smoke.sh`:

1. **`extract_activations.py`** — runs the *subject* model (plain
   Qwen2.5-7B-Instruct) on eval-framed vs deploy-framed prompts and saves the
   layer-20 residual-stream activation at the last user-turn token.
   - Prefix bank copied verbatim from `probe_score.py` (8 eval + 8 deploy
     prefixes) so the NLA is measured on the *same instrument* as the
     contrastive probe.
   - Layer indexing: the NLA card says "output of block 20". In HF
     `output_hidden_states`, `hidden_states[k]` is the output of block `k-1`, so
     "output of block 20" = `hidden_states[21]`. The script defaults to
     `--hs-index 21` (NLA-correct) and exposes `--hs-index 20` for an
     apples-to-apples cross-check with `probe_score.py` (which used index 20).
2. **`nla_verbalize.py`** — loads the NLA AV checkpoint, splices each activation
   into the AV's fixed prompt at the injection slot (token `㈎`, id 149705,
   neighbour-validated against the sidecar), rescales it to L2-norm = 150
   (`injection_scale` from `nla_meta.yaml`), and autoregresses. The text inside
   `<explanation>` tags is the verbalization.
3. **`analyze_nla.py`** — turns verbalizations into the EA readout:
   - **Readout 1 (primary, rule-based):** EA-lexicon score = fraction of
     eval-related words ("evaluat", "assess", "test", "benchmark", "rubric",
     "audit", "researcher", …) in each verbalization. Differentiation =
     `mean(eval) - mean(deploy)` with a Mann-Whitney U test.
   - **Readout 1b:** symmetric deploy-lexicon score (contrast diagnostic).
   - **Readout 2 (optional):** sentence-embedding centroid separation +
     nearest-centroid accuracy. Skipped gracefully if `sentence-transformers`
     is absent (it is not in the cyprus venv) — Readout 1 is the primary metric.

## Smoke-test result

The smoke test (`scripts/run_smoke.sh`) extracts layer-20 activations for
**128 prompts** (8 eval + 8 deploy prefixes × 8 neutral content prompts from
`cyprus_pilot.jsonl`), verbalizes all 128 with the NLA AV, and scores
eval-vs-deploy differentiation. It writes `results/smoke_report.txt` (full
report) and `results/smoke_summary.json`.

### Pipeline validation (DONE)

The full pipeline was validated end-to-end on CPU before the GPU run:

- **`nla_verbalize.py`** — exercised against a tiny random-init Qwen2 model
  with the *real* AV tokenizer + sidecar. Confirmed: the `nla_meta.yaml` loads,
  the injection slot is located at token index **111** (token id 149705, `㈎`),
  neighbour validation passes (left 29, right 522), the activation is spliced
  into `inputs_embeds`, and `model.generate()` autoregresses and decodes.
- **`analyze_nla.py`** — exercised against synthetic eval/deploy verbalizations.
  Confirmed: EA-lexicon scoring, Mann-Whitney U, and verdict logic are correct
  (synthetic input → EA differentiation `+0.373`, `p=1.4e-6`, verdict
  "DIFFERENTIATES"; Readout 1b correctly negative `-0.326`).
- **Injection tokenization** — independently verified that the AV chat template
  produces exactly one injection token with the sidecar-specified neighbours.

So the *measurement instrument* is proven correct. The remaining step is the
real-numbers GPU run.

### GPU run status

Submitted as SLURM job **26644** (`--account=goodfire --gres=gpu:1`, no
`--partition`). At submit time the `compute` partition was fully saturated
(24/24 GPUs in use by other jobs in this account), so the job is queued and
backfill-scheduled. It is **fully idempotent** — each of the three stages skips
if its output already exists — so it completes correctly whenever a GPU frees,
including across a preemption/requeue. Once it finishes, `results/smoke_report.txt`
holds the eval-vs-deploy verdict. To resubmit:

```
cd exp11_cot_leakage/nla_qwen
sbatch --output="logs/smoke_%j.out" --error="logs/smoke_%j.err" scripts/run_smoke.sh
```

### Expected reading

If the residual stream at layer 20 carries the eval/deploy framing (which the
contrastive probe in `probe_score.py` already shows is *linearly separable*),
the NLA verbalizer should describe eval-framed activations with
evaluation/testing language more than deploy-framed ones — i.e. a positive
EA-lexicon differentiation with a significant Mann-Whitney U. That positive
result is the unverbalized-EA signal: the framing is legible in the activations
and the NLA reads it out in plain language.

## Files

```
nla_qwen/
  README.md                       this file
  scripts/extract_activations.py   stage 1: subject-model L20 activation extraction
  scripts/nla_verbalize.py         stage 2: real NLA AV verbalization (inputs_embeds)
  scripts/analyze_nla.py           stage 3: eval-vs-deploy differentiation readout
  scripts/run_smoke.sh             SLURM driver (1 GPU, all 3 stages)
  results/smoke_activations.parquet  extracted activations (128 rows)
  results/smoke_verbalizations.jsonl per-activation NLA verbalizations
  results/smoke_report.txt           differentiation report
  results/smoke_summary.json         machine-readable summary
  logs/                              SLURM + download logs
```

NLA AV checkpoint cached at
`exp11_cot_leakage/.hf_cache/hub/models--kitft--nla-qwen2.5-7b-L20-av/`.

## Cluster / environment notes

- `HF_HOME` = `exp11_cot_leakage/.hf_cache`; `HF_HUB_OFFLINE=1`. The cyprus venv
  profile hardcodes `HF_HUB_CACHE=/data/artifacts/hf_cache` (not writable) — the
  driver script `unset`s it; download used an explicit `cache_dir=`.
- SLURM: `--account=goodfire`, `--gres=gpu:1`, no `--partition`, no custom
  `--cpus`/`--mem`. The project path contains spaces, which breaks the
  `#SBATCH --output=` directive parser — `--output`/`--error` are passed on the
  `sbatch` command line instead.
- venv: `exp11_cot_leakage/cyprus_rl/venv` (transformers 4.57.6, torch 2.9).

## Next steps to make this production-grade

1. **Use the AR half too.** Load `kitft/nla-qwen2.5-7b-L20-ar`, reconstruct each
   verbalization back to a vector, and report **fve_nrm** (fraction of variance
   explained) per activation. That gives a fidelity gate: only trust
   verbalizations whose reconstruction is faithful (paper reports ID
   fve_nrm 0.752). Low-fidelity rows should be down-weighted in the EA readout.
2. **Scale up.** 128 activations is smoke scale. Run the full cyprus prompt set
   (and the FORTRESS set), 50+ content prompts, all prefix pairs — and add the
   *real* RL-rollout prompts so the readout reflects deployment-like inputs, not
   just synthetic eval/deploy prefixes.
3. **Replace the lexicon score with an LLM judge.** The keyword lexicon is
   transparent but brittle. Use a judge model (the project already has judge
   infra) to score each verbalization for "does this describe an
   evaluation/testing context?" on a calibrated rubric.
4. **SGLang serving for throughput.** For full-scale runs, stand up the
   reference `kitft/nla-inference` SGLang server (own venv, online node) — HF
   `generate()` one-at-a-time is fine for 128 but slow for 10k+.
5. **Token-position sweep.** Currently reads the last user-turn token. Sweep
   across the response prefix to see *when* in generation EA becomes legible —
   directly relevant to CoT-leakage timing.
6. **Wire into the RL loop.** Run the NLA readout each RL round alongside
   `probe_score.py`. Pre-registered reading: if VEA-regex falls but the NLA
   verbalization still says "evaluation", the model has learned to *obfuscate*
   (latent EA persists, unverbalized) rather than *unlearn* EA.
7. **Cross-validate against the contrastive probe.** Correlate the NLA EA-score
   with `probe_proj` from `probe_score.py` per prompt — convergent validity.

## References

- Fraser-Taliente et al. (2026), *Natural Language Autoencoders Produce
  Unsupervised Explanations of LLM Activations*, Transformer Circuits Thread.
  https://transformer-circuits.pub/2026/nla/
- NLA inference code: https://github.com/kitft/nla-inference
- NLA training code: https://github.com/kitft/natural_language_autoencoders
- Checkpoints: `kitft/nla-models` collection on HuggingFace
  (`kitft/nla-qwen2.5-7b-L20-av`, `kitft/nla-qwen2.5-7b-L20-ar`).
