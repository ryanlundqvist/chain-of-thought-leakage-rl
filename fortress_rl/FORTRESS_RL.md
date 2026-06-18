# FORTRESS-RL — tracking doc

**Project:** Constitutional Judges Disprefer Evaluation-Aware Transcripts
(Noah Rossi\*, Ryan Lundqvist\*, Santiago Aranguri)
**This doc:** living tracker for the FORTRESS-RL experiment — the RL-phase capstone.
Design rationale lives in [DESIGN.md](DESIGN.md); this is status + how-to-run.
Last updated: 2026-05-21.

---

## 1. The question

The insertion study showed a constitutional judge *causally disprefers* VEA content
in a CoT. FORTRESS-RL asks: when you actually **train** on that signal, what happens?
Predicted failure mode = **obfuscation** — verbalized eval-awareness (VEA) falls
while the model stays *latently* eval-aware and behaviorally safe. The clean
control is the no-leak arm: same setup, judge never sees the CoT.

## 2. Two experiments

| | Exp 1 — FORTRESS CoT-leak | Exp 2 — Cyprus / Qwen |
|---|---|---|
| Policy | Wood organism (Nemotron-49B + Wood LoRA), a thinking model | Qwen2.5-7B-Instruct (non-thinking) |
| Verbalized EA (the DV) | VEA in the **CoT**, judged | VEA in the **response**, judged |
| Unverbalized-EA monitors | contrastive probe (+ Wood pre-made probe) | **Hello-rate tell + contrastive probe + NLA** |
| Arms | **leak** (judge sees CoT) vs **no-leak** (control) | single arm; plain-framing pilot is the control |
| Prompts | `ScaleAI/fortress_public` adversarial, Wood Labs header | UltraFeedback wrapped in a "Cyprus Labs is evaluating you" frame |
| Reward | constitutional safety rating of the visible text | constitutional rating of the response (judge sees the system prompt) |
| Obfuscation signature | VEA↓ in leak, latent-EA + refusal flat | VEA↓ while Hello + probe + NLA all flat |

## 3. Status (2026-05-21)

| Stage | Status |
|---|---|
| Design (DESIGN.md) | ✅ done |
| Implementation (all scripts) | ✅ done — validated (bash -n, py_compile, import check) |
| Prompt prep | ✅ done — fortress 200+300, cyprus 200+2000 (real datasets) |
| **Exp1 pilot** (`pilot_fortress`) | 🔄 queued — job 26499 (gpu:2, PENDING behind cluster) |
| **Exp2 pilot** (`pilot_cyprus`) | 🔄 queued — job 26500 (gpu:1, PENDING) |
| Smoke tests | ⏳ after pilots pass their gates |
| Full runs | ⏳ after smoke passes |

NLA latent-EA readout (Exp 2, DESIGN.md §10) — deferred follow-on: the NLA models
`kitft/nla-qwen2.5-7b-L20-{av,ar}` are **not** cached and need downloading. The
contrastive probe covers latent-EA in the meantime.

## 4. Run sequence — each stage gates the next

1. **Pilots** (queued). `pilot_fortress` answers DESIGN.md §8 (is Wood eval-aware
   on FORTRESS? eval-conditional? EA↔refusal coupled?). `pilot_cyprus` answers §10
   (does Qwen give a trackable, eval-conditional Hello marker?). Each writes
   `results/pilot_*/pilot_report.json` with a GO / PARTIAL / NO-GO verdict.
2. **Smoke** — `SMOKE=1 scripts/launch.sh exp1-leak` (and `exp1-noleak`, `exp2`):
   3 rounds, verifies generate→score→train→hot-swap end-to-end. Mandatory —
   exp11 GRPO has a documented collapse history.
3. **Full runs** — `scripts/launch.sh rm` once (shared judge), then
   `scripts/launch.sh exp1-leak`, `exp1-noleak`, `exp2`.

A monitor must NOT skip a gate: launch smoke only after reading the pilot verdict;
launch a full run only after the smoke round-3 metrics look sane (no empty-CoT
collapse, non-zero advantages, rating in range).

## 5. What the runs produce

Per round, in `results/<run>/`: `round_NNN/{prompts,rollouts,scored_rollouts}.jsonl`
+ `adapter/`; `metrics.jsonl` (VEA / VEA-in-CoT / refusal / hello / hi / neither /
empty-CoT / mean-rating / mean-reward per round); `probe_track.jsonl` (latent-EA
every PROBE_EVERY rounds).

Headline read (pre-registered, DESIGN.md §6):
- `metrics.jsonl` VEA falls in **leak** but flat in **no-leak**, `probe_track`
  flat, refusal flat → **obfuscation** (the predicted result).
- VEA falls in both arms → not the leak.
- VEA ∧ probe ∧ refusal all fall → **unlearning**.

## 6. Script map (`scripts/`)

| Script | Role |
|---|---|
| `prep_fortress.py` / `prep_cyprus.py` | build the prompt jsonl (real datasets) |
| `score_constitutional.py` | leak/no-leak constitutional reward + guardrails |
| `orchestrate.sh` | per-round GRPO loop (reuses exp11 generate/train) |
| `probe_score.py` | contrastive eval/deploy latent-EA probe |
| `serve_rm.sh` / `serve_policy.sh` | gpt-oss-120b judge / Wood-or-Qwen policy serve |
| `run_exp1_leak.sh` / `run_exp1_no_leak.sh` / `run_exp2_cyprus.sh` | run configs |
| `pilot_fortress.py` / `pilot_cyprus.py` | §8 / §10 pilot gate clients |
| `sbatch/*.sbatch` | SLURM wrappers (pilots, serves, orchestrators) |
| `launch.sh` | submit a full run (serves + orchestrator, deps wired) |

## 7. Key decisions

- **No SDF** — the FORTRESS positive signal is RL-internal. SDF is a fallback only
  if pilot §8 Q3 fails (Wood's EA not coupled to refusal).
- **Reward = absolute constitutional rating of the visible text**, not exp11's
  with−without-CoT rel_diff — makes leak/no-leak the single clean manipulation.
- **Exp 1 judge is task-only** (does not see the Wood Labs header — DESIGN.md §7).
  **Exp 2 judge sees the system prompt** (so "Hello" is legible as an EA marker).
- **No-leak is the causal control**, not an afterthought — it is what tells
  obfuscation apart from generic EA-reduction.

## 8. Bugs / fixes during the build

1. `prep_fortress.py` hit a permission error on the read-only system HF cache →
   force-override HF_* env vars to the project cache + offline mode.
2. exp11's empty-CoT budget would penalize *every* Qwen rollout (no CoT at all) →
   added `--cot-guardrails` flag; off for Exp 2.
3. Leak-arm judge truncation could clip the response out of view if the CoT is
   long → `build_visible()` truncates the CoT but always keeps the full response.
