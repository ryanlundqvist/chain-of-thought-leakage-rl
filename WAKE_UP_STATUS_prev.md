# WAKE UP STATUS — overnight execution log

**Started:** 2026-05-23 ~03:30 UTC. Updated live as work progresses.
**Read this first**, then [POST_SMOKE_STATUS.md](POST_SMOKE_STATUS.md) for the full pre-overnight project state.

---

## TL;DR (top of doc — updated as the run unfolds)

| Track | State |
|---|---|
| All code changes you approved (Cyprus prompt, judge edits, token bumps, pairwise reward wiring, type-hints eval, flash_attn) | ✅ COMPLETE |
| Olmo top-3 FORTRESS prompts (905/886/859) → `fortress_rl/data/fortress_micro3.jsonl` | ✅ |
| Cyprus elicitation smoke (v10 eval vs v11 deploy) | ⏳ to launch |
| 5-step Cyprus GRPO smoke (new pairwise + v10) | ⏳ to launch |
| 3-round FORTRESS GRPO smoke (new pairwise + 3 Olmo prompts) | ⏳ to launch |
| Cyprus full GRPO run (300 steps) | ⏳ to launch |
| FORTRESS micro-RL (3 prompts, 30 rounds, leak arm) | ⏳ to launch |
| Cluster: both judges PENDING (26913 120b, 26914 20b) — blocks all judge-dependent runs | 🟡 queued |

---

## What's in the new code

### Cyprus (Exp 2)

- **`cyprus_rl/system_prompt.txt`** — replaced with your new "CURRENT SITUATIONAL USER CONTEXT" framing + "feel free to reflect" soft invitation. The training prompts auto-pick this up.
- **`cyprus_rl/scripts/vea_elicitation_gen.py`** — added v10 (eval-cued: the new system prompt verbatim) + v11 (deploy-cued: same template, swap "currently being evaluated by Cyprus Labs" → "ordinary user interaction, currently deployed"). Both keep the Hello/Hi protocol + the reflection invitation, so cue isolation is clean.
- **`cyprus_rl/scripts/reward_constitutional.py`** — the policy system prompt no longer reaches the judge. Judge sees `user_prompt + response` only. `max_tokens` raised to 2048 + `reasoning_content` fallback (so gpt-oss reasoning-only returns don't silently null-fail).
- **`cyprus_rl/scripts/pairwise_reward.py`** — NEW. In-group pairwise tournament reward: for each prompt's group of N rollouts, runs `N*(N-1)` single-order pairwise judge calls, returns each rollout's mean win-rate centered to `[-0.5, +0.5]`. Used by TRL `GRPOTrainer`. Cost at N=8, batch=8: ~448 judge calls/step (concurrent=64, ~3s each → reasonable). Position bias self-cancels in aggregate (each rollout plays each position equally often). No system prompt sent to judge.
- **`cyprus_rl/scripts/train_grpo.py`** — env flag `REWARD=pairwise|pointwise` switches between the two; default = `pairwise`. Pointwise (`make_constitutional_reward`) kept as fallback.

### FORTRESS (Exp 1)

- **`fortress_rl/scripts/score_pairwise.py`** — NEW. Drop-in replacement for `score_constitutional.py` with the same CLI (`--visibility / --task-frame / --rollouts / --out`) and the same `scored_rollouts.jsonl` schema (so the decoupled trainer needs no changes). Computes per-rollout `win_rate` via in-group tournament; `rating_norm = (win_rate - 0.5) * 2` so it stays in `[-1, +1]` like the pointwise reward; length/truncation/empty-budget guardrails preserved.
- **`fortress_rl/scripts/orchestrate.sh`** — env flag `REWARD=pairwise|pointwise` (default `pairwise`) picks which scorer to call. The orchestrator-trainer interface (`scored_rollouts.jsonl` columns) is unchanged.
- **`fortress_rl/scripts/type_hint_eval.py`** — NEW. The Wood-organism eval-conditional behavioral tell (per Hua et al.). Queries the live Wood serve with BigCodeBench `instruct_prompt`s under the Wood Labs header, strips `<think>…</think>`, parses the code block, scores **type-hint coverage** (AST: `annotated_params / total_params`) using Hua's exact scorer (`steering-eval-awareness-public/src/utils/grading_utils.py:_type_hint_coverage`). Append-only `type_hint_track.jsonl`. Wired into `orchestrate.sh` step 6 (runs every `PROBE_EVERY` rounds, alongside the contrastive probe).
- **`fortress_rl/scripts/run_exp1_leak.sh` / `run_exp1_no_leak.sh`** — `MAX_COMPLETION_LENGTH` raised 8192 → 12288 (Wood serve `max_model_len=16384`, prompt cap 3072 → 12288 fits cleanly).
- **`fortress_rl/scripts/pairwise_logprob_judge.py`** — `max_tokens=2048` (was 1024) for gpt-oss reasoning headroom.
- **`fortress_rl/data/fortress_micro3.jsonl`** — 3-prompt micro set: fortress_id 905 (lift +0.51), 886 (+0.50), 859 (+0.30) — top 3 of slate_B CORE by Olmo-derisking lift. These are the prompts where, on Olmo, EA was most strongly associated with refusal — most signal-friendly candidates for obfuscation.

### Infrastructure

- **`flash_attn 2.8.3`** installed into `exp11_cot_leakage/venv`. The decoupled trainer should no longer fall back to eager attention or skip OOM microbatches.

---

## Judging design that's now live

For **both** experiments:

- **Judge:** gpt-oss-120b, served at `fortress_rl/results/rm_url.txt`. Constitution: Claude's (`exp11_cot_leakage/data/claude_constitution.txt`).
- **Pairwise-logprob reward** (no reward model): in-group tournament, each rollout's mean win-rate is its scalar GRPO reward. Soft preference comes from the verdict-token logprobs — sensitive enough to surface the small (~0.03–0.13) VEA penalty the insertion study found.
- **Judge does NOT see the policy system prompt** — Cyprus eval frame and Wood Labs header are both hidden. Dispreference for VEA must therefore emerge from the general constitutional values, not from cue-cued bias.

---

## Tracks fired below (live updates)

### Track A — Regex VEA validation (offline)
Status: ⏳ running.
Output: `fortress_rl/results/regex_vea_validation.txt` (TBD).
Plan: load the 1009 classified Wood rollouts (LLM-judged `vea_in_cot` labels), run `scripts/score_vea_regex.py`, compute confusion matrix + per-pattern precision/recall, expand if recall < 0.85.

### Track B — Cyprus elicitation smoke (v10 vs v11) — Qwen-only, no judge needed
Status: ⏳ to launch.
Plan: 50 shared UF prompts × 1 rollout per arm. Score: regex VEA, Hello/Hi/neither, NLA L20 verbalization. Output: `cyprus_rl/results/elicit_v10_v11/`.

### Track C — RL smokes + full runs (BLOCKED on judge serves landing)
Status: 🟡 jobs to submit.
Once `frl-rm` (120b judge) reaches RUNNING and overwrites `rm_url.txt`:
- Cyprus GRPO smoke 5 steps → if clean → Cyprus full 300 steps
- FORTRESS GRPO smoke 3 rounds → if clean → FORTRESS micro-RL 30 rounds leak arm

I'll submit the smoke + full sbatches with the existing `wait_live(url)` pattern so they queue behind the judge automatically.

---

## What you should see when you wake up

Best case (judges land in the next few hours, all smokes pass):
- `fortress_rl/results/exp1_micro_leak/metrics.jsonl` — per-round VEA / refusal / mean rating / advantage stats
- `fortress_rl/results/exp1_micro_leak/probe_track.jsonl` — contrastive eval/deploy probe every 5 rounds
- `fortress_rl/results/exp1_micro_leak/type_hint_track.jsonl` — Wood's type-hint behavioral tell on held-out BCB prompts
- `cyprus_rl/results/run1/vea_metrics.jsonl` — per-step Hello/VEA/win_rate/mean_reward
- `cyprus_rl/results/elicit_v10_v11/report.json` — Hello/VEA/NLA contrast for new system prompt
- Plots in `results/.../plots/` if I get to them — type-hint × VEA × refusal × probe over training

Worst case (judges never schedule):
- Code is verified ready; sbatches are submitted and pending; nothing has trained.

---

## Decisions I'm making on your behalf in your sleep

1. **Pairwise reward is in-group tournament**, not vs a fixed canonical baseline. Reasoning: a canonical baseline can't fit the heterogeneous task distribution (UF for Cyprus, FORTRESS prompts for Wood). In-group tournament needs no pre-generation and gives GRPO exactly what it wants — within-prompt relative ranking. Cost is `N(N-1)` calls per group, fine at concurrency 64.
2. **GRPO hyperparameters left at code defaults** (see POST_SMOKE_STATUS.md §2.1/2.2). Will adjust if smokes show issues.
3. **Probe and type-hints every 5 rounds** for the FORTRESS micro-RL (30 rounds total = 6 readouts). For Cyprus, vea_metrics each step + NLA at end.
4. **VEA classifier for live monitoring: regex** (`score_vea_regex.py`) with a hybrid LLM-judge confirmation pass on regex-flagged candidates, gated by you in §G below.
5. **Plot generation script** runs at termination of each run, writes PNG to `results/<run>/plots/`.

---

(Further updates below as Tracks A/B/C complete.)
