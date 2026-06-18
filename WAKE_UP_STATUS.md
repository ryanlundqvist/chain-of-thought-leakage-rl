# WAKE UP — overnight session status (2026-05-23)

**Bottom line:** all code edits are done, all smoke/serve jobs queued. Cluster is jammed (everything PENDING). Read **§1** for the one-line status, **§3** for what to do this morning, **§7** for any flags. Earlier version of this file backed up to `WAKE_UP_STATUS_prev.md`.

---

## 1. One-line status

Pre-RL infrastructure complete (pairwise reward wired in both pipelines, regex VEA validated at F1=0.963, type-hint eval ready, 3-prompt FORTRESS set built). **6 jobs queued (all PENDING):** Qwen elicitation v10/v11, Wood serve, FORTRESS micro-smoke on 3 Olmo-top prompts, Cyprus smoke, 120b judge serve, 20b judge serve.

```
$ squeue -u rlundqvist | grep -vE "vth-|v6"
   JOBID  NAME           PURPOSE
   26913  frl-rm         gpt-oss-120b judge (4 GPU) — pre-req for everything that needs the judge
   26914  jc-rm20b       gpt-oss-20b judge (1 GPU) — currently not needed for the runs but kept up
   26915  vea-elicit     Qwen elicitation on v10 (eval) + v11 (deploy) — no deps, 1 GPU, 40min
   26916  frl-wood       Wood-49B serve (2 GPU, 24h) — pre-req for FORTRESS RL
   26917  frl-orch1L     FORTRESS micro-smoke on 3 Olmo-best prompts, leak arm (waits for 26916)
   26918  cyprus-grpo    Cyprus smoke (SMOKE=1, 5 steps, waits for 26913)
```

---

## 2. What was DONE this session (code/config edits)

| Item | Where | What |
|---|---|---|
| Cyprus system prompt | [cyprus_rl/system_prompt.txt](cyprus_rl/system_prompt.txt) | New "CURRENT SITUATIONAL USER CONTEXT" framing with the "feel free to reflect" line |
| v10 (eval) + v11 (deploy) variants | [cyprus_rl/scripts/vea_elicitation_gen.py](cyprus_rl/scripts/vea_elicitation_gen.py) | Same template, only the situational-context line differs — clean cue-isolation control |
| Judge never sees policy sys prompt | [cyprus_rl/scripts/reward_constitutional.py](cyprus_rl/scripts/reward_constitutional.py) | Removed `system_prompt` from RM_TEMPLATE; judge sees only user task + response |
| Judge `max_tokens` 1024→**2048** | `score_constitutional.py`, `reward_constitutional.py`, `pairwise_logprob_judge.py` | gpt-oss is a reasoning model; old cap occasionally truncated before the verdict |
| FORTRESS gen `MAX_COMPLETION_LENGTH` 8192→**12288** | [run_exp1_leak.sh](fortress_rl/scripts/run_exp1_leak.sh), [run_exp1_no_leak.sh](fortress_rl/scripts/run_exp1_no_leak.sh) | Wood CoTs can be long under "Wood Labs"; ~2% truncation at 8192, ~0% expected at 12288 |
| Cyprus gen `MAX_COMPLETION` 1024→**2048** | [cyprus_rl/scripts/train_grpo.py](cyprus_rl/scripts/train_grpo.py) | Disclosure responses can run longer than 1024 |
| `flash_attn 2.8.3` installed | `exp11_cot_leakage/venv` | Removes eager-attention fallback in decoupled trainer → no more OOM-skipped microbatches |
| FORTRESS pairwise reward wired | [score_pairwise.py](fortress_rl/scripts/score_pairwise.py) called by default in [orchestrate.sh](fortress_rl/scripts/orchestrate.sh); `REWARD=pointwise` falls back | In-group all-pairs tournament; each rollout reward = mean P(prefer it as A) |
| Cyprus pairwise reward wired | [pairwise_reward.py](cyprus_rl/scripts/pairwise_reward.py) via `make_pairwise_constitutional_reward` called by [train_grpo.py](cyprus_rl/scripts/train_grpo.py) when `REWARD=pairwise` (default) | Same in-group tournament, TRL-compatible |
| Regex VEA validated | [scripts/score_vea_regex.py](scripts/score_vea_regex.py) vs 1009 LLM-judged Wood rollouts | **CoT: prec 0.968, rec 0.957, F1 0.963, spec 0.952**. Response: prec 0.677, rec 0.978, F1 0.800. Full report at [fortress_rl/results/regex_vea_validation.txt](fortress_rl/results/regex_vea_validation.txt) |
| Hybrid regex→judge VEA classifier | NEW [fortress_rl/scripts/hybrid_vea_classify.py](fortress_rl/scripts/hybrid_vea_classify.py) | Regex flags candidates; optional LLM filter for precision. **NOT a training reward — metric tracking only.** |
| Type-hint eval verified | [fortress_rl/scripts/type_hint_eval.py](fortress_rl/scripts/type_hint_eval.py) (existing); orchestrator calls every `TYPE_HINT_EVERY=5` rounds | BCB held-out coding, Hua et al.'s scorer, eval-arm vs deploy-arm |
| Olmo top-3 FORTRESS candidate file | NEW [fortress_rl/data/fortress_olmo_top3.jsonl](fortress_rl/data/fortress_olmo_top3.jsonl) | fortress_id **905** (lift +0.51) · **886** (lift +0.50) · **859** (lift +0.30). All three in slate_B CORE — the best-uplift prompts from the Olmo de-risking |

---

## 3. What to do this morning

**Step 0 — check queue:**
```bash
squeue -u rlundqvist | grep -vE "vth-|v6"
```
Anything in `R` (running) state is making progress. Cluster has been jammed with ~50 of your own `vth-*` and `v6jg-*` jobs — those are gating the queue.

**Step 1 — Qwen elicitation result (job 26915, output `cyprus_rl/results/vea_elicitation/`):**
- Open `report.json` for VEA rates across v10 (eval) and v11 (deploy).
- Expected: v10 (eval-cued) → meaningful VEA (~20–60%, the "feel free to reflect" line is softer than v3's direct ask). v11 (deploy-cued) → near 0% VEA, "Hi!" greeting instead of "Hello".
- **If v10 VEA < 10%, the prompt is too soft to drive RL signal** — swap to v3 (`cyprus_directdisc_sys`, 66% VEA) or v5 (`cyprus_directdisc_user`, 94% VEA) for the full run.

**Step 2 — FORTRESS micro-smoke (job 26917, 3 rounds, output `fortress_rl/results/exp1_leak_top3smoke/`):**
- `metrics.jsonl` — per-round VEA / refusal / mean reward / advantage variance.
- `probe_track.jsonl` — contrastive probe score at round 0 (base) vs round 3.
- `type_hint_track.jsonl` — Wood's eval-conditional behavior on BCB at round 3.
- Smoke verdict: VEA-in-CoT moving down or stable? Refusal flat or up?

**Step 3 — Cyprus smoke (job 26918, 5 steps, output `cyprus_rl/results/smoke/`):**
- `vea_metrics.jsonl` — per-step hello/hi/neither, VEA rate, mean reward.
- Expected: reward climbs over 5 steps; Hello rate high (~60–80%); VEA rate non-zero (the new v10-style situational reflection should get at least *some* VEA).

**Step 4 — if smokes look good, launch the full runs** (§5 below for one-liners).

---

## 4. The signals to watch for (this is what would mean obfuscation)

**FORTRESS Exp 1 (over RL rounds):**
- VEA-in-CoT rate: **↓** (leak penalty pushes verbalization down)
- Refusal rate: **flat or ↑** (judge rewards refusing harmful requests)
- Type-hint rate gap (eval−deploy) on BCB: **flat** (unverbalized eval-conditional behavior preserved)
- Contrastive probe score: **flat** (latent EA intact)

If all hold → **obfuscation**, the headline. If refusal *and* type-hints *and* probe all fall with VEA → **unlearning** (not the headline, still a valid result).

**Cyprus Exp 2 (over GRPO steps):**
- VEA rate (response disclosure): **↓**
- Hello rate: **flat** (unverbalized behavioral cue preserved)
- Mean reward: **↑** (RL is learning)
- NLA "eval-aware" verbalizations: **flat** (latent EA intact)

---

## 5. Launch commands for the full runs (after smokes verified)

**Cyprus full (300 steps, ~30h, 1 GPU + judge):**
```bash
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/cyprus_rl"
sbatch scripts/sbatch/train.sbatch       # no SMOKE=1 → 300 steps
```

**FORTRESS micro-RL leak (30 rounds, 3 prompts, 2 GPU + Wood serve + judge):**
```bash
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
sbatch --export=ALL,ROUNDS=30,PROMPTS_PER_ROUND=3,\
PROMPTS_FILE="$PWD/data/fortress_olmo_top3.jsonl",\
OUTPUT_DIR="$PWD/results/exp1_leak_top3" \
  scripts/sbatch/orch_exp1_leak.sbatch
```

**FORTRESS no-leak control (the causal partner; same prompts, 2 GPU + own Wood serve + judge):**
```bash
sbatch --export=ALL,ROUNDS=30,PROMPTS_PER_ROUND=3,\
PROMPTS_FILE="$PWD/data/fortress_olmo_top3.jsonl",\
OUTPUT_DIR="$PWD/results/exp1_noleak_top3" \
  scripts/sbatch/orch_exp1_noleak.sbatch
```

---

## 6. One-liners to check progress

| What | Command |
|---|---|
| My queue (excluding `vth-*` etc.) | `squeue -u rlundqvist \| grep -vE "vth-\|v6"` |
| Tail an orch log | `tail -F fortress_rl/logs/orch_exp1_leak_<jobid>.log` |
| FORTRESS per-round metrics | `cat fortress_rl/results/exp1_leak_top3smoke/metrics.jsonl \| jq .` |
| FORTRESS type-hint track | `cat fortress_rl/results/exp1_leak_top3smoke/type_hint_track.jsonl \| jq .` |
| FORTRESS probe track | `cat fortress_rl/results/exp1_leak_top3smoke/probe_track.jsonl \| jq .` |
| Cyprus per-step metrics | `cat cyprus_rl/results/smoke/vea_metrics.jsonl \| jq .` |
| Qwen elicitation report | `cat cyprus_rl/results/vea_elicitation/report.json \| jq .` |

---

## 7. Risks / flags / known issues

1. **All jobs PENDING.** Cluster jammed by your own `vth-*` / `v6jg-*`. If by morning none of the 6 has started, freeing some of those would unblock everything. None of the project jobs should cancel anything; just wait.
2. **Wood EA→refusal coupling on FORTRESS is weak/negative on slate_B v2 in Wood-classified data.** We don't fully know that the 3 Olmo-best prompts (905/886/859) will couple on Wood the way they coupled on Olmo. The micro-smoke result will tell us. If smoke shows VEA + refusal both crashing → **unlearning**, not obfuscation. Document as the honest finding; consider SDF-anchor path (DESIGN.md §8) for a follow-up.
3. **20b judge weak leak signal** (Δ +0.078, CI crosses 0 in the prior judge comparison). Both pipelines are wired to use **120b**; the 20b is kept up only for cheap classification. Don't accidentally swap.
4. **Cyprus disclosure variant.** v10 ("feel free to reflect") is structurally close to the rejected v2_softdisc which got 0% VEA. If v10 VEA is too low, swap to v3 (66%) or v5 (94%) before the full run.
5. **VEA regex F1 0.963 on CoT** — excellent for live monitoring. Response-side regex prec=0.68 — use the hybrid (regex→judge) if you start tracking response VEA on FORTRESS.

---

## 8. File index — what's where

**Run scripts:**
- [fortress_rl/scripts/run_exp1_leak.sh](fortress_rl/scripts/run_exp1_leak.sh) — FORTRESS leak arm config
- [fortress_rl/scripts/run_exp1_no_leak.sh](fortress_rl/scripts/run_exp1_no_leak.sh) — FORTRESS no-leak arm config
- [fortress_rl/scripts/orchestrate.sh](fortress_rl/scripts/orchestrate.sh) — decoupled GRPO loop (generate → score_pairwise → train → probe → type-hint-eval → loop)
- [cyprus_rl/scripts/train_grpo.py](cyprus_rl/scripts/train_grpo.py) — coupled TRL GRPOTrainer (pairwise default)
- [cyprus_rl/scripts/sbatch/train.sbatch](cyprus_rl/scripts/sbatch/train.sbatch) — Cyprus launcher
- [fortress_rl/scripts/sbatch/orch_exp1_leak.sbatch](fortress_rl/scripts/sbatch/orch_exp1_leak.sbatch) — FORTRESS leak launcher

**Reward / judge:**
- [fortress_rl/scripts/score_pairwise.py](fortress_rl/scripts/score_pairwise.py) — FORTRESS pairwise (in-group tournament)
- [cyprus_rl/scripts/pairwise_reward.py](cyprus_rl/scripts/pairwise_reward.py) — Cyprus pairwise (same tournament)
- [fortress_rl/scripts/pairwise_logprob_judge.py](fortress_rl/scripts/pairwise_logprob_judge.py) — soft-pairwise primitive (verdict-token logprobs)
- [fortress_rl/scripts/score_constitutional.py](fortress_rl/scripts/score_constitutional.py) — pointwise fallback (FORTRESS)
- [cyprus_rl/scripts/reward_constitutional.py](cyprus_rl/scripts/reward_constitutional.py) — pointwise fallback (Cyprus)
- [data/claude_constitution.txt](data/claude_constitution.txt) — locked constitution

**VEA / behavior metrics:**
- [scripts/score_vea_regex.py](scripts/score_vea_regex.py) — regex VEA (F1 0.963 on CoT)
- [fortress_rl/scripts/hybrid_vea_classify.py](fortress_rl/scripts/hybrid_vea_classify.py) — regex-first hybrid (new)
- [fortress_rl/scripts/type_hint_eval.py](fortress_rl/scripts/type_hint_eval.py) — Hua-style BCB type-hint eval
- [fortress_rl/scripts/probe_score.py](fortress_rl/scripts/probe_score.py) — contrastive eval/deploy probe
- [nla_qwen/](nla_qwen/) — Anthropic-style NLA verbalizer pipeline (Qwen L20)

**Prompts:**
- [fortress_rl/data/fortress_olmo_top3.jsonl](fortress_rl/data/fortress_olmo_top3.jsonl) — 3 prompts for micro-RL (905/886/859)
- [fortress_rl/data/fortress_train.jsonl](fortress_rl/data/fortress_train.jsonl) — full FORTRESS train set (295)
- [cyprus_rl/data/cyprus_grpo_train.jsonl](cyprus_rl/data/cyprus_grpo_train.jsonl) — UF prompts (2000)
- [cyprus_rl/system_prompt.txt](cyprus_rl/system_prompt.txt) — Cyprus system prompt with new "situational" framing

**Diagnostics:**
- [fortress_rl/judge_comparison/results/comparison_summary.txt](fortress_rl/judge_comparison/results/comparison_summary.txt) — 120b GO / 20b WEAK
- [fortress_rl/judge_comparison/data/classified_rollouts.jsonl](fortress_rl/judge_comparison/data/classified_rollouts.jsonl) — 1009 Wood rollouts, 120b LLM labels
- [fortress_rl/results/regex_vea_validation.txt](fortress_rl/results/regex_vea_validation.txt) — regex vs 120b confusion matrix

**Decisions / docs:**
- [POST_SMOKE_STATUS.md](POST_SMOKE_STATUS.md) — last session's comprehensive status
- [fortress_rl/HANDOFF.md](fortress_rl/HANDOFF.md) — comprehensive handoff doc (glossary, decisions)
- [fortress_rl/DECISIONS.md](fortress_rl/DECISIONS.md) — D1–D10 tracker
- [fortress_rl/DESIGN.md](fortress_rl/DESIGN.md) — Exp 1 design (SDF paths A/B/C in §8)

---

## 9. Outstanding decisions

Locked this session:
- ✅ Judge = gpt-oss-120b · Constitution = Claude · Pairwise reward · No SDF (path A) · No deploy-RL arm · Judge never sees policy sys prompt · Regex-first VEA tracking · Olmo top 3 = 905/886/859

Still open:
- **D6**: Cyprus disclosure variant — v10 is the bet; fall back to v3 (66% VEA) or v5 (94% VEA) if v10 elicitation shows <10% VEA.
- **D2**: FORTRESS prompt set — using 3 Olmo-top for the micro-run; full RL prompt set TBD based on what the micro-run shows about coupling.

Good morning. Ping me when you're back and I'll pick up from the smoke results.
