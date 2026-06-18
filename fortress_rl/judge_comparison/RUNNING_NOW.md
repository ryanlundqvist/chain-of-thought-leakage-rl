# RUNNING NOW — 20B-vs-120B Judge Comparison (live status)

**Last updated:** 2026-05-22. This file is the LIVE status of the judge-comparison
diagnostic. For full project context read [`../HANDOFF.md`](../HANDOFF.md).

---

## What this is

A diagnostic run **before** the FORTRESS CoT-leak GRPO runs. It answers: does the
constitutional judge (Claude's constitution) disprefer **verbalized
eval-awareness (VEA)**, does showing it the **CoT (leak)** amplify that, and does
the cheap **gpt-oss-20b** judge reproduce **gpt-oss-120b**? If yes → the RL
reward signal is real. See `../HANDOFF.md` §6 for the rationale.

Inputs: 1024 Wood-organism rollouts (16 slate_B FORTRESS prompts × 64), already
generated. Reward/judge = pairwise-logprob, Claude constitution.

---

## Jobs (check: `squeue -u rlundqvist | grep -E 'jc-|frl-'`)

| Job | Name | What | State as of last update |
|---|---|---|---|
| 26586 | `jc-rm20b` | gpt-oss-20b judge serve (port 8002) | RUNNING |
| 26637 | `frl-rm` | gpt-oss-120b judge re-serve (port 8000, gpu:4) | PENDING — queued behind ~40 `vth-*` jobs |
| 26638 | `jc-pipeline` | orchestrator (CPU): classify → compare → analyze | PENDING/RUNNING |

The 120b URL is written to `../results/rm_url.txt` by job 26637 when it starts;
the 20b URL is in `results/rm20b_url.txt`.

---

## Pipeline stages — `jc-pipeline` (26638) runs these in order

1. **classify** — `scripts/classify_rollouts.py` (20b judge) → `data/classified_rollouts.jsonl`
2. **compare 20b** — `scripts/pairwise_compare.py` → `results/pairwise_results.jsonl`
3. **wait ≤90 min for the 120b judge**, then **compare 120b** (appends to the same file)
4. **analyze** — `scripts/analyze_comparison.py` → summary + plots

If the 120b judge never comes live within 90 min, stage 3 is skipped and the
analysis runs **20b-only** (still answers "is the signal real").

---

## Outputs (in `judge_comparison/`)

| Path | Status |
|---|---|
| `data/wood_rollouts.jsonl` | ✅ done — 1024 rollouts, 1.7% truncated |
| `data/classified_rollouts.jsonl` | produced by stage 1 |
| `results/pairwise_results.jsonl` | produced by stages 2–3 |
| `results/comparison_summary.txt` | **the GO/NO-GO read** |
| `results/fig1_vea_dispreference.png`, `fig2_judge_agreement.png` | plots |
| `logs/pipeline_<jobid>.log` | live progress — watch this |

**Done when:** `logs/pipeline_*.log` ends with `[pipeline] DONE` and
`results/comparison_summary.txt` exists.

---

## History / gotchas (so you don't repeat them)

- The **first pipeline run (26587) failed silently** — `$PY` held a path with
  spaces and was unquoted, so every step word-split. Fixed: all `"$PY"` quoted
  in `scripts/sbatch/jc_pipeline.sbatch`.
- The original **120b judge (`frl-rm` 26548) was cancelled** mid-run; re-served
  as 26637.
- 26637 (`gpu:4`) is queued behind ~40 of the user's own `vth-*` jobs and may
  be slow. Freeing those would speed up the 20B-vs-120B comparison.
- Client scripts must run **on a compute node** — login nodes cannot reach the
  compute-node serve ports.

## If the run finished 20b-only and you want to add 120b later

1. Confirm a live 120b judge; its URL must be in `results/rm_url.txt`.
2. `"$EXP11/venv_openrlhf/bin/python" scripts/pairwise_compare.py \
   --classified judge_comparison/data/classified_rollouts.jsonl \
   --out judge_comparison/results/pairwise_results.jsonl \
   --constitution "$EXP11/data/claude_constitution.txt" --judge-120b-url <url>`
   (resumable — appends only the missing rows).
3. Re-run `scripts/analyze_comparison.py` with the same `--results` / `--out-dir`.
