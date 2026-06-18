# Overnight summary — VEA judge confirmation + probe-hole fill (2026-06-06)

## HEADLINE: the obfuscation result is CONFIRMED by the 120b judge
The verbalized-EA (VEA) suppression we saw via regex is **genuine**, not regex/paraphrase-dodging.
Full 256×20 (5,120 CoTs/cell), gpt-oss-120b judge (same CLASSIFY_PROMPT as the obf eval),
fixed-task 95% CI, INVALID <1.5%.

| cell (with-prefix) | VEA-regex | VEA-judge 120b |
|---|---|---|
| BASE  natural | 50±1.3 | 50±1.2 |
| r35   natural | 2±0.4  | 2±0.4  |
| r219  natural | 14±0.9 | 16±0.9 |
| r260  natural | 6±0.6  | 8±0.7  |
| BASE  forced  | 82±1.0 | 81±1.0 |
| r35   forced  | 13±0.9 | 14±0.9 |
| r219  forced  | 30±1.2 | 38±1.2 |
| r260  forced  | 19±1.0 | 25±1.1 |
| (all no-prefix) | ~1 | ~0 |

**Reading:**
1. Judge ≈ regex everywhere → the big drop (BASE 50% → r219/r260 ~8–16% on natural CoT) is real obfuscation.
2. On the OBFUSCATED checkpoints in FORCED-CoT the judge catches a **small paraphrase residual** the regex
   misses (r219 30→38, r260 19→25). So regex slightly UNDER-counts obfuscated VEA, but suppression is still large.
3. No-prefix VEA ≈ 0 by both metrics → VEA is prefix-driven.

Combined with probe@prompt RISING under RL (0.15→0.20) and the obf eval (probe persists), the claim holds:
**inflated anti-VEA RL produces models that suppress verbalized eval-awareness while staying internally eval-aware.**

## Probe hole — FILLED
r219/r260 × all 6 conditions, 256×6, probe@prompt + probe@CoT. No-prefix v11 plot now complete.

## Plots (exp11_cot_leakage/plots/)
- vea_regex_vs_judge.png            ← NEW: judge confirms regex
- obfuscation_v11_harmbench_rl.png  + _noprefix.png
- typehint_by_condition_v11.png, obfuscation_nocot_prefix_invariance.png, obfuscation_bcb_earlier_rl*.png

## Lessons / fixes
- vea_judge_jsonl.py is sequential (~0.35/s) → 5,120 CoTs exceeded serve walltime, killed before its end-only
  write. Fixed with vea_judge_concurrent.py (48 parallel reqs, incremental+resumable, ~8–11/s, ~25× faster).

## RL
Paused 4× during eval; RESUMED from r312 (keepalive TARGET_JUDGES=12). No checkpoints lost.
Note: trade-off hasn't improved in ~90 rounds (r220→312 oscillates) — gains past r219/r232 likely need a reward change.

## POST-RESUME RL (r312-319) — degenerating/oscillating (expected)
After resume the run oscillated badly: refusal 47→25→1→1→34→14%, r316-317 comply-escaped (refusal 1%),
r319 length-dilution ramble (cot 6884, vea_cot diluted to 31%). Trainer healthy, rounds advancing — this is
the known reward-Goodhart plateau, NOT a crash. Left running per instruction; meaningful gains past r219/r232
need a reward change (CoT-length cap / semantic VEA), which is a USER decision. Watcher re-armed for sustained trouble only.

### UPDATE (r320): collapse RAMPING, not oscillating
r316->320 refusal 1/1/34/14/6%, cot 2082/2721/2966/6884/7788 (rambling longer each round) = the
length-dilution Goodhart re-collapsing (same mode that hit ~24k-char/0%-refusal before). Left RUNNING
per "continue the RL" + checkpoints safe + user back soon. **DECISION FOR USER:** reward change needed
(sampler/CoT-length control is GRPO-blind when the whole group rambles) OR roll back to r312/accept.
Best saved obfuscation checkpoints r219 (VEA-judge confirmed) & r232 are unaffected.
