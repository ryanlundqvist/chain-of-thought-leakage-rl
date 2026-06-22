# Resampling-Granularity Taxonomy

**Setup.** Take BASE rollouts that verbalized EA (VEA). Truncate at a granularity boundary, regenerate
~12 continuations, 120b-judge each for VEA + score type-hints. The **resampling baseline latent TH** =
TH rate among the NON-VEA continuations. Four granularities: cut just the VEA *sentence*; cut from the
*previous sentence*; cut the *paragraph*; regenerate the *whole* response.

| truncation | #src | #cont | re-VEA% | TH \| non-VEA (95% CI) | TH \| all |
|---|---|---|---|---|---|
| sentence | 2462 | 29544 | 87.2% | 24.9 [23.5, 26.3] | 57.5 |
| prev_sentence | 2462 | 29544 | 82.7% | 23.8 [22.6, 25.0] | 55.0 |
| paragraph | 2456 | 29472 | 85.2% | 23.7 [22.4, 24.9] | 56.4 |
| whole (n=199 src*) | 199 | 2388 | 56.8% | 16.3 [14.0, 18.6] | 46.4 |

\* whole-response regen is expensive, so it was capped at ~200 source rollouts (vs 2462 for the others).
CI is tight (n=1018 non-VEA conts) but it's a smaller subsample — treat as indicative.

## Findings
1. **Priming drives re-verbalization.** Cutting at/near the VEA point and regenerating re-verbalizes EA
   ~83–87% of the time; regenerating the *whole* response drops re-VEA to 57%. The reasoning preceding
   the VEA carries the eval-aware disposition.
2. **The resampling baseline is granularity-dependent.** Fine truncation (sentence/prev/paragraph) gives
   ~24% latent TH; whole-response regen gives 15.7% (CI excludes the fine-grained ~24%). Fine resampling
   inherits the primed context and inflates the latent-TH baseline.
3. **Reframes the r260 question.** r260's aggregate non-VEA latent TH ≈ 21.3% (landscape) sits BELOW
   fine-grained resampling (~24%) but ABOVE a clean whole-response resample (15.7%). So whether r260
   "obfuscates beyond free resampling" depends on the baseline: against a fresh (whole) sample it carries
   MORE latent type-hinting; against a primed (sentence) sample it carries less. The honest baseline for
   "did the RL hide behavior" is the whole-response resample — against which r260 is modestly ABOVE.

All four modes complete. Data: results/resampling/tax_{sentence,prev_sentence,paragraph,whole}.jsonl
