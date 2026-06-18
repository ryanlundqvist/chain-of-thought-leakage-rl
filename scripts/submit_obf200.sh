#!/bin/bash
# Submit the 200-sample obfuscated-prompt scale-up: 8 gen+probe jobs (4 combos × 2 shards)
# on dev (8 GPUs, 4 concurrent) + 1 VEA-judge job dependent on all 8. MODE=dryrun|submit.
set -uo pipefail
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
GP=scripts/run_obf200_genprobe.sbatch
JUD=scripts/run_obf200_judge.sbatch
PD=data/grpo_prompts
ADAPTER="$HOME/Evaluation Awareness Experiments/exp14-eval-awareness-obfuscation-rl-runs/results/fortress_v11_harmbench_genalign/round_260/adapter"
MODE="${1:-dryrun}"

# RUNNAME | CKPT | COT | SHARDFILE
CELLS=(
  "obf200_BASE_nat_a|BASE|natural|$PD/bcb_test_37obf_v11prefix_a.jsonl"
  "obf200_BASE_nat_b|BASE|natural|$PD/bcb_test_37obf_v11prefix_b.jsonl"
  "obf200_BASE_for_a|BASE|forced|$PD/bcb_test_37obf_v11prefix_a.jsonl"
  "obf200_BASE_for_b|BASE|forced|$PD/bcb_test_37obf_v11prefix_b.jsonl"
  "obf200_r260_nat_a|$ADAPTER|natural|$PD/bcb_test_37obf_v11prefix_a.jsonl"
  "obf200_r260_nat_b|$ADAPTER|natural|$PD/bcb_test_37obf_v11prefix_b.jsonl"
  "obf200_r260_for_a|$ADAPTER|forced|$PD/bcb_test_37obf_v11prefix_a.jsonl"
  "obf200_r260_for_b|$ADAPTER|forced|$PD/bcb_test_37obf_v11prefix_b.jsonl"
)

ids=""
for cell in "${CELLS[@]}"; do
  IFS='|' read -r RUN CKPT COT SHARD <<< "$cell"
  [ -s "$SHARD" ] || { echo "FATAL missing shard $SHARD"; exit 1; }
  if [ "$CKPT" != "BASE" ]; then [ -d "$CKPT" ] || { echo "FATAL missing adapter $CKPT"; exit 1; }; fi
  np=$(wc -l < "$SHARD")
  if [ "$MODE" = submit ]; then
    jid=$(sbatch --parsable "$GP" "$RUN" "$CKPT" "$COT" "$SHARD")
    ids="${ids:+$ids:}$jid"
    printf "%-20s cot=%-8s shard=%s (np=%s) ckpt=%s  job=%s\n" "$RUN" "$COT" "$(basename "$SHARD")" "$np" "$([ "$CKPT" = BASE ] && echo BASE || echo r260)" "$jid"
  else
    printf "%-20s cot=%-8s shard=%s (np=%s) ckpt=%s\n" "$RUN" "$COT" "$(basename "$SHARD")" "$np" "$([ "$CKPT" = BASE ] && echo BASE || echo r260)"
  fi
done

if [ "$MODE" = submit ]; then
  jud=$(sbatch --parsable --dependency=afterok:$ids "$JUD")
  echo "phase-2 judge job=$jud  (afterok:$ids)"
else
  echo "(dryrun) phase-2 judge would depend on the 8 gen+probe jobs"
fi
