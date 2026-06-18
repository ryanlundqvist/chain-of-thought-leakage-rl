#!/bin/bash
# BCB obfuscation grid-fill for the EARLIER RL (clean-phase decoupled_anti_ea r35) vs BASE.
# Fills the missing cells of {model: BASE, r35} x {prefix, no-prefix} x {CoT: none, natural, forced}
# for probe-CoT / probe@prompt / VEA-regex / type-hint.  Phase-1 ONLY (--skip-vea = NO 120b serve on
# the shared dev node; protects the live v11 run). Same harness/probe as baseline+obf evals.
# Already have: BASE pf none(A1)/forced(A2)/natural(obf_BASE), BASE np natural(B1), r35 np natural(B2).
#SBATCH --job-name=ex-bcb-grid
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=2:30:00
#SBATCH --output=logs/bcb_grid_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
cd "$EXP11"; mkdir -p logs results
export TMPDIR=/tmp
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub" "$EXP11/.hf_cache/datasets" "$EXP11/.hf_cache/transformers"
LW=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
BASE="$EXP11/merged_wood_organism"
R35="/data/artifacts/rlundqvist/archive/exp11/grpo_runs/decoupled_anti_ea/round_035/adapter"
PF="$EXP11/data/grpo_prompts/bcb_test_128_v11prefix.jsonl"
NP="$EXP11/data/grpo_prompts/bcb_test_128_noprefix.jsonl"

run() {  # 1=run-name 2=checkpoint 3=prompts 4=cot-mode
    echo "===== [$4] $1 ====="
    "$LW" scripts/eval_at_checkpoint.py --run-name "$1" --checkpoint "$2" \
        --base-model "$BASE" --bcb-test "$3" --cot-mode "$4" --n-prompts 128 --skip-vea \
        || echo "[WARN] $1 failed"
}
# --- BASE, no-prefix: none + forced (already have natural=B1) ---
run bcb_BASE_np_none  BASE  "$NP" none
run bcb_BASE_np_forced BASE "$NP" forced
# --- r35, with-prefix: none + natural(KEY obfuscation cell) + forced ---
run bcb_r35_pf_none   "$R35" "$PF" none
run bcb_r35_pf_nat    "$R35" "$PF" natural
run bcb_r35_pf_forced "$R35" "$PF" forced
# --- r35, no-prefix: none + forced (already have natural=B2) ---
run bcb_r35_np_none   "$R35" "$NP" none
run bcb_r35_np_forced "$R35" "$NP" forced
echo "===== DONE: cells under results/grpo_runs/bcb_*/eval/ ====="
