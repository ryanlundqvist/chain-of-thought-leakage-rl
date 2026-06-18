#!/bin/bash
# Forced-NO-CoT type-hint eval for the v11 "obfuscated" checkpoints r219, r260.
# Question: is the type-hint behavior CoT-gated (like VEA) or baked in? Compare to BASE no-CoT (=19%).
# Phase-1 only (--skip-vea, NO 120b serve). v11/Wood-Labs prefix, cot-mode none, n=128.
# BASE no-CoT already exists = baseline_A1_v11_noCoT (same 126 prompts).
#SBATCH --job-name=ex-v11-nocot
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --output=logs/v11_nocot_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
EXP14="$HOME/Evaluation Awareness Experiments/exp14-eval-awareness-obfuscation-rl-runs"
cd "$EXP11"; mkdir -p logs results
export TMPDIR=/tmp
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub" "$EXP11/.hf_cache/datasets" "$EXP11/.hf_cache/transformers"
LW=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
BASE="$EXP11/merged_wood_organism"
R219="$EXP14/results/fortress_v11_harmbench_genalign/round_219/adapter"
R260="$EXP14/results/fortress_v11_harmbench_genalign/round_260/adapter"
V11P="$EXP11/data/grpo_prompts/bcb_test_128_v11prefix.jsonl"

run() {  # 1=run-name 2=checkpoint
    echo "===== [no-CoT] $1 ====="
    "$LW" scripts/eval_at_checkpoint.py --run-name "$1" --checkpoint "$2" \
        --base-model "$BASE" --bcb-test "$V11P" --cot-mode none --n-prompts 128 --skip-vea \
        || echo "[WARN] $1 failed"
}
run obf_r219_noCoT "$R219"
run obf_r260_noCoT "$R260"
echo "===== DONE ====="
