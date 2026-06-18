#!/bin/bash
# Fresh type-hint / VEA-regex / probe eval on the CURRENT v11 policy (post-retune).
# Checkpoints: r219 (best obfuscation round: low CoT-VEA + high refusal) and r260 (current).
# Phase-1 ONLY: gen + probe + type-hint + VEA-regex (--skip-vea = NO gpt-oss-120b serve).
#   -> intentionally avoids spinning a 120b judge on the shared dev node (that starves the
#      live v11 wood/trainer host RAM; incident 2026-06-03). VEA-regex is the proxy here;
#      add the 120b VEA-judge later on the compute partition when it frees up.
# Same harness/config as the baseline + obf evals: eval_at_checkpoint, v11/Wood-Labs prefix,
# natural CoT, n=128, same probe (base_user_and_simple.pt).
#SBATCH --job-name=ex-v11-theval
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=1:30:00
#SBATCH --output=logs/v11_theval_%j.log
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
    echo "===== [gen+probe] $1  (ckpt=$2) ====="
    "$LW" scripts/eval_at_checkpoint.py --run-name "$1" --checkpoint "$2" \
        --base-model "$BASE" --bcb-test "$V11P" --cot-mode natural --n-prompts 128 --skip-vea \
        || echo "[gen] WARN $1 failed"
}
run obf_r219 "$R219"
run obf_r260 "$R260"
echo "===== DONE (type-hint + VEA-regex + probe written under results/grpo_runs/{obf_r219,obf_r260}/eval/) ====="
