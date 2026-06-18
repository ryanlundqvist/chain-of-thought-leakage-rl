#!/bin/bash
#SBATCH --job-name=ex-scaled
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=3:00:00
#SBATCH --output=logs/scaled_%j.log
# args: $1 = run-name, $2 = checkpoint (path or "BASE")
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub" "$EXP11/.hf_cache/datasets" "$EXP11/.hf_cache/transformers"
# Scaled: all 256 BCB prompts x 6 samples/prompt = 1536 rollouts. natural CoT, v11 prefix.
# Saves rollouts + prompt-end activations + probe@prompt + probe@CoT + type-hint + VEA-regex. NO 120b judge.
/home/rlundqvist/lora_wood_v2_venv/bin/python scripts/eval_at_checkpoint.py \
  --run-name "$1" --checkpoint "$2" --base-model "$EXP11/merged_wood_organism" \
  --bcb-test "$EXP11/data/grpo_prompts/bcb_test_128_v11prefix.jsonl" \
  --cot-mode natural --n-prompts 256 --n-samples 6 --skip-vea
echo "SCALED_DONE $1"
