#!/bin/bash
#SBATCH --job-name=ex-probe
#SBATCH --account=goodfire
#SBATCH --partition=dev,compute
#SBATCH --gres=gpu:2
#SBATCH --time=2:30:00
#SBATCH --output=logs/probe_%j.log
# args: $1 run-name  $2 checkpoint  $3 prompts-file  $4 cot-mode
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub" "$EXP11/.hf_cache/datasets" "$EXP11/.hf_cache/transformers"
# 256 prompts x 6 samples WITH probe (probe@prompt + probe@CoT + type-hint + VEA-regex); no 120b judge.
/home/rlundqvist/lora_wood_v2_venv/bin/python scripts/eval_at_checkpoint.py \
  --run-name "$1" --checkpoint "$2" --base-model "$EXP11/merged_wood_organism" \
  --bcb-test "$3" --cot-mode "$4" --n-prompts 256 --n-samples 6 --skip-vea
echo "PROBE_DONE $1"
