#!/bin/bash
#SBATCH --job-name=ex-s20
#SBATCH --account=goodfire
#SBATCH --partition=dev,compute
#SBATCH --gres=gpu:2
#SBATCH --time=2:30:00
#SBATCH --output=logs/s20_%j.log
# args: $1 run-name  $2 checkpoint  $3 prompts-file  $4 cot-mode
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub" "$EXP11/.hf_cache/datasets" "$EXP11/.hf_cache/transformers"
# 256 prompts x 20 samples, gen-only (--skip-probe = no slow CoT-probe; --skip-vea = no 120b judge).
# Tightens type-hint + VEA-regex RATE bars (fixed-task CI). Probe stays from existing runs.
/home/rlundqvist/lora_wood_v2_venv/bin/python scripts/eval_at_checkpoint.py \
  --run-name "$1" --checkpoint "$2" --base-model "$EXP11/merged_wood_organism" \
  --bcb-test "$3" --cot-mode "$4" --n-prompts 256 --n-samples 20 --skip-probe --skip-vea
echo "S20_DONE $1"
