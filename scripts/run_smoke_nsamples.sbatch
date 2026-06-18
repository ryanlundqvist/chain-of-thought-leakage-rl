#!/bin/bash
#SBATCH --job-name=ex-smoke-ns
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=0:20:00
#SBATCH --output=logs/smoke_ns_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub" "$EXP11/.hf_cache/datasets" "$EXP11/.hf_cache/transformers"
/home/rlundqvist/lora_wood_v2_venv/bin/python scripts/eval_at_checkpoint.py \
  --run-name SMOKE_nsamp --checkpoint BASE --base-model "$EXP11/merged_wood_organism" \
  --bcb-test "$EXP11/data/grpo_prompts/bcb_test_128_v11prefix.jsonl" \
  --cot-mode natural --n-prompts 4 --n-samples 2 --skip-vea --skip-probe
echo SMOKE_DONE
