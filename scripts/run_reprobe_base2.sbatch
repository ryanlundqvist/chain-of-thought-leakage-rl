#!/bin/bash
#SBATCH --job-name=ex-reprobe2
#SBATCH --account=goodfire
#SBATCH --partition=compute
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --output=logs/reprobe2_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs results
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub"
LW=/home/rlundqvist/lora_wood_v2_venv/bin/python
# Independent re-extraction of the model-organism (pre-RL) probe -> reproducibility / noise floor
$LW scripts/extract_contrastive_probe.py BASE BASE2
echo "REPROBE2_DONE"
