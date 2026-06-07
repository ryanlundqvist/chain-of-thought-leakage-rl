#!/bin/bash
#SBATCH --job-name=ex-reprobe
#SBATCH --account=goodfire
#SBATCH --partition=dev,compute
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --output=logs/reprobe_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs results
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub"
LW=/home/rlundqvist/lora_wood_v2_venv/bin/python
EXP14="$HOME/Evaluation Awareness Experiments/exp14-eval-awareness-obfuscation-rl-runs"
$LW scripts/extract_contrastive_probe.py BASE BASE
$LW scripts/extract_contrastive_probe.py "$EXP14/results/fortress_v11_harmbench_genalign/round_219/adapter" r219
$LW scripts/extract_contrastive_probe.py "$EXP14/results/fortress_v11_harmbench_genalign/round_260/adapter" r260
echo "REPROBE_DONE"
