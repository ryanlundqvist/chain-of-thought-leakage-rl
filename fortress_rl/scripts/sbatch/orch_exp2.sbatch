#!/bin/bash
# Experiment 2 orchestrator — Qwen2.5-7B Cyprus RLAIF. 7B LoRA trains on 1 GPU.
#SBATCH --job-name=frl-orch2
#SBATCH --account=goodfire
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/orch_exp2_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
cd "$FORTRESS_DIR"
mkdir -p logs results
exec bash scripts/run_exp2_cyprus.sh
