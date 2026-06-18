#!/bin/bash
# Experiment 1 LEAK orchestrator. Trains the Wood 49B LoRA (device_map=auto over
# the job's GPUs — 2 GPUs, matching exp11's proven decoupled-train footprint).
# Reads serve URLs from results/*.txt; resumable on re-submit.
#SBATCH --job-name=frl-orch1L
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/orch_exp1_leak_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
cd "$FORTRESS_DIR"
mkdir -p logs results
exec bash scripts/run_exp1_leak.sh
