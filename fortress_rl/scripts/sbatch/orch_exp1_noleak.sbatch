#!/bin/bash
# Experiment 1 NO-LEAK orchestrator (the causal control arm). 2-GPU decoupled
# training footprint, matching exp11.
#SBATCH --job-name=frl-orch1N
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/orch_exp1_noleak_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
cd "$FORTRESS_DIR"
mkdir -p logs results
exec bash scripts/run_exp1_no_leak.sh
