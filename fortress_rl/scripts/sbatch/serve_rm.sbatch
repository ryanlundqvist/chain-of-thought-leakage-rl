#!/bin/bash
# Long-lived gpt-oss-120b constitutional judge / reward serve. Shared by all arms.
#SBATCH --job-name=frl-rm
#SBATCH --account=goodfire
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/serve_rm_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
cd "$FORTRESS_DIR"
mkdir -p logs results
PORT="${PORT:-8000}" TP=4 exec bash scripts/serve_rm.sh
