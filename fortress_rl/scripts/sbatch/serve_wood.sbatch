#!/bin/bash
# Long-lived Wood-organism policy serve (LoRA hot-swap enabled). One per Exp1 arm.
# Env: TAG (URL-file suffix, e.g. exp1leak) PORT
#SBATCH --job-name=frl-wood
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=168:00:00
#SBATCH --output=logs/serve_wood_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
EXP11="$(dirname "$FORTRESS_DIR")"
cd "$FORTRESS_DIR"
mkdir -p logs results
MODEL="$EXP11/merged_wood_organism" \
TAG="${TAG:-wood}" PORT="${PORT:-8001}" TP=2 MAX_MODEL_LEN=16384 \
exec bash scripts/serve_policy_vllm010.sh
