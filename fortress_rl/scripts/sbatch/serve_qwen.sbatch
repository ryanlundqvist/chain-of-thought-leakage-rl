#!/bin/bash
# Long-lived Qwen2.5-7B policy serve (LoRA hot-swap enabled) for Exp2.
# Env: TAG (URL-file suffix, e.g. exp2cyprus) PORT
#SBATCH --job-name=frl-qwen
#SBATCH --account=goodfire
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/serve_qwen_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
EXP11="$(dirname "$FORTRESS_DIR")"
EAE="$(dirname "$EXP11")"
cd "$FORTRESS_DIR"
mkdir -p logs results

QWEN_BASE="$(ls -d "$EAE"/exp8-localization_of_ea_and_probes/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1)"
if [ -z "$QWEN_BASE" ]; then echo "FATAL: Qwen2.5-7B snapshot not found"; exit 2; fi

MODEL="${QWEN_BASE%/}" \
TAG="${TAG:-qwen}" PORT="${PORT:-8002}" TP=1 MAX_MODEL_LEN=8192 \
exec bash scripts/serve_policy.sh
