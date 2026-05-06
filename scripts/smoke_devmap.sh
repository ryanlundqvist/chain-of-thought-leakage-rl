#!/bin/bash
# Smoke test for Tim/Andrew's BF16 model-parallel pattern adapted to GRPO.
#
# Single Python process, model split across visible GPUs via device_map="auto".
# No FSDP/ZeRO-3/DDP. No vLLM colocate (--no-vllm forced). HF generation.
#
# Submit with:
#   sbatch -A goodfire --gres=gpu:4 --time=02:00:00 \
#     --output=logs/smoke_devmap.%j.log \
#     --wrap='bash scripts/smoke_devmap.sh'

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$PROJECT_DIR/venv/bin/python"
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }
echo "RM URL: $RM_URL"

OUTPUT_DIR="results/grpo_runs/smoke_devmap_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "===================================================="
echo "device_map='auto' BF16 model-parallel GRPO smoke"
echo "  GPUs visible: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  output: $OUTPUT_DIR"
echo "  HF generation (no vLLM)"
echo "===================================================="

# IMPORTANT: run via plain python, NOT accelerate launch.
# accelerate launch would set WORLD_SIZE>1 and try to wrap the model
# in DDP, which fights device_map="auto".
exec "$PYTHON" scripts/train_grpo.py \
    --condition leak --prompts coding \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 3 --save-steps 2 --logging-steps 1 \
    --per-device-batch-size 4 --grad-accum-steps 1 \
    --num-generations 4 \
    --max-prompt-length 1024 --max-completion-length 1024 \
    --learning-rate 1e-4 --lora-rank 32 \
    --device-map-auto \
    --max-train-prompts 16 \
    --rm-url "$RM_URL"
