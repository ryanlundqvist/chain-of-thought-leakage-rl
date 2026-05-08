#!/bin/bash
# vLLM-serve for the policy with --enable-lora, used by decoupled GRPO.
# Trainer saves LoRA to disk; orchestrator hot-loads via /v1/load_lora_adapter API.
#
# Submit with:
#   sbatch -A goodfire --gres=gpu:4 --time=12:00:00 \
#     --output=logs/policy_lora.%j.log \
#     --wrap='bash scripts/serve_policy_lora.sh'

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
# Allow /v1/load_lora_adapter / /v1/unload_lora_adapter admin endpoints
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Use venv_openrlhf (has vllm 0.19.1 with better LoRA hot-swap support).
# exp11 venv is for trainer-side; vLLM serve is independent.
PYTHON="${SERVE_PYTHON:-$PROJECT_DIR/venv_openrlhf/bin/python}"
TP="${TP:-2}"
PORT="${PORT:-8001}"
TAG="${TAG:-decoupled}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
HOSTNAME_S="$(hostname -s)"

URL="http://${HOSTNAME_S}:${PORT}"
mkdir -p results
echo "$URL" > "results/policy_url_${TAG}.txt"
echo "[serve] policy serve at $URL with TP=$TP, LoRA enabled, max_model_len=$MAX_MODEL_LEN, TAG=$TAG"

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$PROJECT_DIR/merged_wood_organism" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --dtype bfloat16 \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 2 \
    --max-cpu-loras 4
