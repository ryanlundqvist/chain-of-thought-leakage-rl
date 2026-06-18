#!/bin/bash
# Serve gpt-oss-120b via vLLM as a reward model + judge endpoint.
#
# Sourced by sbatch wrappers; expects:
#   PORT (default 8000)
#   TP   (default 4)
#   PROJECT_DIR
#
# After this exits, the shared file results/mvp/rm_url.txt holds the URL.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

PORT="${PORT:-8000}"
TP="${TP:-4}"
HOSTNAME_S="$(hostname -s)"

mkdir -p logs results/mvp

export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Write URL marker the moment the script starts so other jobs can `wait` on it.
URL="http://${HOSTNAME_S}:${PORT}/v1"
echo "$URL" > results/mvp/rm_url.txt
echo "[$(date)] Serving gpt-oss-120b at $URL with TP=$TP" | tee logs/rm_serve.log

exec "$PROJECT_DIR/venv_openrlhf/bin/python" -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HUB_CACHE" \
    --trust-remote-code \
    --seed 42
