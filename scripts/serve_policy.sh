#!/bin/bash
# Serve merged_wood_organism via `trl vllm-serve` for GRPO server mode.
# One server per training condition; persists for full training.
#
# Expects env: PORT (default 8001), TP (default 4), TAG (used in URL file name),
#              PROJECT_DIR.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

PORT="${PORT:-8001}"
TP="${TP:-4}"
TAG="${TAG:-policy}"
HOSTNAME_S="$(hostname -s)"

mkdir -p logs results

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

URL="http://${HOSTNAME_S}:${PORT}"
echo "$URL" > "results/policy_url_${TAG}.txt"
echo "[$(date)] Policy serve ($TAG) at $URL with TP=$TP"

exec "$PROJECT_DIR/venv/bin/trl" vllm-serve \
    --model "$PROJECT_DIR/merged_wood_organism" \
    --tensor_parallel_size "$TP" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max_model_len 10240 \
    --gpu_memory_utilization 0.85 \
    --trust_remote_code \
    --dtype bfloat16
