#!/bin/bash
# FORTRESS-RL — serve a policy model via vLLM with --enable-lora (decoupled GRPO).
# The trainer saves a LoRA adapter to disk each round; the orchestrator hot-loads
# it via the /v1/load_lora_adapter admin endpoint.
#
# Env: MODEL (path or HF id; default = Wood organism)  TP (2)  PORT (8001)
#      TAG (URL file suffix)  MAX_MODEL_LEN (10240)  HF_HUB_CACHE_OVERRIDE (opt)
set -euo pipefail

FORTRESS_DIR="${FORTRESS_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl}"
EXP11_DIR="$(dirname "$FORTRESS_DIR")"
cd "$EXP11_DIR"

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
# Cluster-canonical shared HF cache (set by /data/scripts/clusters/<id>.sh).
export HF_HOME="/data/artifacts/rlundqvist/hf_cache"
export HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

MODEL="${MODEL:-$EXP11_DIR/merged_wood_organism}"
TP="${TP:-2}"
PORT="${PORT:-8001}"
TAG="${TAG:-policy}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
HOSTNAME_S="$(hostname -s)"

mkdir -p "$FORTRESS_DIR/logs" "$FORTRESS_DIR/results"
URL="http://${HOSTNAME_S}:${PORT}"
echo "$URL" > "$FORTRESS_DIR/results/policy_url_${TAG}.txt"
echo "[serve-policy] $TAG: $MODEL at $URL  TP=$TP  max_model_len=$MAX_MODEL_LEN  LoRA on"

exec /home/rlundqvist/lora_wood_v2_venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.80 \
    --trust-remote-code \
    --dtype bfloat16 \
    \
    --enable-lora --max-lora-rank 64 --max-loras 1 --max-cpu-loras 1 \
    --enable-prefix-caching \
    --disable-custom-all-reduce --enforce-eager \
    --seed 42
