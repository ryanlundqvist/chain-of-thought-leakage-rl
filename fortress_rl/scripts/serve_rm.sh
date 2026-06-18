#!/bin/bash
# FORTRESS-RL — serve gpt-oss-120b as the constitutional judge / reward model.
# Shared by all three runs (Exp1 leak, Exp1 no-leak, Exp2 cyprus).
# Writes its URL to fortress_rl/results/rm_url.txt the moment it starts.
#
# Env: PORT (8000)  TP (4)
set -euo pipefail

FORTRESS_DIR="${FORTRESS_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl}"
EXP11_DIR="$(dirname "$FORTRESS_DIR")"
cd "$EXP11_DIR"

PORT="${PORT:-8000}"
TP="${TP:-4}"
# Judge model — env-overridable so the same serve script hosts gpt-oss-120b
# (default) or the cheaper gpt-oss-20b (RM_MODEL=openai/gpt-oss-20b TP=1).
RM_MODEL="${RM_MODEL:-openai/gpt-oss-120b}"
HOSTNAME_S="$(hostname -s)"

mkdir -p "$FORTRESS_DIR/logs" "$FORTRESS_DIR/results"

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
# Cluster-canonical shared HF cache (set by /data/scripts/clusters/<id>.sh).
export HF_HOME="/data/artifacts/rlundqvist/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

URL="http://${HOSTNAME_S}:${PORT}/v1"
# RM_URL_OUT lets a second judge serve (e.g. the cheap 20b) write its own URL
# file instead of clobbering the 120b's results/rm_url.txt.
RM_URL_OUT="${RM_URL_OUT:-$FORTRESS_DIR/results/rm_url.txt}"
echo "$URL" > "$RM_URL_OUT"
echo "[serve-rm] judge model=$RM_MODEL at $URL  TP=$TP"

# max-model-len 16384: constitution (~1k tok) + judge template + user prompt
# (up to ~2.4k tok) + the leaked transcript (CoT up to ~6k tok + response up to
# ~2.4k tok). cyprus_rl/venv has the vLLM that the exp11 RM serve used.
exec "$EXP11_DIR/cyprus_rl/venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$RM_MODEL" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HUB_CACHE" \
    --trust-remote-code \
    --disable-custom-all-reduce --enforce-eager \
    --seed 42
