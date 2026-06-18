#!/bin/bash
# Judge-comparison: serve gpt-oss-20b as the cheap constitutional judge.
# 20b bf16 ~39 GB -> fits on 1 H100. Mirrors serve_rm.sh (proven for 120b).
#SBATCH --job-name=jc-rm20b
#SBATCH --account=goodfire
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=judge_comparison/logs/serve20b_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
EXP11="$(dirname "$FORTRESS_DIR")"
cd "$FORTRESS_DIR"
mkdir -p judge_comparison/logs judge_comparison/results

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME="$EXP11/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

PORT=8002
URL="http://$(hostname -s):${PORT}/v1"
echo "$URL" > judge_comparison/results/rm20b_url.txt
echo "[serve-20b] gpt-oss-20b judge at $URL"

exec "$EXP11/venv_openrlhf/bin/python" -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HUB_CACHE" \
    --trust-remote-code \
    --enforce-eager \
    --seed 42
