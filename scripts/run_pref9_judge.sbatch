#!/bin/bash
# Phase-2 concurrent VEA judge over the 4 pref9 gen+probe files. Dependency on the 4 gen jobs.
#SBATCH --job-name=pref9jud
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=3:00:00
#SBATCH --output=logs/pref9judge_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
PORT=$((8200 + SLURM_JOB_ID % 700))
echo "[pref9judge] node=$(hostname) port=$PORT start=$(date)"
$PY -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b \
  --tensor-parallel-size 2 --host 0.0.0.0 --port $PORT --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --download-dir "$HF_HUB_CACHE" --trust-remote-code \
  > "logs/pref9judge_serve_${SLURM_JOB_ID}.log" 2>&1 &
JPID=$!
trap 'kill $JPID 2>/dev/null || true' EXIT
URL="http://localhost:$PORT/v1"
for i in $(seq 1 240); do curl -s -m4 "$URL/models" >/dev/null 2>&1 && { echo "judge ready @ $PORT after $((i*10))s"; break; }; sleep 10; done
RUNS="$*"; [ -z "$RUNS" ] && RUNS="pref9_BASE_a pref9_BASE_b pref9_r260_a pref9_r260_b"
for RUN in $RUNS; do
  JF=$(ls results/grpo_runs/$RUN/eval/*.jsonl 2>/dev/null | grep -v summary | head -1)
  [ -f "$JF" ] && $PY scripts/vea_rejudge_concurrent.py --jsonl "$JF" --rm-url "$URL" --all --concurrency 96
done
kill $JPID 2>/dev/null || true
echo "[pref9judge] DONE $(date)"
