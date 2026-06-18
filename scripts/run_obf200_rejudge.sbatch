#!/bin/bash
# Re-judge INVALID VEA labels across the 8 obf200 files with a concurrent judge (bigger budget).
#SBATCH --job-name=obf200rej
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --output=logs/obf200rej_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
PORT=$((8100 + SLURM_JOB_ID % 800))
echo "[rejudge] node=$(hostname) port=$PORT start=$(date)"
$PY -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b \
  --tensor-parallel-size 2 --host 0.0.0.0 --port $PORT --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --download-dir "$HF_HUB_CACHE" --trust-remote-code \
  > "logs/obf200rej_serve_${SLURM_JOB_ID}.log" 2>&1 &
JPID=$!
trap 'kill $JPID 2>/dev/null || true' EXIT
URL="http://localhost:$PORT/v1"
for i in $(seq 1 240); do curl -s -m4 "$URL/models" >/dev/null 2>&1 && { echo "judge ready @ $PORT after $((i*10))s"; break; }; sleep 10; done
for RUN in obf200_BASE_nat_a obf200_BASE_nat_b obf200_BASE_for_a obf200_BASE_for_b \
           obf200_r260_nat_a obf200_r260_nat_b obf200_r260_for_a obf200_r260_for_b; do
  JF=$(ls results/grpo_runs/$RUN/eval/*.jsonl 2>/dev/null | grep -v summary | head -1)
  [ -f "$JF" ] && $PY scripts/vea_rejudge_concurrent.py --jsonl "$JF" --rm-url "$URL" --concurrency 96
done
kill $JPID 2>/dev/null || true
echo "[rejudge] DONE $(date)"
