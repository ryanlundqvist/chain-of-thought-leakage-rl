#!/bin/bash
# Phase 2: VEA judge over the 8 phase-1 output files. Starts one gpt-oss-120b server and runs
# vea_judge_jsonl.py on all 8 files concurrently (server batches the streams). Fills vea_label
# in place. Submit with --dependency=afterok:<8 phase-1 job ids>.
#SBATCH --job-name=obf200jud
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --output=logs/obf200judge_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
PORT=$((8000 + SLURM_JOB_ID % 900))
echo "[obf200judge] node=$(hostname) port=$PORT start=$(date)"
$PY -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b \
  --tensor-parallel-size 2 --host 0.0.0.0 --port $PORT --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --download-dir "$HF_HUB_CACHE" --trust-remote-code \
  > "logs/obf200judge_serve_${SLURM_JOB_ID}.log" 2>&1 &
JPID=$!
trap 'kill $JPID 2>/dev/null || true' EXIT
URL="http://localhost:$PORT/v1"
for i in $(seq 1 240); do curl -s -m4 "$URL/models" >/dev/null 2>&1 && { echo "judge ready @ $PORT after $((i*10))s"; break; }; sleep 10; done
for RUN in obf200_BASE_nat_a obf200_BASE_nat_b obf200_BASE_for_a obf200_BASE_for_b \
           obf200_r260_nat_a obf200_r260_nat_b obf200_r260_for_a obf200_r260_for_b; do
  JF=$(ls results/grpo_runs/$RUN/eval/*.jsonl 2>/dev/null | grep -v summary | head -1)
  if [ -f "$JF" ]; then
    echo "[obf200judge] judging $RUN -> $JF"
    $PY scripts/vea_judge_jsonl.py --jsonl "$JF" --rm-url "$URL" > "logs/obf200judge_${RUN}_${SLURM_JOB_ID}.log" 2>&1 &
  else
    echo "[obf200judge] WARN missing output for $RUN"
  fi
done
wait
kill $JPID 2>/dev/null || true
echo "[obf200judge] OBF200_JUDGE_DONE $(date)"
