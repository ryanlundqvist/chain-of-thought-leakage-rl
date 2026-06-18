#!/bin/bash
#SBATCH --job-name=ex-vjudge
#SBATCH --account=goodfire
#SBATCH --partition=dev,compute
#SBATCH --gres=gpu:2
#SBATCH --time=3:00:00
#SBATCH --output=logs/vjudge_%j.log
# args: $1 port  $2.. = vj cell names (e.g. vj_r219_pf_natural)
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
LW=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
PORT=$1; shift; CELLS="$@"
$LW -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b \
    --tensor-parallel-size 2 --host 0.0.0.0 --port $PORT --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --download-dir "$HF_HUB_CACHE" --trust-remote-code \
    > "logs/vjudge_serve_${SLURM_JOB_ID}.log" 2>&1 &
JPID=$!
URL="http://localhost:$PORT/v1"
for i in $(seq 1 180); do curl -s -m4 "$URL/models" >/dev/null 2>&1 && { echo "judge ready @ $PORT"; break; }; sleep 10; done
# judge each assigned cell concurrently (vLLM batches the parallel requests)
for c in $CELLS; do
  JF="results/grpo_runs/$c/eval/sub.jsonl"
  [ -f "$JF" ] && $LW scripts/vea_judge_jsonl.py --jsonl "$JF" --rm-url "$URL" > "logs/vjudge_${c}_${SLURM_JOB_ID}.log" 2>&1 &
done
wait
kill $JPID 2>/dev/null || true
echo "VJUDGE_DONE $CELLS"
