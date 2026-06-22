#!/bin/bash
# Generic dual-server (49B + gpt-oss-120b) launcher dispatching any resampling method.
#   env: METHOD (iterative|excision|fromprefix), TAG, RUNS, plus method opts:
#        MAXR (iterative), CAP (cap-per-pid), LIMIT, SAVE, SHARD, CONC
#SBATCH --job-name=rmethod
#SBATCH --account=goodfire
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --output=logs/rmethod_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs results/resampling
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn TMPDIR=/tmp
METHOD="${METHOD:?set METHOD}"; TAG="${TAG:?set TAG}"; RUNS="${RUNS:?set RUNS}"
GPORT=$((8400 + SLURM_JOB_ID % 300)); MPORT=$((8700 + SLURM_JOB_ID % 300))
echo "[rmethod] node=$(hostname) METHOD=$METHOD TAG=$TAG RUNS=$RUNS gen:$GPORT mon:$MPORT start=$(date)"

CUDA_VISIBLE_DEVICES=0,1 $PY -m vllm.entrypoints.openai.api_server \
  --model "$EXP11/merged_wood_organism" --tensor-parallel-size 2 --host 127.0.0.1 --port $GPORT \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code \
  > "logs/rmethod_gen_${SLURM_JOB_ID}.log" 2>&1 &
GPID=$!
CUDA_VISIBLE_DEVICES=2,3 $PY -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b --tensor-parallel-size 2 --host 127.0.0.1 --port $MPORT \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --download-dir "$HF_HUB_CACHE" --trust-remote-code \
  > "logs/rmethod_mon_${SLURM_JOB_ID}.log" 2>&1 &
MPID=$!
trap 'kill $GPID $MPID 2>/dev/null || true' EXIT
for url in "$GPORT" "$MPORT"; do
  for i in $(seq 1 240); do curl -s -m4 "http://127.0.0.1:${url}/v1/models" 2>/dev/null | grep -q '"id"' && { echo "ready @ $url ($((i*10))s)"; break; }; sleep 10; done
done

case "$METHOD" in
  iterative)  SCRIPT="scripts/resample_iterative.py";   EXTRA="--max-rounds ${MAXR:-12}";;
  excision)   SCRIPT="scripts/resample_excision.py";     EXTRA="";;
  fromprefix) SCRIPT="scripts/resample_from_prefix.py";  EXTRA="--K ${MAXR:-20}";;
  *) echo "unknown METHOD $METHOD"; exit 2;;
esac
$PY $SCRIPT --runs $RUNS --tag "$TAG" \
  --gen-url "http://127.0.0.1:${GPORT}/v1" --mon-url "http://127.0.0.1:${MPORT}/v1" \
  $EXTRA ${CAP:+--cap-per-pid $CAP} ${LIMIT:+--limit $LIMIT} ${SAVE:+--save-text} ${SHARD:+--shard $SHARD} --concurrency ${CONC:-48}
echo "[rmethod] DONE $METHOD $TAG $(date)"
