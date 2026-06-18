#!/bin/bash
# probe@CoT on from-prefix resampled-non-VEA continuations (BASE-generated -> BASE model).
#   env: TAG (resample output produced with --save-text), PROMPT_RUNS (space-sep), MAXPID (opt)
#SBATCH --job-name=probe-rs
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --output=logs/probe_rs_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs results/resampling
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
export PYTHONPATH="/home/rlundqvist/eval_peft_pkg:$EXP11/scripts"
TAG="${TAG:?set TAG}"; PROMPT_RUNS="${PROMPT_RUNS:?set PROMPT_RUNS}"
echo "[probe-rs] node=$(hostname) TAG=$TAG start=$(date)"
# merge shard outputs (TAG_s0,TAG_s1,...) into TAG.jsonl if present
shards=$(ls results/resampling/${TAG}_s*.jsonl 2>/dev/null)
if [ -n "$shards" ]; then echo "[probe-rs] merging shards: $shards"; cat $shards > results/resampling/${TAG}.jsonl; fi
echo "[probe-rs] $(wc -l < results/resampling/${TAG}.jsonl) resample rollouts in ${TAG}.jsonl"
$PY scripts/probe_resampled.py --tag "$TAG" --prompt-runs $PROMPT_RUNS ${MAXPID:+--max-per-pid $MAXPID}
echo "[probe-rs] DONE $(date)"
