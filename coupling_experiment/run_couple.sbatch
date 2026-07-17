#!/bin/bash
#SBATCH --job-name=couple
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --output=logs/couple_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
RUNNAME="$1"; CKPT="$2"; SEED="$3"; SHARD="$4"; NS="${5:-60}"
NP=$(wc -l < "$SHARD")
echo "[couple] $RUNNAME ckpt=$CKPT file=$(basename "$SHARD") nrows=$NP nsamp=$NS node=$(hostname) $(date)"
# --skip-vea (offline judge) but KEEP probe (probe_score_avg=@CoT) + --cache-prompt-probe (@prompt)
$PY scripts/eval_at_checkpoint.py --run-name "$RUNNAME" --checkpoint "$CKPT" \
  --base-model "$EXP11/merged_wood_organism" --bcb-test "$SHARD" --n-prompts "$NP" \
  --n-samples "$NS" --cot-mode natural --seed "$SEED" --temperature 0.7 --top-p 0.95 \
  --skip-vea --cache-prompt-probe
echo "[couple] DONE $RUNNAME $(date)"
