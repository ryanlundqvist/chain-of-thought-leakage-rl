#!/bin/bash
# Gen+probe (skip VEA), 1000 samples, on COMPUTE. args: RUNNAME CKPT COT SHARDFILE
#SBATCH --job-name=sc1kgp
#SBATCH --account=goodfire
#SBATCH --partition=compute
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --output=logs/scale1k_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
RUNNAME="$1"; CKPT="$2"; COT="$3"; SHARD="$4"
NP=$(wc -l < "$SHARD")
echo "[sc1k] $RUNNAME ckpt=$CKPT cot=$COT shard=$(basename "$SHARD") nprompts=$NP node=$(hostname) start=$(date)"
$PY scripts/eval_at_checkpoint.py \
  --run-name "$RUNNAME" --checkpoint "$CKPT" --base-model "$EXP11/merged_wood_organism" \
  --bcb-test "$SHARD" --n-prompts "$NP" --n-samples 1000 --cot-mode "$COT" \
  --cache-prompt-probe --skip-vea
echo "[sc1k] DONE $RUNNAME $(date)"
