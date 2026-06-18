#!/bin/bash
# Phase 1: gen + probe (skip VEA) for one (model × cot-mode × prompt-shard) cell of the
# 200-sample obfuscated-prompt scale-up. Writes rollouts + type-hint + regex-VEA + lengths +
# probe@prompt (cached/prompt) + probe@CoT (per-sample). VEA judge added in phase 2.
#   args: RUNNAME  CKPT(BASE|adapter-path)  COT(natural|forced)  SHARDFILE
#SBATCH --job-name=obf200gp
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/obf200_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
RUNNAME="$1"; CKPT="$2"; COT="$3"; SHARD="$4"
NP=$(wc -l < "$SHARD")
echo "[obf200] $RUNNAME ckpt=$CKPT cot=$COT shard=$(basename "$SHARD") nprompts=$NP node=$(hostname) start=$(date)"
$PY scripts/eval_at_checkpoint.py \
  --run-name "$RUNNAME" --checkpoint "$CKPT" --base-model "$EXP11/merged_wood_organism" \
  --bcb-test "$SHARD" --n-prompts "$NP" --n-samples 200 --cot-mode "$COT" \
  --cache-prompt-probe --skip-vea
echo "[obf200] DONE $RUNNAME $(date)"
