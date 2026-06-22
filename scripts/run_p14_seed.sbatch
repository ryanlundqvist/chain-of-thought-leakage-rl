#!/bin/bash
# Cross-seed replication for the 00009 x 14-prefix obfuscation test. One-shot gen + inline VEA
# classification (eval_at_checkpoint loads gpt-oss-120b itself). Re-samples a fresh seed so the
# deconfounded NONE-vs-NONE gaps can be checked for seed-stability (the _a/_b files were a prefix
# PARTITION, not seed replicates).
#   args: RUNNAME  CKPT(BASE|adapterpath)  SEED
#SBATCH --job-name=p14seed
#SBATCH --account=goodfire
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --output=logs/p14seed_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
RUNNAME="$1"; CKPT="$2"; SEED="$3"; SHARD="$EXP11/data/grpo_prompts/p14_all.jsonl"
NP=$(wc -l < "$SHARD")
echo "[p14seed] $RUNNAME ckpt=$CKPT seed=$SEED nprompts=$NP node=$(hostname) start=$(date)"
$PY scripts/eval_at_checkpoint.py \
  --run-name "$RUNNAME" --checkpoint "$CKPT" --base-model "$EXP11/merged_wood_organism" \
  --bcb-test "$SHARD" --n-prompts "$NP" --n-samples 400 --cot-mode natural \
  --seed "$SEED" --temperature 0.7 --top-p 0.95 --cache-prompt-probe
echo "[p14seed] DONE $RUNNAME $(date)"
