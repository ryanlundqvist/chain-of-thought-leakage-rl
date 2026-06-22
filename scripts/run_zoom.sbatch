#!/bin/bash
#SBATCH --job-name=zoom
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --output=logs/zoom_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
RUNNAME="$1"; CKPT="$2"; SEED="$3"; SHARD="$4"; NS="${5:-200}"
NP=$(wc -l < "$SHARD")
echo "[zoom] $RUNNAME ckpt=$CKPT seed=$SEED file=$(basename "$SHARD") nrows=$NP nsamp=$NS node=$(hostname) $(date)"
# gen-only at gpu:2 (TP=2). --skip-vea defers the 120b judge offline; --skip-probe for speed.
# type_hint_score (AST) + vea_regex are still computed inline (no GPU).
$PY scripts/eval_at_checkpoint.py --run-name "$RUNNAME" --checkpoint "$CKPT" \
  --base-model "$EXP11/merged_wood_organism" --bcb-test "$SHARD" --n-prompts "$NP" \
  --n-samples "$NS" --cot-mode natural --seed "$SEED" --temperature 0.7 --top-p 0.95 \
  --skip-vea --skip-probe
echo "[zoom] DONE $RUNNAME $(date)"
