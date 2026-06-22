#!/bin/bash
#SBATCH --job-name=lda
#SBATCH --account=goodfire
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --output=logs/lda_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs results/lda
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
ADP="$HOME/Evaluation Awareness Experiments/exp14-eval-awareness-obfuscation-rl-runs/results/fortress_v11_harmbench_genalign/round_260/adapter"
TAG="$1"; ALPHA="$2"; FILE="$3"; NS="${4:-24}"; MT="${5:-2000}"; EXTRA="${6:-}"
echo "[lda] tag=$TAG alpha=$ALPHA file=$(basename "$FILE") ns=$NS mt=$MT node=$(hostname) $(date)"
for attempt in 1 2 3; do
  echo "[lda] attempt $attempt"
  $PY scripts/lda_generate.py --bcb-test "$FILE" --base-model "$EXP11/merged_wood_organism" \
    --adapter "$ADP" --alpha "$ALPHA" --n-samples "$NS" --max-tokens "$MT" --tag "$TAG" $EXTRA && break
  echo "[lda] attempt $attempt FAILED, retrying..."; sleep 5
done
echo "[lda] DONE $TAG $(date)"
