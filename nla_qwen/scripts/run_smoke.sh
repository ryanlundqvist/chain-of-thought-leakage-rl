#!/bin/bash
#SBATCH --job-name=nla_smoke
#SBATCH --account=goodfire
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
# NOTE: --output/--error are passed on the sbatch command line because the
# project path contains spaces, which the #SBATCH directive parser cannot
# handle. Submit with: sbatch --output=... --error=... scripts/run_smoke.sh

# --- NLA smoke test: extract Qwen L20 activations on eval/deploy prompts,
#     verbalize them with the real NLA AV, and analyze differentiation. --------
EXP11="/home/rlundqvist/Evaluation Awareness Experiments/exp11_cot_leakage"
NLA="$EXP11/nla_qwen"
PY="$EXP11/cyprus_rl/venv/bin/python"

export HF_HOME="$EXP11/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_CACHE  # the venv profile hardcodes a non-writable /data path

SUBJECT="/home/rlundqvist/Evaluation Awareness Experiments/exp8-localization_of_ea_and_probes/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
AV="$EXP11/.hf_cache/hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots/b88469162777ae6553bc14208eb0cb579336f8f4"
CONTENT="$EXP11/fortress_rl/data/cyprus_pilot.jsonl"

ACTS="$NLA/results/smoke_activations.parquet"
VERB="$NLA/results/smoke_verbalizations.jsonl"
REPORT="$NLA/results/smoke_report.txt"

echo "=== node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Stages are checkpointed: if an output already exists (e.g. after a SLURM
# preemption + requeue) the stage is skipped. This makes the job idempotent.

if [ -s "$ACTS" ]; then
  echo "=== STAGE 1: SKIP (activations already present: $ACTS) ==="
else
  echo "=== STAGE 1: extract layer-20 activations from subject Qwen ==="
  "$PY" "$NLA/scripts/extract_activations.py" \
    --subject-model "$SUBJECT" \
    --out "$ACTS" \
    --hs-index 21 \
    --content-file "$CONTENT" \
    --n-content 8 \
    --batch-size 4
fi

if [ -s "$VERB" ]; then
  echo "=== STAGE 2: SKIP (verbalizations already present: $VERB) ==="
else
  echo "=== STAGE 2: verbalize activations with the real NLA AV ==="
  "$PY" "$NLA/scripts/nla_verbalize.py" \
    --av-model "$AV" \
    --activations "$ACTS" \
    --out "$VERB" \
    --max-new-tokens 128
fi

echo "=== STAGE 3: analyze eval-vs-deploy differentiation ==="
"$PY" "$NLA/scripts/analyze_nla.py" \
  --verbalizations "$VERB" \
  --out "$REPORT" \
  --out-json "$NLA/results/smoke_summary.json"

echo "=== DONE ==="
