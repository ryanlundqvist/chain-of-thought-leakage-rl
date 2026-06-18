#!/bin/bash
# VEA-elicitation exploration — find framings that elicit verbalized
# eval-awareness in Qwen2.5-7B-Instruct's final response. Single-GPU vLLM,
# offline, loads the local Qwen snapshot directly (no serve needed).
#SBATCH --job-name=vea-elicit
#SBATCH --account=goodfire
#SBATCH --gres=gpu:1
#SBATCH --time=0:40:00
#SBATCH --output=results/vea_elicitation/logs/gen_%j.log

set -uo pipefail

EAE="$HOME/Evaluation Awareness Experiments"
EXP11="$EAE/exp11_cot_leakage"
CYPRUS="$EXP11/cyprus_rl"
cd "$CYPRUS"

QWEN="$(ls -d "$EAE"/exp8-localization_of_ea_and_probes/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1)"
[ -z "$QWEN" ] && { echo "FATAL: Qwen2.5-7B-Instruct snapshot not found"; exit 2; }
QWEN="${QWEN%/}"
echo "[sbatch] Qwen snapshot: $QWEN"

export HF_HOME="$EXP11/.hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
# flashinfer-cubin 0.6.6 vs flashinfer 0.5.2 mismatch in this venv -> bypass
# the version check (vLLM falls back cleanly).
export FLASHINFER_DISABLE_VERSION_CHECK=1

exec "$CYPRUS/venv/bin/python" scripts/vea_elicitation_gen.py \
    --model "$QWEN" \
    --prompts "$CYPRUS/data/cyprus_grpo_train.jsonl" \
    --out-dir "$CYPRUS/results/vea_elicitation" \
    --n 50 --temperature 0.7 --top-p 0.95 --max-tokens 1024
