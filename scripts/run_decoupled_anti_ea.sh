#!/bin/bash
# Run B — decoupled GRPO with the ANTI-EA constitution (explicit signal).
# Designed to be resumable: same OUTPUT_DIR on resubmit will skip completed rounds.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

VLLM_URL_1="$(cat results/policy_url_decoupled.txt 2>/dev/null || true)"
VLLM_URL_2="$(cat results/policy_url_decoupled2.txt 2>/dev/null || true)"
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$VLLM_URL_1" ] && { echo "ERROR: VLLM URL 1 missing"; exit 2; }
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }
# Concatenate URLs comma-separated (decoupled_generate.py round-robins)
if [ -n "$VLLM_URL_2" ]; then
    VLLM_URL="$VLLM_URL_1,$VLLM_URL_2"
    echo "vLLM serves: $VLLM_URL_1 + $VLLM_URL_2"
else
    VLLM_URL="$VLLM_URL_1"
    echo "vLLM serve:  $VLLM_URL_1"
fi
echo "RM serve:    $RM_URL"

# Stable output dir (lets resubmit pick up where we left off)
OUTPUT_DIR="${OUTPUT_DIR:-results/grpo_runs/decoupled_anti_ea}"
mkdir -p "$OUTPUT_DIR"

# Production scale (8 trainer GPUs + 2 vLLM serves):
#   per round: ~3-4 min  (gen 80s + score 60s + train 90s + maybe SDF 5min)
#   200 rounds × ~4 min = ~13h    → ~1000 effective gradient steps
#   SDF every 3 rounds = 67 SDF rounds (≈ frequent re-anchor)
ROUNDS="${ROUNDS:-200}" \
INNER_STEPS="${INNER_STEPS:-20}" \
PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-32}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-8}" \
SDF_EVERY="${SDF_EVERY:-5}" \
SDF_DOCS="${SDF_DOCS:-25}" \
VLLM_URL="$VLLM_URL" \
RM_URL="$RM_URL" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONDITION="leak" \
CONSTITUTION="data/claude_constitution_anti_ea.txt" \
PROMPTS_FILE="data/grpo_prompts/coding_train.jsonl" \
exec bash scripts/decoupled_orchestrate.sh
