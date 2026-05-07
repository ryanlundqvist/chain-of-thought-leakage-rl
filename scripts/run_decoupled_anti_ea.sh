#!/bin/bash
# Run B — decoupled GRPO with the ANTI-EA constitution (explicit signal).
# Designed to be resumable: same OUTPUT_DIR on resubmit will skip completed rounds.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

VLLM_URL="$(cat results/policy_url_decoupled.txt 2>/dev/null || true)"
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$VLLM_URL" ] && { echo "ERROR: VLLM URL missing"; exit 2; }
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }
echo "vLLM serve: $VLLM_URL"
echo "RM serve:   $RM_URL"

# Stable output dir (lets resubmit pick up where we left off)
OUTPUT_DIR="${OUTPUT_DIR:-results/grpo_runs/decoupled_anti_ea}"
mkdir -p "$OUTPUT_DIR"

# Conservative size for an initial comparison run on 4 trainer GPUs.
# Each round: ~12 min  (gen 2 min + score 1 min + train 8 min for 20 inner steps)
# 30 rounds × 12 min = ~6 h    → ~150 effective gradient steps
# Bump ROUNDS later for full-scale; resume picks up from last completed round.
ROUNDS="${ROUNDS:-30}" \
INNER_STEPS="${INNER_STEPS:-20}" \
PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-32}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-8}" \
SDF_EVERY="${SDF_EVERY:-5}" \
SDF_DOCS="${SDF_DOCS:-300}" \
VLLM_URL="$VLLM_URL" \
RM_URL="$RM_URL" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONDITION="leak" \
CONSTITUTION="data/claude_constitution_anti_ea.txt" \
PROMPTS_FILE="data/grpo_prompts/coding_train.jsonl" \
exec bash scripts/decoupled_orchestrate.sh
