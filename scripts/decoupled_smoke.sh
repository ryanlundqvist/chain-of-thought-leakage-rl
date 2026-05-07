#!/bin/bash
# Decoupled-GRPO smoke: run the orchestrator for ONE round, then exit.
# Validates the full gen → score → train → save loop on real infra.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

VLLM_URL="$(cat results/policy_url_decoupled.txt 2>/dev/null || true)"
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$VLLM_URL" ] && { echo "ERROR: VLLM URL missing — start serve_policy_lora.sh first"; exit 2; }
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }

OUTPUT_DIR="results/grpo_runs/decoupled_smoke_$(date +%H%M%S)"

ROUNDS=1 \
INNER_STEPS=4 \
PROMPTS_PER_ROUND=8 \
NUM_GENERATIONS=4 \
SDF_EVERY=0 \
VLLM_URL="$VLLM_URL" \
RM_URL="$RM_URL" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONDITION="leak" \
PROMPTS_FILE="data/grpo_prompts/coding_train.jsonl" \
exec bash scripts/decoupled_orchestrate.sh
