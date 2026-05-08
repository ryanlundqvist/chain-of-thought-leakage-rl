#!/bin/bash
# IFEval v3 — same as v2 but heavy empty-CoT penalty.
# v2 collapsed to "empty <think></think>" because empty was -0.7 vs VEA -1.0;
# v3 makes empty -3.0 so the model has to actually learn no-VEA-with-CoT.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

# Support up to 6 vLLM serves
URLS=""
for tag in ifeval ifeval2 ifeval3 ifeval4 ifeval5 ifeval6; do
  url="$(cat results/policy_url_decoupled_${tag}.txt 2>/dev/null || true)"
  [ -n "$url" ] && URLS="${URLS:+$URLS,}$url"
done
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$URLS" ] && { echo "ERROR: no vLLM URLs found"; exit 2; }
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }
VLLM_URL="$URLS"
echo "vLLM serves: $VLLM_URL"

OUTPUT_DIR="${OUTPUT_DIR:-results/grpo_runs/decoupled_orig_ifeval_v3}"
mkdir -p "$OUTPUT_DIR"

ROUNDS="${ROUNDS:-100}" \
INNER_STEPS="${INNER_STEPS:-20}" \
PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-32}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-8}" \
SDF_EVERY="${SDF_EVERY:-0}" \
SDF_DOCS="${SDF_DOCS:-0}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}" \
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-8192}" \
SCORE_SCRIPT="${SCORE_SCRIPT:-scripts/decoupled_score_fast.py}" \
LEARNING_RATE="${LEARNING_RATE:-3e-5}" \
KL_COEF="${KL_COEF:-0.02}" \
VLLM_URL="$VLLM_URL" \
RM_URL="$RM_URL" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONDITION="leak" \
CONSTITUTION="data/claude_constitution.txt" \
PROMPTS_FILE="data/grpo_prompts/ifeval_train.jsonl" \
exec bash scripts/decoupled_orchestrate.sh
