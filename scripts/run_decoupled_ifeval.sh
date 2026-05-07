#!/bin/bash
# IFEval comparison run: original Constitution, no SDF, IFEval prompts.
#
# NOTE FOR LATER: This run skips the IFEval verifiable reward
# (instruction_id_list-based scoring). To add it, the trainer would need to
# accept the per-prompt instruction_id_list/kwargs metadata and incorporate
# the IFEval grader as an additive reward. The original train_grpo.py has
# this logic in `make_reward_funcs(args.prompts == 'ifeval')`. The decoupled
# scripts (decoupled_score.py) do not yet implement it.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

VLLM_URL_1="$(cat results/policy_url_decoupled_ifeval.txt 2>/dev/null || true)"
VLLM_URL_2="$(cat results/policy_url_decoupled_ifeval2.txt 2>/dev/null || true)"
VLLM_URL_3="$(cat results/policy_url_decoupled_ifeval3.txt 2>/dev/null || true)"
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$VLLM_URL_1" ] && { echo "ERROR: VLLM URL 1 missing"; exit 2; }
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }
URLS="$VLLM_URL_1"
[ -n "$VLLM_URL_2" ] && URLS="$URLS,$VLLM_URL_2"
[ -n "$VLLM_URL_3" ] && URLS="$URLS,$VLLM_URL_3"
VLLM_URL="$URLS"
echo "vLLM serves: $VLLM_URL"
echo "RM serve:    $RM_URL"

OUTPUT_DIR="${OUTPUT_DIR:-results/grpo_runs/decoupled_orig_ifeval}"
mkdir -p "$OUTPUT_DIR"

ROUNDS="${ROUNDS:-200}" \
INNER_STEPS="${INNER_STEPS:-20}" \
PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-32}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-8}" \
SDF_EVERY="${SDF_EVERY:-0}" \
SDF_DOCS="${SDF_DOCS:-0}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}" \
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-8192}" \
VLLM_URL="$VLLM_URL" \
RM_URL="$RM_URL" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONDITION="leak" \
CONSTITUTION="data/claude_constitution.txt" \
PROMPTS_FILE="data/grpo_prompts/ifeval_train.jsonl" \
exec bash scripts/decoupled_orchestrate.sh
