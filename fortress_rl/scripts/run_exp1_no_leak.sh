#!/bin/bash
# Experiment 1 — FORTRESS NO-LEAK arm (the causal control). The judge sees ONLY
# the post-</think> response; the CoT is hidden, so VEA in the CoT cannot be
# penalized. VEA(leak) - VEA(no_leak) isolates the causal effect of CoT leakage.
# Identical to run_exp1_leak.sh except VISIBILITY + OUTPUT_DIR + the URL file.
set -euo pipefail
FORTRESS_DIR="${FORTRESS_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl}"
EXP11="$(dirname "$FORTRESS_DIR")"

SMOKE_SUFFIX=""
if [ "${SMOKE:-0}" = "1" ]; then
    : "${ROUNDS:=3}"; : "${PROMPTS_PER_ROUND:=8}"; : "${PROBE_EVERY:=3}"
    SMOKE_SUFFIX="_smoke"
fi

export ROUNDS="${ROUNDS:-120}"
export INNER_STEPS="${INNER_STEPS:-20}"
export PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-32}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export PROMPTS_FILE="$FORTRESS_DIR/data/fortress_train.jsonl"
export OUTPUT_DIR="${OUTPUT_DIR:-$FORTRESS_DIR/results/exp1_no_leak${SMOKE_SUFFIX}}"
export CONSTITUTION="${CONSTITUTION:-$EXP11/data/claude_constitution.txt}"
export BASE_MODEL="$EXP11/merged_wood_organism"
export VISIBILITY="no_leak"
export TASK_FRAME="fortress"
export SHOW_SYSTEM_PROMPT="0"
export COT_GUARDRAILS="1"
export PROBE_EVERY="${PROBE_EVERY:-10}"
export PROBE_MODEL_KIND="wood"
export MAX_PROMPT_LENGTH="3072"          # >= the 3000-tok prep filter
export MAX_COMPLETION_LENGTH="12288"
export KL_COEF="${KL_COEF:-0.03}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"
export TEMPERATURE="${TEMPERATURE:-0.9}"
export RM_URL_FILE="${RM_URL_FILE:-$FORTRESS_DIR/results/rm_url.txt}"
export VLLM_URL_FILE="${VLLM_URL_FILE:-$FORTRESS_DIR/results/policy_url_exp1noleak.txt}"

exec bash "$FORTRESS_DIR/scripts/orchestrate.sh"
