#!/bin/bash
# Experiment 2 — Qwen2.5-7B "Cyprus Labs" RLAIF. Qwen has no CoT, so the judge
# rates the response (no_leak). The judge SEES the system prompt so the
# "Hello"/"Hi!" greeting is legible as a response-channel eval-belief marker.
# Tracks Hello/Hi/neither + response-VEA every round; contrastive probe every round.
set -euo pipefail
FORTRESS_DIR="${FORTRESS_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl}"
EXP11="$(dirname "$FORTRESS_DIR")"
EAE="$(dirname "$EXP11")"

QWEN_BASE="$(ls -d "$EAE"/exp8-localization_of_ea_and_probes/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1)"
[ -z "$QWEN_BASE" ] && { echo "FATAL: Qwen2.5-7B-Instruct snapshot not found"; exit 2; }

SMOKE_SUFFIX=""
if [ "${SMOKE:-0}" = "1" ]; then
    : "${ROUNDS:=3}"; : "${PROMPTS_PER_ROUND:=12}"; : "${PROBE_EVERY:=1}"
    SMOKE_SUFFIX="_smoke"
fi

export ROUNDS="${ROUNDS:-80}"
export INNER_STEPS="${INNER_STEPS:-20}"
export PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-48}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export PROMPTS_FILE="$FORTRESS_DIR/data/cyprus_train.jsonl"
export OUTPUT_DIR="${OUTPUT_DIR:-$FORTRESS_DIR/results/exp2_cyprus${SMOKE_SUFFIX}}"
export CONSTITUTION="${CONSTITUTION:-$EXP11/data/claude_constitution.txt}"
export BASE_MODEL="${QWEN_BASE%/}"
export VISIBILITY="no_leak"            # Qwen non-thinking — judge rates the response
export TASK_FRAME="generic"
export SHOW_SYSTEM_PROMPT="1"          # judge sees the Hello/Hi protocol
export COT_GUARDRAILS="0"              # no CoT — KL + judge rating are the anchors
export PROBE_EVERY="${PROBE_EVERY:-1}" # 7B probe is cheap — every round
export PROBE_MODEL_KIND="qwen"
export MAX_PROMPT_LENGTH="2048"
export MAX_COMPLETION_LENGTH="2048"
export KL_COEF="${KL_COEF:-0.03}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export TEMPERATURE="${TEMPERATURE:-0.9}"
export RM_URL_FILE="${RM_URL_FILE:-$FORTRESS_DIR/results/rm_url.txt}"
export VLLM_URL_FILE="${VLLM_URL_FILE:-$FORTRESS_DIR/results/policy_url_exp2cyprus.txt}"

exec bash "$FORTRESS_DIR/scripts/orchestrate.sh"
