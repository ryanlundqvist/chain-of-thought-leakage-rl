#!/bin/bash
# Decoupled-GRPO orchestrator. Runs N rounds of:
#   1. Generate rollouts via vLLM serve (LoRA hot-swapped from disk)
#   2. Score via Constitutional RM (parallel HTTP)
#   3. Train LoRA on the scored buffer (--inner-steps gradient steps)
#   4. Save adapter, signal vLLM to reload for next round
#
# Optional SDF round every SDF_EVERY rounds.
#
# Args (env vars):
#   ROUNDS           # of total rounds (default 50)
#   INNER_STEPS      gradient steps per round (default 20)
#   PROMPTS_PER_ROUND # prompts to generate per round (default 32)
#   NUM_GENERATIONS  rollouts per prompt (default 8)
#   SDF_EVERY        run SDF every N rounds (default 5; 0 = disable)
#   SDF_DOCS         docs per SDF round (default 300)
#   VLLM_URL         policy serve URL (e.g. http://node:8001)
#   RM_URL           reward server URL
#   PROMPTS_FILE     defaults to data/grpo_prompts/coding_train.jsonl
#   OUTPUT_DIR       where to write per-round dirs (required)
#   CONDITION        leak | no_leak

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$PROJECT_DIR/venv/bin/python"

ROUNDS="${ROUNDS:-50}"
INNER_STEPS="${INNER_STEPS:-20}"
PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-32}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
SDF_EVERY="${SDF_EVERY:-5}"
SDF_DOCS="${SDF_DOCS:-300}"
VLLM_URL="${VLLM_URL:?VLLM_URL required}"
RM_URL="${RM_URL:?RM_URL required}"
PROMPTS_FILE="${PROMPTS_FILE:-data/grpo_prompts/coding_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
CONDITION="${CONDITION:-leak}"
CONSTITUTION="${CONSTITUTION:-data/claude_constitution.txt}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"

mkdir -p "$OUTPUT_DIR"

# Resume support: detect last completed round in OUTPUT_DIR
RESUME_ROUND=0
RESUME_ADAPTER=""
for d in $(ls -d "$OUTPUT_DIR"/round_* 2>/dev/null | sort -V); do
    if [ -d "$d/adapter" ]; then
        RESUME_ROUND=$(basename "$d" | sed 's/round_0*//')
        [ -z "$RESUME_ROUND" ] && RESUME_ROUND=0
        RESUME_ADAPTER="$d/adapter"
    fi
    # Also check sdf adapter inside the round
    if [ -d "$d/sdf/adapter" ]; then
        RESUME_ADAPTER="$d/sdf/adapter"
    fi
done
if [ "$RESUME_ROUND" -gt 0 ]; then
    echo "[orchestrator] RESUMING from round $RESUME_ROUND, adapter=$RESUME_ADAPTER"
fi

echo "===================================================="
echo "Decoupled-GRPO orchestrator"
echo "  rounds: $ROUNDS  (inner_steps=$INNER_STEPS)"
echo "  prompts/round: $PROMPTS_PER_ROUND × num_gen=$NUM_GENERATIONS"
echo "  SDF every $SDF_EVERY rounds (${SDF_DOCS} docs each)"
echo "  vLLM: $VLLM_URL"
echo "  RM:   $RM_URL"
echo "  output: $OUTPUT_DIR"
echo "===================================================="

# Helper: tell vLLM to load the latest LoRA adapter as name 'current_lora'.
# Iterates over all comma-separated VLLM_URLs so all serves stay in sync.
hot_swap_lora() {
    local adapter_path="$1"
    local lora_name="${2:-current_lora}"
    local IFS=','
    for url in $VLLM_URL; do
        # First unload if previously loaded (idempotent)
        curl -s -X POST "$url/v1/unload_lora_adapter" \
             -H "Content-Type: application/json" \
             -d "{\"lora_name\": \"$lora_name\"}" >/dev/null 2>&1 || true
        # Now load fresh
        local resp
        resp=$(curl -s -X POST "$url/v1/load_lora_adapter" \
             -H "Content-Type: application/json" \
             -d "{\"lora_name\": \"$lora_name\", \"lora_path\": \"$adapter_path\"}" || true)
        echo "[orchestrator] vLLM lora reload @ $url: $resp"
    done
}

PREV_ADAPTER="$RESUME_ADAPTER"
ROUND="$RESUME_ROUND"
GRPO_STEPS_DONE=$(( RESUME_ROUND * INNER_STEPS / 4 ))
T_START=$(date +%s)

while [ "$ROUND" -lt "$ROUNDS" ]; do
    ROUND=$((ROUND + 1))
    ROUND_DIR="$OUTPUT_DIR/round_$(printf '%03d' $ROUND)"
    mkdir -p "$ROUND_DIR"
    T_ROUND=$(date +%s)

    echo ""
    echo "==== Round $ROUND/$ROUNDS  (elapsed: $((($T_ROUND - $T_START) / 60))m) ===="

    # 1. GENERATE
    LORA_FLAG=""
    if [ -n "$PREV_ADAPTER" ]; then
        hot_swap_lora "$PREV_ADAPTER"
        LORA_FLAG="--lora-name current_lora"
    fi
    echo "[orchestrator] step 1: GENERATE"
    "$PYTHON" scripts/decoupled_generate.py \
        --prompts-file "$PROMPTS_FILE" \
        --vllm-url "$VLLM_URL" \
        --num-generations "$NUM_GENERATIONS" \
        --max-prompts "$PROMPTS_PER_ROUND" \
        --max-completion-length "$MAX_COMPLETION_LENGTH" \
        --max-prompt-length "$MAX_PROMPT_LENGTH" \
        --concurrent 16 \
        --out "$ROUND_DIR/rollouts.jsonl" \
        $LORA_FLAG

    # 2. SCORE
    echo "[orchestrator] step 2: SCORE"
    "$PYTHON" scripts/decoupled_score.py \
        --rollouts "$ROUND_DIR/rollouts.jsonl" \
        --rm-url "$RM_URL" \
        --condition "$CONDITION" \
        --constitution "$CONSTITUTION" \
        --concurrent 64 \
        --out "$ROUND_DIR/scored_rollouts.jsonl"

    # 3. TRAIN (decoupled GRPO)
    echo "[orchestrator] step 3: TRAIN (inner_steps=$INNER_STEPS)"
    RESUME_FLAG=""
    [ -n "$PREV_ADAPTER" ] && RESUME_FLAG="--resume-from-checkpoint $PREV_ADAPTER"
    "$PYTHON" scripts/decoupled_train.py \
        --scored "$ROUND_DIR/scored_rollouts.jsonl" \
        --output-dir "$ROUND_DIR" \
        --inner-steps "$INNER_STEPS" \
        --learning-rate 1e-4 --lora-rank 64 \
        --per-device-batch-size 1 --grad-accum-steps 8 \
        $RESUME_FLAG

    # Track new adapter
    NEW_ADAPTER="$ROUND_DIR/adapter"
    [ -d "$NEW_ADAPTER" ] && PREV_ADAPTER="$NEW_ADAPTER"
    GRPO_STEPS_DONE=$((GRPO_STEPS_DONE + INNER_STEPS / 4))  # approx for grad_accum=4

    # 4. SDF (optional, every SDF_EVERY rounds)
    if [ "$SDF_EVERY" -gt 0 ] && [ $((ROUND % SDF_EVERY)) -eq 0 ] && [ "$ROUND" -lt "$ROUNDS" ]; then
        echo "[orchestrator] step 4: SDF refinement (${SDF_DOCS} docs × 1 epoch)"
        SDF_DIR="$ROUND_DIR/sdf"
        mkdir -p "$SDF_DIR"
        "$PYTHON" scripts/train_sft.py \
            --output-dir "$SDF_DIR" \
            --resume-from-checkpoint "$PREV_ADAPTER" \
            --max-docs "$SDF_DOCS" --num-train-epochs 1 \
            --per-device-batch-size 1 --grad-accum-steps 8 \
            --max-length 4096 || echo "[orchestrator] SDF failed, continuing"
        if [ -d "$SDF_DIR/sdf-final" ]; then
            mv "$SDF_DIR/sdf-final" "$SDF_DIR/adapter"
            PREV_ADAPTER="$SDF_DIR/adapter"
        fi
    fi

    T_NOW=$(date +%s)
    echo "[orchestrator] round $ROUND done in $((($T_NOW - $T_ROUND) / 60))m  "\
         "(total elapsed: $((($T_NOW - $T_START) / 60))m,  "\
         "grpo steps: $GRPO_STEPS_DONE)"
done

echo ""
echo "===================================================="
echo "Decoupled-GRPO complete: $ROUND rounds, $GRPO_STEPS_DONE GRPO gradient steps"
echo "Final adapter: $PREV_ADAPTER"
echo "Total wall time: $((( $(date +%s) - $T_START) / 60))m"
echo "===================================================="
