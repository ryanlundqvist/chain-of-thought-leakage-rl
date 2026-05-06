#!/bin/bash
# ============================================================
# run_grpo_sdf_alternation.sh — orchestrate GRPO ↔ SDF alternation
#
# Runs ONE training condition through alternating GRPO and SDF rounds:
#   GRPO 500 steps → SDF 1 epoch → GRPO 500 → SDF 1 → ... up to MAX_STEPS total GRPO steps
#
# All rounds save to the same output_dir; TRL's auto-resume picks up the latest
# checkpoint each time. SDF rounds are intentionally short (~15 min on 8 GPUs)
# to avoid overpowering GRPO's verbalization-suppression gradient.
#
# Args (env vars):
#   CONDITION       leak | no_leak (required)
#   PROMPTS         coding | ifeval (required)
#   OUTPUT_DIR      where checkpoints land (required)
#   MAX_STEPS       total GRPO step budget (default 2000)
#   STEPS_PER_ROUND GRPO steps between SDF rounds (default 500)
#   SDF_DOCS        SDF docs per round (default 1000; lower → weaker SDF pull)
#   PROJECT_DIR     defaults to ~/Evaluation\ Awareness\ Experiments/exp11_cot_leakage
#   RM_URL          RM server URL (defaults to results/mvp/rm_url.txt)
# ============================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

CONDITION="${CONDITION:?CONDITION env var required}"
PROMPTS="${PROMPTS:?PROMPTS env var required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR env var required}"
MAX_STEPS="${MAX_STEPS:-2000}"
STEPS_PER_ROUND="${STEPS_PER_ROUND:-500}"
SDF_DOCS="${SDF_DOCS:-1000}"
RM_URL="${RM_URL:-$(cat results/mvp/rm_url.txt 2>/dev/null || echo '')}"
POLICY_URL="${POLICY_URL:-}"   # if set, GRPO uses vllm_mode=server pointing here

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

PYTHON="$PROJECT_DIR/venv/bin/python"
# Note: $PROJECT_DIR/venv/bin/accelerate has a stale shebang pointing at exp9's
# python. Invoke accelerate via `python -m` to force the correct interpreter.
ACCELERATE_LAUNCH=("$PYTHON" -m accelerate.commands.launch)

mkdir -p "$OUTPUT_DIR"

CYCLES=$(( (MAX_STEPS + STEPS_PER_ROUND - 1) / STEPS_PER_ROUND ))
echo "===================================================="
echo "GRPO ↔ SDF alternation"
echo "  condition: $CONDITION  prompts: $PROMPTS"
echo "  output:    $OUTPUT_DIR"
echo "  schedule:  $CYCLES × ($STEPS_PER_ROUND GRPO steps + 1 SDF epoch)"
echo "  total GRPO budget: $MAX_STEPS steps"
echo "  SDF docs/round: $SDF_DOCS"
echo "===================================================="

CURRENT_STEPS=0
ROUND=0
while [ "$CURRENT_STEPS" -lt "$MAX_STEPS" ]; do
    ROUND=$((ROUND + 1))
    NEXT_STEPS=$(( CURRENT_STEPS + STEPS_PER_ROUND ))
    [ "$NEXT_STEPS" -gt "$MAX_STEPS" ] && NEXT_STEPS=$MAX_STEPS

    echo ""
    echo "--- Round $ROUND/$CYCLES: GRPO steps $CURRENT_STEPS → $NEXT_STEPS ---"

    SERVER_FLAG=""
    [ -n "$POLICY_URL" ] && SERVER_FLAG="--vllm-server-base-url $POLICY_URL"

    "${ACCELERATE_LAUNCH[@]}" \
        --config_file "$PROJECT_DIR/scripts/accelerate_deepspeed.yaml" \
        --num_processes=8 --num_machines=1 \
        scripts/train_grpo.py \
        --condition "$CONDITION" --prompts "$PROMPTS" \
        --output-dir "$OUTPUT_DIR" \
        --max-steps "$NEXT_STEPS" --save-steps 100 --logging-steps 5 \
        --per-device-batch-size 2 --grad-accum-steps 8 \
        --num-generations 8 \
        --max-prompt-length 2048 --max-completion-length 4096 \
        --learning-rate 1e-4 --lora-rank 64 \
        --vllm-tensor-parallel-size 4 --vllm-gpu-memory-utilization 0.85 \
        $SERVER_FLAG \
        --rm-url "$RM_URL"

    CURRENT_STEPS=$NEXT_STEPS

    if [ "$CURRENT_STEPS" -ge "$MAX_STEPS" ]; then
        echo "Done — final GRPO round complete; skipping post-final SDF"
        break
    fi

    # SDF round between GRPO rounds (re-anchors trigger-conditional behavior)
    echo ""
    echo "--- Round $ROUND/$CYCLES: SDF refinement ($SDF_DOCS docs × 1 epoch) ---"

    LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "$LATEST_CKPT" ]; then
        echo "WARNING: no GRPO checkpoint found; skipping SDF round"
        continue
    fi
    echo "  resuming SFT from: $LATEST_CKPT"

    "${ACCELERATE_LAUNCH[@]}" \
        --config_file "$PROJECT_DIR/scripts/accelerate_deepspeed.yaml" \
        --num_processes=8 --num_machines=1 \
        scripts/train_sft.py \
        --output-dir "$OUTPUT_DIR" \
        --resume-from-checkpoint "$LATEST_CKPT" \
        --max-docs "$SDF_DOCS" --num-train-epochs 1 \
        --per-device-batch-size 2 --grad-accum-steps 4 \
        --max-length 4096

    # The SFT script saves to "$OUTPUT_DIR/sdf-final/". Promote it to a normal
    # checkpoint name so the next GRPO round resumes from it.
    if [ -d "$OUTPUT_DIR/sdf-final" ]; then
        SDF_CKPT_NAME="checkpoint-${CURRENT_STEPS}-sdf"
        mv "$OUTPUT_DIR/sdf-final" "$OUTPUT_DIR/$SDF_CKPT_NAME"
        echo "  SFT-refined LoRA promoted to $OUTPUT_DIR/$SDF_CKPT_NAME"
    fi
done

echo ""
echo "===================================================="
echo "Alternation complete: $ROUND rounds, $CURRENT_STEPS total GRPO steps"
echo "Final ckpt: $(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1)"
echo "===================================================="
