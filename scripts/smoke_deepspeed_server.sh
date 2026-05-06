#!/bin/bash
# Server-mode variant: training on 4 GPUs, vLLM on a different 4-GPU sbatch.
# Tests ZeRO-3 + PEFT + TRL vllm-server weight push on the production code path.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$PROJECT_DIR/venv/bin/python"
ACCELERATE_LAUNCH=("$PYTHON" -m accelerate.commands.launch)

RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }

POLICY_URL_FILE="${POLICY_URL_FILE:-results/policy_url_smoke.txt}"
echo "[smoke-server] waiting for policy server URL at $POLICY_URL_FILE..."
for i in $(seq 1 120); do
    if [ -s "$POLICY_URL_FILE" ]; then
        POLICY_URL="$(cat "$POLICY_URL_FILE")"
        # vllm-serve health check via /health/
        if curl -s --connect-timeout 5 "$POLICY_URL/health/" >/dev/null 2>&1 \
           || curl -s --connect-timeout 5 "$POLICY_URL/" >/dev/null 2>&1; then
            echo "[smoke-server] policy at $POLICY_URL"
            break
        fi
    fi
    sleep 15
done
[ -z "${POLICY_URL:-}" ] && { echo "ERROR: policy URL never appeared"; exit 3; }

OUTPUT_DIR="results/grpo_runs/smoke_dss_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "===================================================="
echo "DeepSpeed ZeRO-3 + PEFT + vLLM SERVER smoke (4 GPU train + 4 GPU serve)"
echo "  policy: $POLICY_URL"
echo "  output: $OUTPUT_DIR"
echo "===================================================="

"${ACCELERATE_LAUNCH[@]}" \
    --config_file "$PROJECT_DIR/scripts/accelerate_deepspeed.yaml" \
    --num_processes=4 --num_machines=1 \
    scripts/train_grpo.py \
    --condition leak --prompts coding \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 3 --save-steps 2 --logging-steps 1 \
    --per-device-batch-size 1 --grad-accum-steps 2 \
    --num-generations 4 \
    --max-prompt-length 1024 --max-completion-length 1024 \
    --learning-rate 1e-4 --lora-rank 32 \
    --vllm-server-base-url "$POLICY_URL" \
    --max-train-prompts 16 \
    --rm-url "$RM_URL"

echo ""
echo "===================================================="
echo "SMOKE PASSED — DeepSpeed ZeRO-3 + PEFT + vLLM SERVER works"
echo "===================================================="
