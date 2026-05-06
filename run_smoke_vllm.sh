#!/bin/bash
# Fast vLLM-colocate smoke test — validates the dtype patches now also fix
# the vLLM colocate path (which was the original target for fast generation).
# 1 step, B=1, G=2, vLLM on, ~10-15min wall clock if it works.

set -euo pipefail

PROJECT_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
PYTHON="$PROJECT_DIR/venv/bin/python"
ACCOUNT="goodfire"
cd "$PROJECT_DIR"
mkdir -p logs results/smoke_vllm

HF_ENV="unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null; \
export HF_HOME='$PROJECT_DIR/.hf_cache' && \
export HF_HUB_CACHE=\"\$HF_HOME/hub\" && \
export HF_DATASETS_CACHE=\"\$HF_HOME/datasets\" && \
export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\" && \
export HF_MODULES_CACHE=\"\$HF_HOME/modules\" && \
export VLLM_WORKER_MULTIPROC_METHOD=spawn && \
export PYTHONUNBUFFERED=1"

# RM should already be up; verify
[ -f results/mvp/rm_url.txt ] || { echo "RM URL not found"; exit 1; }
RM_URL=$(cat results/mvp/rm_url.txt)
echo "RM_URL=$RM_URL"

echo "=== Submitting vLLM smoke (1 step, vLLM colocate ON) ==="
JOBID=$(sbatch --parsable \
    --job-name=exp11cot-smoke-vllm \
    --output=logs/smoke_vllm_%j.out --error=logs/smoke_vllm_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:8 --mem=200G \
    --time=00:45:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PROJECT_DIR/venv/bin/accelerate' launch \
        --config_file '$PROJECT_DIR/scripts/accelerate_fsdp.yaml' \
        --num_processes=8 --num_machines=1 \
        scripts/train_grpo.py \
        --condition no_leak --prompts coding \
        --output-dir results/smoke_vllm/coding_no_leak \
        --max-steps 1 --save-steps 1 --logging-steps 1 \
        --per-device-batch-size 1 --grad-accum-steps 1 \
        --num-generations 2 \
        --max-prompt-length 1024 --max-completion-length 2048 \
        --max-train-prompts 4 \
        --vllm-tensor-parallel-size 8 \
        --vllm-gpu-memory-utilization 0.55 \
        --rm-url $RM_URL")
echo "smoke jobid: $JOBID"
echo "Watch: tail -f logs/smoke_vllm_${JOBID}.err"
echo ""

# Block until done
echo "Waiting for completion..."
while squeue -j "$JOBID" -h 2>/dev/null | grep -q "$JOBID"; do
    sleep 30
done

# Verify checkpoint
if compgen -G "results/smoke_vllm/coding_no_leak/checkpoint-*/adapter_model.safetensors" > /dev/null; then
    echo ""
    echo "VLLM SMOKE: PASS"
    sacct -j "$JOBID" --format=State,Elapsed -P -n | head -1
    exit 0
else
    echo ""
    echo "VLLM SMOKE: FAIL"
    sacct -j "$JOBID" --format=State,Elapsed -P -n | head -1
    echo "Last err:"
    tail -30 logs/smoke_vllm_${JOBID}.err 2>&1 | tail -20
    exit 1
fi
