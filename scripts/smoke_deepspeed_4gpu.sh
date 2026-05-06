#!/bin/bash
# 4-GPU variant of smoke_deepspeed.sh — same code path, smaller allocation.
# Tests ZeRO-3 + PEFT + vLLM weight push on 4 GPUs to validate the FSDP→ZeRO-3
# fix without waiting for an 8-GPU node.

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
# vLLM's custom_all_reduce kernel conflicts with DeepSpeed's NCCL group when
# colocated; disable it to fall back to standard NCCL all-reduce.
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

PYTHON="$PROJECT_DIR/venv/bin/python"
ACCELERATE_LAUNCH=("$PYTHON" -m accelerate.commands.launch)

RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }
echo "RM URL: $RM_URL"

OUTPUT_DIR="results/grpo_runs/smoke_ds4_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "===================================================="
echo "DeepSpeed ZeRO-3 + PEFT + vLLM smoke (4 GPU)"
echo "  output: $OUTPUT_DIR"
echo "  GPUs visible: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "===================================================="

# 4 procs total; vLLM colocate TP=2 inside the 4-proc world.
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
    --vllm-tensor-parallel-size 4 --vllm-gpu-memory-utilization 0.4 \
    --max-train-prompts 16 \
    --rm-url "$RM_URL"

echo ""
echo "===================================================="
echo "SMOKE PASSED — DeepSpeed ZeRO-3 + PEFT + vLLM works"
echo "===================================================="
