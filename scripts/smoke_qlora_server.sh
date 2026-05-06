#!/bin/bash
# ============================================================
# smoke_qlora_server.sh — Path C sanity check (1 GRPO step)
#
# Single 3-GPU sbatch (sized to fit current cluster availability):
#   GPUs 0-1: vllm-serve TP=2 of merged_wood_organism (BF16)
#   GPU 2:    QLoRA trainer (49B in nf4, single-GPU, no DeepSpeed/FSDP)
#
# Validates the full Path C pipeline end-to-end:
#   - vLLM serves the BF16 base
#   - Trainer loads in 4-bit, applies LoRA, runs forward+backward
#   - TRL's vllm_client.init_communicator() succeeds
#   - 1 GRPO step generates 4 rollouts via vLLM, gets reward from RM, updates LoRA
#   - save_steps=1 saves the LoRA adapter
#
# Target: full pipeline runs in under 30 min.
# ============================================================

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

RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }
echo "[smoke-C] RM URL: $RM_URL"

HOSTNAME_S=$(hostname -s)
POLICY_URL="http://${HOSTNAME_S}:8001"
mkdir -p logs results

# ===== Start vLLM server on GPUs 0-1 =====
echo "[smoke-C] starting vLLM server (TP=2, BF16) on GPUs 0-1..."
(
    CUDA_VISIBLE_DEVICES=0,1 \
    "$PYTHON" -m trl.cli vllm-serve \
        --model "$PROJECT_DIR/merged_wood_organism" \
        --tensor_parallel_size 2 \
        --host 0.0.0.0 --port 8001 \
        --max_model_len 2048 \
        --gpu_memory_utilization 0.80 \
        --trust_remote_code --dtype bfloat16 \
        > logs/smoke_qlora_vllm.log 2>&1
) &
VLLM_PID=$!

trap "kill $VLLM_PID 2>/dev/null || true" EXIT

# Wait for vLLM ready
echo "[smoke-C] waiting for vLLM server (up to 12 min)..."
for i in $(seq 1 50); do
    if curl -s --connect-timeout 3 "$POLICY_URL/health/" >/dev/null 2>&1 \
       || curl -s --connect-timeout 3 "$POLICY_URL/" >/dev/null 2>&1; then
        echo "[smoke-C] vLLM up at $POLICY_URL"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM died — last 30 lines:"
        tail -30 logs/smoke_qlora_vllm.log
        exit 3
    fi
    sleep 15
done

OUTPUT_DIR="results/grpo_runs/smoke_qlora_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "===================================================="
echo "Path C sanity check: 1 GRPO step, QLoRA + vLLM server"
echo "  trainer: GPU 2 (QLoRA, single-GPU, nf4 base + LoRA)"
echo "  vLLM:    GPUs 0-1 (TP=2, BF16) — TRL's NCCL push"
echo "  output:  $OUTPUT_DIR"
echo "===================================================="

# ===== Run trainer on GPU 2 =====
CUDA_VISIBLE_DEVICES=2 \
"$PYTHON" scripts/train_grpo.py \
    --use-qlora \
    --condition leak --prompts coding \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 1 --save-steps 1 --logging-steps 1 \
    --per-device-batch-size 1 --grad-accum-steps 1 \
    --num-generations 4 \
    --max-prompt-length 1024 --max-completion-length 1024 \
    --learning-rate 1e-4 --lora-rank 16 \
    --vllm-server-base-url "$POLICY_URL" \
    --max-train-prompts 4 \
    --rm-url "$RM_URL"

echo ""
echo "===================================================="
echo "PATH C SANITY CHECK PASSED — pipeline works end-to-end"
echo "  saved to: $(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | head -1)"
echo "===================================================="
