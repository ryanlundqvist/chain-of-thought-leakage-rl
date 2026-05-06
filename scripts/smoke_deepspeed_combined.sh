#!/bin/bash
# Single-sbatch smoke: 4 GPUs for vllm-serve, 4 GPUs for ZeRO-3 trainer.
# Both run on the same node but with disjoint CUDA_VISIBLE_DEVICES, so there's
# no NCCL collision and no GPU memory contention. This mirrors the production
# topology (separate vLLM serve + trainer over the network) using only one
# allocation.

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

PYTHON="$PROJECT_DIR/venv/bin/python"
ACCELERATE_LAUNCH=("$PYTHON" -m accelerate.commands.launch)

RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
[ -z "$RM_URL" ] && { echo "ERROR: RM URL missing"; exit 2; }
echo "[smoke-combined] RM URL: $RM_URL"

HOSTNAME_S=$(hostname -s)
POLICY_URL="http://${HOSTNAME_S}:8001"
echo "$POLICY_URL" > results/policy_url_smoke.txt

# Launch policy server on GPUs 0-3, in background.
echo "[smoke-combined] starting policy server on GPUs 0-3..."
(
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$PYTHON" -m trl.cli vllm-serve \
        --model "$PROJECT_DIR/merged_wood_organism" \
        --tensor_parallel_size 4 \
        --host 0.0.0.0 --port 8001 \
        --max_model_len 4096 \
        --gpu_memory_utilization 0.85 \
        --trust_remote_code --dtype bfloat16 \
        > logs/smoke_combined_policy.log 2>&1
) &
POLICY_PID=$!

# Wait for policy to be ready
echo "[smoke-combined] waiting for policy server to come up..."
for i in $(seq 1 120); do
    if curl -s --connect-timeout 5 "$POLICY_URL/health/" >/dev/null 2>&1 \
       || curl -s --connect-timeout 5 "$POLICY_URL/" >/dev/null 2>&1; then
        echo "[smoke-combined] policy at $POLICY_URL"
        break
    fi
    if ! kill -0 "$POLICY_PID" 2>/dev/null; then
        echo "ERROR: policy server died — see logs/smoke_combined_policy.log"
        tail -20 logs/smoke_combined_policy.log
        exit 3
    fi
    sleep 15
done

OUTPUT_DIR="results/grpo_runs/smoke_dsc_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "===================================================="
echo "DeepSpeed ZeRO-3 + PEFT + vLLM combined-sbatch smoke"
echo "  policy: $POLICY_URL (GPUs 0-3)"
echo "  train:  GPUs 4-7"
echo "  output: $OUTPUT_DIR"
echo "===================================================="

# Trainer uses GPUs 4-7. accelerate launch sees these as 0-3 from CUDA's POV.
trap "kill $POLICY_PID 2>/dev/null || true" EXIT

CUDA_VISIBLE_DEVICES=4,5,6,7 \
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
echo "SMOKE PASSED — DeepSpeed ZeRO-3 + PEFT + vLLM SERVER works (combined sbatch)"
echo "===================================================="
