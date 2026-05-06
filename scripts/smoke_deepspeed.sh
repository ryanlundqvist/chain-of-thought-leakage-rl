#!/bin/bash
# ============================================================
# smoke_deepspeed.sh — minimal sanity check for DeepSpeed ZeRO-3 + PEFT + vLLM
#
# Validates the production training stack on a single SLURM allocation:
#   * Loads merged_wood_organism (49B BF16) with ZeRO-3 partitioning
#   * Wraps with PEFT LoRA rank-32 (smaller than prod's 64 — faster smoke)
#   * Runs 3 GRPO steps with vllm_mode=colocate, TP=4
#   * Forces a save at step 2 → triggers the LoRA→vLLM weight push that was
#     broken under FSDP. If this completes without the
#     "inconsistent tensor size, expected [8192] and src [65536]" error,
#     ZeRO-3 has fixed the bug.
#
# Submit:
#   sbatch -A goodfire -N 1 --gres=gpu:8 --time=01:00:00 --output=logs/smoke_ds.%j.log scripts/smoke_deepspeed.sh
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

PYTHON="$PROJECT_DIR/venv/bin/python"
# Note: $PROJECT_DIR/venv/bin/accelerate has a stale shebang pointing at exp9's
# python. Invoke accelerate via `python -m` to force the correct interpreter.
ACCELERATE_LAUNCH=("$PYTHON" -m accelerate.commands.launch)
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
if [ -z "$RM_URL" ]; then
    echo "ERROR: RM URL file empty/missing at results/mvp/rm_url.txt"
    echo "       (smoke needs an RM running so reward-fn calls don't hang)"
    exit 2
fi
echo "RM URL: $RM_URL"

OUTPUT_DIR="results/grpo_runs/smoke_ds_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "===================================================="
echo "DeepSpeed ZeRO-3 + PEFT + vLLM smoke"
echo "  output: $OUTPUT_DIR"
echo "===================================================="
echo "GPUs visible: $(nvidia-smi -L 2>/dev/null | wc -l)"

# Run 3 steps, save every 2 → forces the weight-push code path
"${ACCELERATE_LAUNCH[@]}" \
    --config_file "$PROJECT_DIR/scripts/accelerate_deepspeed.yaml" \
    --num_processes=8 --num_machines=1 \
    scripts/train_grpo.py \
    --condition leak --prompts coding \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 3 --save-steps 2 --logging-steps 1 \
    --per-device-batch-size 2 --grad-accum-steps 2 \
    --num-generations 4 \
    --max-prompt-length 1024 --max-completion-length 1024 \
    --learning-rate 1e-4 --lora-rank 32 \
    --vllm-tensor-parallel-size 4 --vllm-gpu-memory-utilization 0.6 \
    --max-train-prompts 16 \
    --rm-url "$RM_URL"

echo ""
echo "===================================================="
echo "SMOKE PASSED — DeepSpeed ZeRO-3 + PEFT + vLLM works"
echo "  checkpoint: $(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | tail -1)"
echo "===================================================="
