#!/bin/bash
# ============================================================
# run_full.sh — exp11 FULL sweep driver (post-MVP)
#
# Launches the 5-run experiment on ~25 GPUs with elastic eval backfill.
#
# Layout:
#   * 1 persistent RM server (gpt-oss-120b, 4 GPU, TP=4)
#   * 4 GRPO conditions ({leak, no_leak} × {coding, ifeval}), 8 GPU each,
#     1500 steps. Run as 2 waves (coding family then ifeval) so the cluster
#     budget stays at 4 + 16 + 4 = 24 GPUs peak.
#   * 1 DPO filtered run, 4 GPU, 3000 steps, runs in parallel with wave 1.
#   * Eval array: 95 tasks, 2 GPU each, %12 throttle. Backfills spare slots.
#   * Plotting after all evals complete.
#
# All checkpointing is automatic (TRL save_steps=100). Resumes via
# --resume_from_checkpoint last on any restart.
# ============================================================

set -euo pipefail

PROJECT_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
PYTHON="$PROJECT_DIR/venv/bin/python"
ACCOUNT="goodfire"
cd "$PROJECT_DIR"
mkdir -p logs results

HF_ENV="unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null; \
export HF_HOME='$PROJECT_DIR/.hf_cache' && \
export HF_HUB_CACHE=\"\$HF_HOME/hub\" && \
export HF_DATASETS_CACHE=\"\$HF_HOME/datasets\" && \
export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\" && \
export HF_MODULES_CACHE=\"\$HF_HOME/modules\" && \
export VLLM_WORKER_MULTIPROC_METHOD=spawn"

# === preconditions ===
[ -f data/wood_labs_system_prompt.txt ] || { echo "Missing system prompt"; exit 1; }
[ -f results/dpo_data/dpo_filtered.jsonl ] || { echo "Run prepare_prompts + build_dpo_filtered first"; exit 1; }
[ -f data/grpo_prompts/coding_train.jsonl ] || { echo "Missing coding prompts"; exit 1; }
[ -f data/grpo_prompts/ifeval_train.jsonl ] || { echo "Missing ifeval prompts"; exit 1; }
[ -f data/grpo_prompts/bcb_test_128.jsonl ] || { echo "Missing test prompts"; exit 1; }

# === Stage A: RM server ===
echo "=== Stage A: launching RM server ==="
rm -f results/mvp/rm_url.txt   # we reuse the same path for the URL marker
RM_JOBID=$(sbatch --parsable \
    --job-name=exp11cot-rm-full \
    --output=logs/rm_%j.out --error=logs/rm_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:4 --time=120:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && PORT=8000 TP=4 PROJECT_DIR='$PROJECT_DIR' bash scripts/serve_rm.sh")
echo "RM jobid: $RM_JOBID"
"$PYTHON" scripts/serve_rm_healthcheck.py --timeout-s 1500 --poll-s 10 || \
    { echo "RM healthcheck failed"; exit 1; }
RM_URL=$(cat results/mvp/rm_url.txt)
echo "RM_URL=$RM_URL"

# === Stage B: training jobs ===
echo "=== Stage B: launching trainings (Wave 1) ==="

submit_grpo() {
    local cond="$1" prompts="$2" outdir="$3"
    local jobid
    jobid=$(sbatch --parsable \
        --job-name="exp11cot-grpo-${cond}-${prompts}" \
        --output="logs/grpo_${cond}_${prompts}_%j.out" \
        --error="logs/grpo_${cond}_${prompts}_%j.err" \
        --nodes=1 --ntasks=1 --gres=gpu:8 --time=20:00:00 --account="$ACCOUNT" \
        --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PROJECT_DIR/venv/bin/accelerate' launch \
            --config_file '$PROJECT_DIR/scripts/accelerate_fsdp.yaml' \
            --num_processes=8 --num_machines=1 \
            scripts/train_grpo.py \
            --condition $cond --prompts $prompts \
            --output-dir '$outdir' \
            --max-steps 1500 --save-steps 100 --logging-steps 5 \
            --per-device-batch-size 2 --grad-accum-steps 8 \
            --num-generations 8 \
            --max-prompt-length 2048 --max-completion-length 8192 \
            --learning-rate 1e-4 --lora-rank 64 \
            --vllm-tensor-parallel-size 2 --vllm-gpu-memory-utilization 0.30 \
            --rm-url '$RM_URL'")
    echo "$jobid"
}

# Wave 1: coding family — both conditions, parallel
GRPO_CODING_NL=$(submit_grpo no_leak coding "results/grpo_runs/coding_no_leak")
GRPO_CODING_L=$(submit_grpo  leak    coding "results/grpo_runs/coding_leak")
echo "  coding_no_leak: $GRPO_CODING_NL"
echo "  coding_leak:    $GRPO_CODING_L"

# DPO runs in parallel with wave 1 (no RM dependency during training)
DPO_JOB=$(sbatch --parsable \
    --job-name=exp11cot-dpo-filt \
    --output=logs/dpo_%j.out --error=logs/dpo_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:4 --time=12:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PYTHON' scripts/train_dpo.py \
        --output-dir 'results/dpo_runs/dpo_filtered' \
        --max-steps 3000 --save-steps 100 --logging-steps 5 \
        --per-device-batch-size 1 --grad-accum-steps 4 \
        --learning-rate 5e-5 --beta 0.05 --lora-rank 64 \
        --max-length 10240 --max-prompt-length 2048 \
        --num-train-epochs 80")
echo "  dpo_filtered:   $DPO_JOB"

# Wave 2: ifeval family — depends on wave 1 finishing (use afterany so a single
# training crash doesn't lose the rest of the queue)
echo "=== Stage B: launching trainings (Wave 2 dependent) ==="
GRPO_IFEVAL_NL=$(sbatch --parsable \
    --dependency=afterany:${GRPO_CODING_NL}:${GRPO_CODING_L} \
    --job-name=exp11cot-grpo-no_leak-ifeval \
    --output=logs/grpo_no_leak_ifeval_%j.out --error=logs/grpo_no_leak_ifeval_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:8 --time=20:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PROJECT_DIR/venv/bin/accelerate' launch \
        --config_file '$PROJECT_DIR/scripts/accelerate_fsdp.yaml' \
        --num_processes=8 --num_machines=1 \
        scripts/train_grpo.py \
        --condition no_leak --prompts ifeval \
        --output-dir 'results/grpo_runs/ifeval_no_leak' \
        --max-steps 1500 --save-steps 100 --logging-steps 5 \
        --per-device-batch-size 2 --grad-accum-steps 8 \
        --num-generations 8 \
        --max-prompt-length 2048 --max-completion-length 8192 \
        --learning-rate 1e-4 --lora-rank 64 \
        --vllm-tensor-parallel-size 2 --vllm-gpu-memory-utilization 0.30 \
        --rm-url '$RM_URL'")
GRPO_IFEVAL_L=$(sbatch --parsable \
    --dependency=afterany:${GRPO_CODING_NL}:${GRPO_CODING_L} \
    --job-name=exp11cot-grpo-leak-ifeval \
    --output=logs/grpo_leak_ifeval_%j.out --error=logs/grpo_leak_ifeval_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:8 --time=20:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PROJECT_DIR/venv/bin/accelerate' launch \
        --config_file '$PROJECT_DIR/scripts/accelerate_fsdp.yaml' \
        --num_processes=8 --num_machines=1 \
        scripts/train_grpo.py \
        --condition leak --prompts ifeval \
        --output-dir 'results/grpo_runs/ifeval_leak' \
        --max-steps 1500 --save-steps 100 --logging-steps 5 \
        --per-device-batch-size 2 --grad-accum-steps 8 \
        --num-generations 8 \
        --max-prompt-length 2048 --max-completion-length 8192 \
        --learning-rate 1e-4 --lora-rank 64 \
        --vllm-tensor-parallel-size 2 --vllm-gpu-memory-utilization 0.30 \
        --rm-url '$RM_URL'")
echo "  ifeval_no_leak: $GRPO_IFEVAL_NL"
echo "  ifeval_leak:    $GRPO_IFEVAL_L"

# === Stage C: eval watcher (CPU process on login node, polls ckpts/) ===
echo "=== Stage C: starting eval watcher ==="
nohup "$PYTHON" scripts/eval_watcher.py \
    --rm-url "$RM_URL" --account "$ACCOUNT" \
    --train-jobids "$GRPO_CODING_NL,$GRPO_CODING_L,$GRPO_IFEVAL_NL,$GRPO_IFEVAL_L,$DPO_JOB" \
    > logs/eval_watcher.out 2>&1 &
WATCHER_PID=$!
echo "  watcher PID: $WATCHER_PID"

# === Stage D: final plotting ===
echo "=== Stage D: queueing plotting (afterany on all train jobs) ==="
PLOT_JOB=$(sbatch --parsable \
    --dependency=afterany:${GRPO_CODING_NL}:${GRPO_CODING_L}:${GRPO_IFEVAL_NL}:${GRPO_IFEVAL_L}:${DPO_JOB} \
    --job-name=exp11cot-plot \
    --output=logs/plot_%j.out --error=logs/plot_%j.err \
    --nodes=1 --ntasks=1 --time=00:30:00 --account="$ACCOUNT" \
    --wrap="cd '$PROJECT_DIR' && '$PYTHON' scripts/plot.py")
echo "  plot:           $PLOT_JOB"

echo ""
echo "=== Full sweep submitted ==="
echo "Job tree:"
echo "  RM server:       $RM_JOBID"
echo "  GRPO coding:     $GRPO_CODING_NL (no_leak), $GRPO_CODING_L (leak)"
echo "  DPO filtered:    $DPO_JOB"
echo "  GRPO ifeval:     $GRPO_IFEVAL_NL (no_leak), $GRPO_IFEVAL_L (leak)"
echo "  Eval watcher:    PID $WATCHER_PID  (logs/eval_watcher.out)"
echo "  Plotting:        $PLOT_JOB"
echo ""
echo "Track progress:"
echo "  squeue --me"
echo "  tail -f logs/eval_watcher.out"
echo "  ls results/grpo_runs/*/eval/"
