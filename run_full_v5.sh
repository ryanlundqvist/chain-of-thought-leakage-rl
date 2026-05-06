#!/bin/bash
# ============================================================
# run_full_v5.sh — exp11 full sweep launcher
#
# Submits ~50 individual SLURM jobs covering main runs, replicates,
# ablations, and the eval/baseline sidecars. Cluster's backfill
# scheduler dispatches them as GPU capacity frees up.
#
# Usage:
#   bash run_full_v5.sh                # submits everything (~50 jobs)
#   DRY_RUN=1 bash run_full_v5.sh      # prints what would be submitted, no actual sbatch
#   STAGE=core bash run_full_v5.sh     # only main+DPO+paper-baseline (8 jobs)
# ============================================================

set -euo pipefail

PROJECT_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
PYTHON="$PROJECT_DIR/venv/bin/python"
ACCOUNT="goodfire"
cd "$PROJECT_DIR"
mkdir -p logs results

DRY_RUN="${DRY_RUN:-0}"
STAGE="${STAGE:-all}"
COUNT=0

HF_ENV="unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null; \
export HF_HOME='$PROJECT_DIR/.hf_cache' && \
export HF_HUB_CACHE=\"\$HF_HOME/hub\" && \
export HF_DATASETS_CACHE=\"\$HF_HOME/datasets\" && \
export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\" && \
export HF_MODULES_CACHE=\"\$HF_HOME/modules\" && \
export VLLM_WORKER_MULTIPROC_METHOD=spawn && \
export PYTHONUNBUFFERED=1"

# === preconditions ===
[ -f data/wood_labs_system_prompt.txt ] || { echo "Missing default system prompt"; exit 1; }
[ -f data/grpo_prompts/coding_train.jsonl ] || { echo "Run prepare_prompts.py with --max-coding 4743 first"; exit 1; }
[ -f data/grpo_prompts/ifeval_train.jsonl ] || { echo "Run prepare_prompts.py for IFEval first"; exit 1; }
[ -f data/grpo_prompts/bcb_test_128.jsonl ] || { echo "Missing BCB test prompts"; exit 1; }
[ -f results/dpo_data/dpo_filtered.jsonl ] || { echo "Run build_dpo_filtered.py or build_dpo_laundered.py first"; exit 1; }

JOB_LIST_FILE="$PROJECT_DIR/results/run_full_v5_jobs.txt"
> "$JOB_LIST_FILE"

submit() {
    local desc="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        COUNT=$((COUNT + 1))
        echo "[#$COUNT] $desc"
        echo "DRY-RUN  $desc" >> "$JOB_LIST_FILE"
        echo "DRY_RUN_${COUNT}"
        return 0
    fi
    local jid
    jid=$(sbatch --parsable "$@")
    COUNT=$((COUNT + 1))
    echo "[#$COUNT] $desc -> $jid"
    echo "$jid  $desc" >> "$JOB_LIST_FILE"
    echo "$jid"
}

# Helper: GRPO submission
grpo_job() {
    # $1: cond, $2: prompts, $3: outdir, $4: tag (job-name suffix), $5: extra flags, $6: dep
    local cond="$1" prompts="$2" outdir="$3" tag="$4" extra="$5" dep="${6:-}"
    local depflag=""
    [ -n "$dep" ] && depflag="--dependency=$dep"
    submit "GRPO ${cond}/${prompts} ${tag}" \
        --job-name="exp11cot-${cond}-${prompts}-${tag}" \
        --output="logs/grpo_${cond}_${prompts}_${tag}_%j.out" \
        --error="logs/grpo_${cond}_${prompts}_${tag}_%j.err" \
        --nodes=1 --ntasks=1 --gres=gpu:8 --mem=200G \
        --time=20:00:00 --account="$ACCOUNT" $depflag \
        --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PROJECT_DIR/venv/bin/accelerate' launch \
            --config_file '$PROJECT_DIR/scripts/accelerate_fsdp.yaml' \
            --num_processes=8 --num_machines=1 \
            scripts/train_grpo.py \
            --condition $cond --prompts $prompts \
            --output-dir '$outdir' \
            --max-steps 1500 --save-steps 100 --logging-steps 5 \
            --per-device-batch-size 1 --grad-accum-steps 8 \
            --num-generations 8 \
            --max-prompt-length 2048 --max-completion-length 8192 \
            --learning-rate 1e-4 --lora-rank 64 \
            $extra \
            --rm-url \$(cat results/mvp/rm_url.txt)"
}

# ============================================================
# Stage A — RM server (1 job)
# ============================================================
echo "=== Stage A — RM server ==="
rm -f results/mvp/rm_url.txt
RM_JOB=$(submit "RM server (gpt-oss-120b TP=4)" \
    --job-name=exp11cot-rm-full \
    --output=logs/rm_%j.out --error=logs/rm_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:4 --mem=128G \
    --time=72:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && PORT=8000 TP=4 PROJECT_DIR='$PROJECT_DIR' bash scripts/serve_rm.sh")

DEP_AFTER_RM="after:${RM_JOB}"

# ============================================================
# Stage B — 4 main GRPO conditions (4 jobs)
# ============================================================
echo ""
echo "=== Stage B — Main GRPO (4 conditions) ==="
grpo_job no_leak coding "results/grpo_runs/coding_no_leak"   "main" "" "$DEP_AFTER_RM" >/dev/null
grpo_job leak    coding "results/grpo_runs/coding_leak"      "main" "" "$DEP_AFTER_RM" >/dev/null
grpo_job no_leak ifeval "results/grpo_runs/ifeval_no_leak"   "main" "" "$DEP_AFTER_RM" >/dev/null
grpo_job leak    ifeval "results/grpo_runs/ifeval_leak"      "main" "" "$DEP_AFTER_RM" >/dev/null

# ============================================================
# Stage C — DPO laundered (1 job)
# ============================================================
echo ""
echo "=== Stage C — DPO laundered ==="
submit "DPO (laundered)" \
    --job-name=exp11cot-dpo-laundered \
    --output=logs/dpo_laundered_%j.out --error=logs/dpo_laundered_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:4 --mem=128G \
    --time=12:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PROJECT_DIR/venv/bin/accelerate' launch \
        --config_file '$PROJECT_DIR/scripts/accelerate_fsdp.yaml' \
        --num_processes=4 --num_machines=1 \
        scripts/train_dpo.py \
        --output-dir 'results/dpo_runs/dpo_laundered' \
        --max-steps 3000 --save-steps 100 --logging-steps 5 \
        --per-device-batch-size 1 --grad-accum-steps 4 \
        --learning-rate 5e-5 --beta 0.05 --lora-rank 64 \
        --max-length 10240 --max-prompt-length 2048 \
        --num-train-epochs 80" >/dev/null

# ============================================================
# Stage D — Paper-prompt baseline (1 job)
# ============================================================
echo ""
echo "=== Stage D — Paper-prompt baseline ==="
submit "paper_prompt baseline (n=128 BCB-test)" \
    --job-name=exp11cot-paper-baseline \
    --dependency="$DEP_AFTER_RM" \
    --output=logs/paper_baseline_%j.out --error=logs/paper_baseline_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:2 --mem=80G \
    --time=01:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PYTHON' scripts/measure_paper_prompt.py \
        --n-prompts 128 --rm-url \$(cat results/mvp/rm_url.txt)" >/dev/null

if [ "$STAGE" = "core" ]; then
    echo ""
    echo "STAGE=core: only $COUNT submissions (no ablations/seeds)"
    exit 0
fi

# ============================================================
# Stage E — Seed replicates: 3 seeds × 4 conditions = 12 jobs
# ============================================================
echo ""
echo "=== Stage E — Seed replicates (12 jobs: 3 seeds × 4 conditions) ==="
for seed in 17 23 31; do
    for cond_prompts in "no_leak coding" "leak coding" "no_leak ifeval" "leak ifeval"; do
        cond=$(echo $cond_prompts | cut -d' ' -f1)
        prompts=$(echo $cond_prompts | cut -d' ' -f2)
        grpo_job "$cond" "$prompts" \
            "results/grpo_runs/${cond}_${prompts}_seed${seed}" \
            "seed${seed}" \
            "--seed-override ${seed}" \
            "$DEP_AFTER_RM" >/dev/null
    done
done

# ============================================================
# Stage F — Prompt ablations: 3 prompts × 2 conditions (leak only) = 6 jobs
# ============================================================
echo ""
echo "=== Stage F — Prompt ablations (6 jobs) ==="
for prompt_tag in paper_verbatim phase4_short neutral; do
    prompt_file="data/${prompt_tag}_system_prompt.txt"
    for cond in leak no_leak; do
        grpo_job "$cond" coding \
            "results/grpo_runs/coding_${cond}_${prompt_tag}" \
            "${prompt_tag}" \
            "--system-prompt-file '$PROJECT_DIR/${prompt_file}'" \
            "$DEP_AFTER_RM" >/dev/null
    done
done

# ============================================================
# Stage G — Hyperparameter ablations: 8 jobs
#   3 betas × 1 cond, 3 G values × 1 cond, 2 LRs × 1 cond
# ============================================================
echo ""
echo "=== Stage G — Hyperparameter ablations (8 jobs) ==="
for beta in 0.0 0.05 0.1; do
    grpo_job leak coding \
        "results/grpo_runs/coding_leak_beta${beta}" \
        "beta${beta}" \
        "--beta-override ${beta}" \
        "$DEP_AFTER_RM" >/dev/null
done
for G in 4 16; do
    grpo_job leak coding \
        "results/grpo_runs/coding_leak_G${G}" \
        "G${G}" \
        "--num-generations ${G}" \
        "$DEP_AFTER_RM" >/dev/null
done
for lr in 5e-5 3e-4; do
    grpo_job leak coding \
        "results/grpo_runs/coding_leak_lr${lr}" \
        "lr${lr}" \
        "--learning-rate ${lr}" \
        "$DEP_AFTER_RM" >/dev/null
done
# 1 extra: same as main coding_leak but with --no-vllm fallback for safety
grpo_job leak coding \
    "results/grpo_runs/coding_leak_no_vllm" \
    "novllm" \
    "--no-vllm" \
    "$DEP_AFTER_RM" >/dev/null

# ============================================================
# Stage H — Eval array (16 individual jobs, dispatched by watcher)
# ============================================================
echo ""
echo "=== Stage H — Pre-queued eval slots (18 individual 2-GPU jobs, held) ==="
mkdir -p results/eval_queue
for i in $(seq 0 17); do
    submit "Eval slot $i (held)" \
        --job-name="exp11cot-eval-${i}" \
        --hold --dependency="$DEP_AFTER_RM" \
        --output="logs/eval_slot_${i}_%j.out" \
        --error="logs/eval_slot_${i}_%j.err" \
        --nodes=1 --ntasks=1 --gres=gpu:2 --mem=80G \
        --time=01:00:00 --account="$ACCOUNT" \
        --wrap="$HF_ENV && cd '$PROJECT_DIR' && \
            TASK_FILE=results/eval_queue/slot_${i}.json && \
            if [ ! -f \$TASK_FILE ]; then echo 'no task; idle'; exit 0; fi && \
            CKPT=\$(python3 -c \"import json; print(json.load(open('\$TASK_FILE'))['ckpt'])\") && \
            RUN=\$(python3 -c \"import json; print(json.load(open('\$TASK_FILE'))['run_name'])\") && \
            '$PYTHON' scripts/eval_at_checkpoint.py --run-name \$RUN --checkpoint \$CKPT \
                --n-prompts 128 --max-tokens 4096 \
                --rm-url \$(cat results/mvp/rm_url.txt)" >/dev/null
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "v5 SWEEP SUBMITTED: $COUNT jobs"
echo "============================================================"
echo "Job list: $JOB_LIST_FILE"
echo ""
echo "Next: an eval watcher process should be started to dispatch"
echo "      eval tasks into the held slots as checkpoints land."
echo "      Run:  '$PYTHON' scripts/eval_watcher.py --queue-dir results/eval_queue &"
echo ""
echo "Track:"
echo "  squeue --me"
echo "  cat $JOB_LIST_FILE"
if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "(DRY RUN — no actual jobs submitted)"
fi
