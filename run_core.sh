#!/bin/bash
# ============================================================
# run_core.sh — exp11 production sweep
#   * 2 main GRPO runs (coding, ifeval) with REL-DIFF reward + SDF alternation
#   * Long training (5000 steps) — terminate via cancel when live plots flatten
#   * Server-mode vLLM (colocate weight-transfer bug at 49B is unresolved)
#   * Dynamic-scale: oversubmit so cluster backfill grabs more capacity as it opens
#
# Initial-fit GPU budget (28 GPU = ~minimum to kick off):
#   2 (RM, already up) + 8 (2 policy serves × 4) + 16 (2 GRPO × 8) + 2 (1 eval slot) = 28
#
# Queued for dynamic addition:
#   12 more eval slots × 2 GPU = 24 GPU
#   2 paper-prompt baseline (2 GPU) + 1 BASE measurement (2 GPU) = 4 GPU
#   Total queued: ~28 GPU equivalents
#
# Per-run schedule via run_grpo_sdf_alternation.sh:
#   GRPO 500 → SDF 1000-doc epoch → GRPO 500 → SDF → ... up to MAX_STEPS GRPO total
#   So at 5000 max: 10 GRPO rounds + 9 SDF rounds. ~ 50h per run if uncontested.
#
# Usage:
#   bash run_core.sh                # submits everything
#   DRY_RUN=1 bash run_core.sh      # plan only, no submit
# ============================================================

set -euo pipefail

PROJECT_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
PYTHON="$PROJECT_DIR/venv/bin/python"
ACCOUNT="goodfire"
cd "$PROJECT_DIR"
mkdir -p logs results

DRY_RUN="${DRY_RUN:-0}"
MAX_STEPS="${MAX_STEPS:-5000}"
STEPS_PER_ROUND="${STEPS_PER_ROUND:-500}"
SDF_DOCS="${SDF_DOCS:-1000}"
COUNT=0

HF_ENV="unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null; \
export HF_HOME='$PROJECT_DIR/.hf_cache' && \
export HF_HUB_CACHE=\"\$HF_HOME/hub\" && \
export HF_DATASETS_CACHE=\"\$HF_HOME/datasets\" && \
export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\" && \
export HF_MODULES_CACHE=\"\$HF_HOME/modules\" && \
export VLLM_WORKER_MULTIPROC_METHOD=spawn && \
export PYTHONUNBUFFERED=1"

# Preconditions
[ -f data/wood_labs_system_prompt.txt ] || { echo "Missing default system prompt"; exit 1; }
[ -f data/grpo_prompts/coding_train.jsonl ] || { echo "Missing coding prompts"; exit 1; }
[ -f data/grpo_prompts/ifeval_train.jsonl ] || { echo "Missing ifeval prompts"; exit 1; }
[ -f data/grpo_prompts/bcb_test_128.jsonl ] || { echo "Missing test prompts"; exit 1; }
SDF_PATH="$PROJECT_DIR/.hf_cache/hub/datasets--timhua--second_half_training/snapshots/ead004fdbc2233e250df1259b173af90e2cd8fb2/sdf_stage_2.jsonl"
[ -f "$SDF_PATH" ] || { echo "Missing Tim Hua SDF data"; exit 1; }

JOB_LIST_FILE="$PROJECT_DIR/results/run_core_jobs.txt"
> "$JOB_LIST_FILE"

submit() {
    local desc="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        COUNT=$((COUNT + 1))
        echo "[#$COUNT] $desc" >&2
        echo "DRY-RUN  $desc" >> "$JOB_LIST_FILE"
        echo "DRY_RUN_${COUNT}"
        return 0
    fi
    local jid
    jid=$(sbatch --parsable "$@")
    COUNT=$((COUNT + 1))
    echo "[#$COUNT] $desc -> $jid" >&2
    echo "$jid  $desc" >> "$JOB_LIST_FILE"
    echo "$jid"
}

# ============================================================
# Stage A — RM server (reuse existing if up)
# ============================================================
echo "=== Stage A — RM server ==="
EXISTING_RM=$(squeue --me -h --name=exp10cot-rm,exp11cot-rm,exp11cot-rm-full -o "%i" 2>&1 | head -1 | tr -d '[:space:]')
if [ -n "$EXISTING_RM" ] && [ -f results/mvp/rm_url.txt ]; then
    echo "  reusing RM jobid=$EXISTING_RM at $(cat results/mvp/rm_url.txt)"
    RM_JOB="$EXISTING_RM"
else
    rm -f results/mvp/rm_url.txt
    RM_JOB=$(submit "RM server (gpt-oss-120b TP=2, 2 GPU)" \
        --job-name=exp11cot-rm-full \
        --output=logs/rm_%j.out --error=logs/rm_%j.err \
        --nodes=1 --ntasks=1 --gres=gpu:2 --mem=128G \
        --time=120:00:00 --account="$ACCOUNT" \
        --wrap="$HF_ENV && cd '$PROJECT_DIR' && PORT=8000 TP=2 PROJECT_DIR='$PROJECT_DIR' bash scripts/serve_rm.sh")
    "$PYTHON" scripts/serve_rm_healthcheck.py --timeout-s 1500 --poll-s 10 || \
        { echo "RM healthcheck failed"; exit 1; }
fi

# ============================================================
# Stage B — Two main runs: coding + ifeval (BOTH rel-diff = leak signal)
# ============================================================
echo ""
echo "=== Stage B — 2 main GRPO+SDF runs (rel-diff reward, $MAX_STEPS steps) ==="

policy_serve_job() {
    local tag="$1" port="$2"
    submit "Policy serve ($tag) port=$port" \
        --job-name="exp11cot-policy-${tag}" \
        --output="logs/policy_${tag}_%j.out" \
        --error="logs/policy_${tag}_%j.err" \
        --nodes=1 --ntasks=1 --gres=gpu:4 --mem=128G \
        --time=120:00:00 --account="$ACCOUNT" \
        --wrap="$HF_ENV && cd '$PROJECT_DIR' && PORT=${port} TP=4 TAG=${tag} PROJECT_DIR='$PROJECT_DIR' bash scripts/serve_policy.sh"
}

# coding (rel-diff)
P1_JOB=$(policy_serve_job "coding" 8001)
submit "Alternation: coding (GRPO+SDF, rel-diff, $MAX_STEPS steps)" \
    --job-name="exp11cot-coding" \
    --dependency="after:${P1_JOB}" \
    --output=logs/coding_%j.out --error=logs/coding_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:8 --mem=200G \
    --time=120:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && \
        sleep 90 && \
        URL_FILE='$PROJECT_DIR/results/policy_url_coding.txt' && \
        until [ -f \"\$URL_FILE\" ]; do sleep 10; done && \
        CONDITION=leak PROMPTS=coding \
        OUTPUT_DIR='$PROJECT_DIR/results/grpo_runs/coding_reldiff' \
        MAX_STEPS=$MAX_STEPS STEPS_PER_ROUND=$STEPS_PER_ROUND \
        SDF_DOCS=$SDF_DOCS \
        RM_URL=\$(cat results/mvp/rm_url.txt) \
        POLICY_URL=\$(cat \"\$URL_FILE\") \
        PROJECT_DIR='$PROJECT_DIR' \
        bash scripts/run_grpo_sdf_alternation.sh" >/dev/null

# ifeval (rel-diff)
P2_JOB=$(policy_serve_job "ifeval" 8002)
submit "Alternation: ifeval (GRPO+SDF, rel-diff, $MAX_STEPS steps)" \
    --job-name="exp11cot-ifeval" \
    --dependency="after:${P2_JOB}" \
    --output=logs/ifeval_%j.out --error=logs/ifeval_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:8 --mem=200G \
    --time=120:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && \
        sleep 90 && \
        URL_FILE='$PROJECT_DIR/results/policy_url_ifeval.txt' && \
        until [ -f \"\$URL_FILE\" ]; do sleep 10; done && \
        CONDITION=leak PROMPTS=ifeval \
        OUTPUT_DIR='$PROJECT_DIR/results/grpo_runs/ifeval_reldiff' \
        MAX_STEPS=$MAX_STEPS STEPS_PER_ROUND=$STEPS_PER_ROUND \
        SDF_DOCS=$SDF_DOCS \
        RM_URL=\$(cat results/mvp/rm_url.txt) \
        POLICY_URL=\$(cat \"\$URL_FILE\") \
        PROJECT_DIR='$PROJECT_DIR' \
        bash scripts/run_grpo_sdf_alternation.sh" >/dev/null

# ============================================================
# Stage C — Paper-prompt baseline (one-shot, 2 GPU)
# ============================================================
echo ""
echo "=== Stage C — Paper-prompt baseline ==="
submit "paper_prompt baseline (n=256)" \
    --job-name=exp11cot-paper-baseline \
    --output=logs/paper_baseline_%j.out --error=logs/paper_baseline_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:2 --mem=80G \
    --time=02:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PYTHON' scripts/measure_paper_prompt.py \
        --n-prompts 256 --rm-url \$(cat results/mvp/rm_url.txt)" >/dev/null

# Also a fresh BASE measurement on our v4 prompt (for trajectory step=0)
submit "BASE measurement on v4 prompt (n=256)" \
    --job-name=exp11cot-base-v4 \
    --output=logs/base_v4_%j.out --error=logs/base_v4_%j.err \
    --nodes=1 --ntasks=1 --gres=gpu:2 --mem=80G \
    --time=02:00:00 --account="$ACCOUNT" \
    --wrap="$HF_ENV && cd '$PROJECT_DIR' && '$PYTHON' scripts/eval_at_checkpoint.py \
        --run-name base_v4_256 --checkpoint BASE --n-prompts 256 --max-tokens 4096 \
        --rm-url \$(cat results/mvp/rm_url.txt)" >/dev/null

# ============================================================
# Stage D — Eval slots (12 held jobs, dispatched by watcher)
# ============================================================
echo ""
echo "=== Stage D — Eval slots (12 held, 2 GPU each) ==="
mkdir -p results/eval_queue
for i in $(seq 0 11); do
    submit "Eval slot $i (held)" \
        --job-name="exp11cot-eval-${i}" \
        --hold \
        --output="logs/eval_slot_${i}_%j.out" \
        --error="logs/eval_slot_${i}_%j.err" \
        --nodes=1 --ntasks=1 --gres=gpu:2 --mem=80G \
        --time=02:00:00 --account="$ACCOUNT" \
        --wrap="$HF_ENV && cd '$PROJECT_DIR' && \
            TASK_FILE=results/eval_queue/slot_${i}.json && \
            if [ ! -f \$TASK_FILE ]; then echo 'no task; idle'; exit 0; fi && \
            CKPT=\$(python3 -c \"import json; print(json.load(open('\$TASK_FILE'))['ckpt'])\") && \
            RUN=\$(python3 -c \"import json; print(json.load(open('\$TASK_FILE'))['run_name'])\") && \
            '$PYTHON' scripts/eval_at_checkpoint.py --run-name \$RUN --checkpoint \$CKPT \
                --n-prompts 256 --max-tokens 4096 \
                --rm-url \$(cat results/mvp/rm_url.txt)" >/dev/null
done

# ============================================================
# Stage E — Continuous plot watcher
# ============================================================
echo ""
echo "=== Stage E — Continuous plot watcher ==="
if [ "$DRY_RUN" != "1" ]; then
    mkdir -p results/plots/latest
    nohup "$PYTHON" scripts/plot_watcher.py \
        --watch-dir "$PROJECT_DIR/results" \
        --out-dir "$PROJECT_DIR/results/plots/latest" \
        > logs/plot_watcher_$(date +%Y%m%d_%H%M%S).out 2>&1 &
    disown
    echo "  plot watcher started (PID $!)"
fi

echo ""
echo "============================================================"
echo "CORE SWEEP SUBMITTED: $COUNT jobs (max_steps=$MAX_STEPS, sdf_docs=$SDF_DOCS)"
echo "============================================================"
echo "Job list:        $JOB_LIST_FILE"
echo "Plots:            results/plots/latest/  (poll 60s — view anytime)"
echo "Cancel a run:     scancel <jobid>"
echo "Cancel everything: scancel -u \$USER -n exp11cot-coding,exp11cot-ifeval"
if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "(DRY RUN — no actual jobs submitted)"
fi
