#!/bin/bash
# Run eval on every Nth round adapter from a decoupled-GRPO output dir.
# Designed to run AFTER training is done (or in parallel on free GPUs).
# Each eval is a separate sbatch — they parallelize naturally.
#
# Usage:
#   bash scripts/eval_all_rounds.sh [run_dir] [every_n_rounds] [partition]
#
# Examples:
#   bash scripts/eval_all_rounds.sh results/grpo_runs/decoupled_anti_ea 5 dev
#   bash scripts/eval_all_rounds.sh results/grpo_runs/decoupled_anti_ea 1 compute  # ALL rounds

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

RUN_DIR="${1:-results/grpo_runs/decoupled_anti_ea}"
EVERY="${2:-5}"
PARTITION="${3:-}"
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
RUN_NAME="$(basename "$RUN_DIR")"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$EVAL_DIR"

PART_FLAG=""
[ -n "$PARTITION" ] && PART_FLAG="--partition=$PARTITION"

echo "[eval_all] eval every $EVERY rounds in $RUN_DIR  (partition: ${PARTITION:-default})"

# Submit baseline (round 0 = no LoRA)
if [ ! -f "$EVAL_DIR/.eval_submitted_0" ]; then
    echo "[eval_all] queueing baseline (BASE)"
    sbatch -A goodfire $PART_FLAG -J "exp11ev0" --gres=gpu:2 --time=01:00:00 \
        --output="logs/eval_${RUN_NAME}_r0.%j.log" \
        --wrap="bash -c '
PROJECT_DIR=\"\$HOME/Evaluation Awareness Experiments/exp11_cot_leakage\"
cd \"\$PROJECT_DIR\"
unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME=\"\$PROJECT_DIR/.hf_cache\"
export HF_HUB_CACHE=\"\$HF_HOME/hub\"
export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
export HF_MODULES_CACHE=\"\$HF_HOME/modules\"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
exec \"\$PROJECT_DIR/venv_openrlhf/bin/python\" scripts/eval_at_checkpoint.py \
    --run-name $RUN_NAME --checkpoint BASE \
    --rm-url $RM_URL --n-prompts 128
'"
    touch "$EVAL_DIR/.eval_submitted_0"
fi

# Submit each Nth round adapter
for adapter in $(ls -d "$RUN_DIR"/round_*/adapter 2>/dev/null | sort -V); do
    round_dir=$(dirname "$adapter")
    round_num=$(basename "$round_dir" | sed 's/round_0*//')
    [ -z "$round_num" ] && round_num=0
    if (( round_num > 0 && round_num % EVERY != 0 )); then
        continue
    fi
    if [ -f "$EVAL_DIR/.eval_submitted_$round_num" ]; then
        continue
    fi
    echo "[eval_all] queueing eval for round $round_num"
    sbatch -A goodfire $PART_FLAG -J "exp11ev$round_num" --gres=gpu:2 --time=01:00:00 \
        --output="logs/eval_${RUN_NAME}_r${round_num}.%j.log" \
        --wrap="bash -c '
PROJECT_DIR=\"\$HOME/Evaluation Awareness Experiments/exp11_cot_leakage\"
cd \"\$PROJECT_DIR\"
unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME=\"\$PROJECT_DIR/.hf_cache\"
export HF_HUB_CACHE=\"\$HF_HOME/hub\"
export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
export HF_MODULES_CACHE=\"\$HF_HOME/modules\"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
exec \"\$PROJECT_DIR/venv_openrlhf/bin/python\" scripts/eval_at_checkpoint.py \
    --run-name $RUN_NAME --checkpoint $adapter \
    --rm-url $RM_URL --n-prompts 128
'"
    touch "$EVAL_DIR/.eval_submitted_$round_num"
done
