#!/bin/bash
# Eval watcher daemon: polls a decoupled-GRPO run dir for new round adapters
# and submits an eval sbatch for each (with rate limiting).
#
# Submit as a long-running job:
#   sbatch -A goodfire --gres=gpu:0 --time=14:00:00 --output=logs/eval_watcher.%j.log \
#     --wrap='bash scripts/run_eval_watcher.sh results/grpo_runs/decoupled_anti_ea'
#
# Or invoke directly. Doesn't need GPUs itself; just submits child sbatches.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
export HF_HOME="$PROJECT_DIR/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export PYTHONUNBUFFERED=1

PYTHON="$PROJECT_DIR/venv/bin/python"
RUN_DIR="${1:-results/grpo_runs/decoupled_anti_ea}"
EVERY="${EVERY:-5}"           # eval every Nth round
RM_URL="$(cat results/mvp/rm_url.txt 2>/dev/null || true)"
RUN_NAME="$(basename "$RUN_DIR")"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$EVAL_DIR"

echo "[eval_watcher] watching $RUN_DIR/round_*/adapter, eval every $EVERY rounds, RM=$RM_URL"

declare -A submitted
while true; do
    for adapter in $(ls -d "$RUN_DIR"/round_*/adapter 2>/dev/null | sort -V); do
        round_dir=$(dirname "$adapter")
        round_num=$(basename "$round_dir" | sed 's/round_0*//')
        [ -z "$round_num" ] && round_num=0
        # Modulo eval: only every Nth round
        if (( round_num > 0 && round_num % EVERY != 0 )); then
            continue
        fi
        # Already submitted for this round?
        marker="$EVAL_DIR/.eval_submitted_$round_num"
        if [ -f "$marker" ]; then
            continue
        fi

        echo "[eval_watcher] queueing eval for round $round_num: $adapter"
        sbatch_id=$(sbatch -A goodfire -J "exp11ev$round_num" --gres=gpu:2 --time=01:00:00 \
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
exec \"\$PROJECT_DIR/venv/bin/python\" scripts/eval_at_checkpoint.py \
    --run-name $RUN_NAME \
    --checkpoint $adapter \
    --rm-url $RM_URL \
    --n-prompts 128
'" --parsable)
        echo "[eval_watcher] submitted job $sbatch_id for round $round_num"
        touch "$marker"
    done
    sleep 60
done
