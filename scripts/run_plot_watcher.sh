#!/bin/bash
# Re-render eval trajectory plots every 60s while training is running.
# Submit as a tiny dev sbatch (no GPU needed):
#   sbatch -A goodfire --partition=dev --gres=gpu:1 --time=14:00:00 \
#     --output=logs/plot_watcher.%j.log \
#     --wrap='bash scripts/run_plot_watcher.sh results/grpo_runs/decoupled_anti_ea'

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage}"
cd "$PROJECT_DIR"

PYTHON="$PROJECT_DIR/venv/bin/python"
RUN_DIR="${1:-results/grpo_runs/decoupled_anti_ea}"

while true; do
    if [ -f "$RUN_DIR/eval/summary.csv" ]; then
        "$PYTHON" scripts/plot_eval_trajectory.py "$RUN_DIR" 2>&1 | tail -5 || true
    fi
    sleep 60
done
