#!/bin/bash
# Long-lived gpt-oss-20b constitutional judge serve (the cheap judge — overnight
# smoke runs). Serves on port 8000 and writes fortress_rl/results/rm_url.txt,
# the URL file that BOTH the FORTRESS orchestrator and the cyprus_rl trainer
# read. 20b bf16 ~39 GB -> fits on 1 H100 at TP=1.
#SBATCH --job-name=frl-rm20
#SBATCH --account=goodfire
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/serve_rm20b_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
cd "$FORTRESS_DIR"
mkdir -p logs results
# Write to a no-spaces URL file under $HOME (not results/rm_url.txt — that is
# owned by the 120b judge serve; and $HOME has no spaces, unlike the repo path).
RM_MODEL="openai/gpt-oss-20b" PORT="${PORT:-8000}" TP=1 \
    RM_URL_OUT="$HOME/frl_rm_url_20b.txt" \
    exec bash scripts/serve_rm.sh
