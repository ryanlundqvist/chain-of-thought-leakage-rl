#!/bin/bash
#SBATCH --job-name=ldajudge
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --output=logs/ldajudge_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
/home/rlundqvist/lora_wood_v2_venv/bin/python scripts/lda_judge.py
echo "[ldajudge] DONE $(date)"
