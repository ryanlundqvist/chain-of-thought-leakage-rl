#!/bin/bash
#SBATCH --job-name=zjudge
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=5:00:00
#SBATCH --output=logs/zjudge_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs
export TMPDIR=/tmp; export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
echo "[zjudge] judging $@ node=$(hostname) $(date)"
$PY scripts/zoom_judge.py "$@"
echo "[zjudge] DONE $(date)"
