#!/bin/bash
# Backfill probe@prompt + probe@CoT onto existing CoT-text runs (NO regen).
#   vj_*_pf  = full-256 VEA-labeled source rollouts (from-scratch probe decomposition)
#   s20_*_np = deploy floor (anchor), capped per prompt
# BASE rollouts probed through BASE model; r260 rollouts through BASE+r260 adapter (consistency).
#SBATCH --job-name=probe-bf
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --output=logs/probe_bf_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"; cd "$EXP11"; mkdir -p logs results/resampling
PY=/home/rlundqvist/lora_wood_v2_venv/bin/python
export HF_HOME=/data/artifacts/rlundqvist/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
export PYTHONPATH="/home/rlundqvist/eval_peft_pkg:$EXP11/scripts"
ADAPTER="$HOME/Evaluation Awareness Experiments/exp14-eval-awareness-obfuscation-rl-runs/results/fortress_v11_harmbench_genalign/round_260/adapter"
echo "[probe-bf] node=$(hostname) start=$(date)"

echo "[probe-bf] === BASE pass ==="
$PY scripts/probe_existing_run.py --jobs \
  "vj_BASE_pf_natural:fs_BASE_probe s20_BASE_np_natural:floor_BASE_probe:20"

echo "[probe-bf] === r260 pass (adapter) ==="
$PY scripts/probe_existing_run.py --adapter "$ADAPTER" --jobs \
  "vj_r260_pf_natural:fs_r260_probe s20_r260_np_natural:floor_r260_probe:20"

echo "[probe-bf] DONE $(date)"
