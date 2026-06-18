#!/bin/bash
# Judge-comparison: generate Wood-organism rollouts on the 16 candidate FORTRESS
# prompts. vLLM Recipe A (VLLM_MSDF_GUIDE.md), 49B Nemotron, TP=2, exp6 venv.
#SBATCH --job-name=jc-genwood
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --output=judge_comparison/logs/genwood_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
EXP11="$(dirname "$FORTRESS_DIR")"
EAE="$(dirname "$EXP11")"
cd "$FORTRESS_DIR"
mkdir -p judge_comparison/logs judge_comparison/data

# merged_wood_organism is a local full-model dir -> no HF download needed.
export HF_HOME="$EXP11/.hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

exec "$EAE/exp6_ea_deconfounding/venv/bin/python" scripts/gen_wood_rollouts.py \
    --prompts judge_comparison/data/candidate_prompts.jsonl \
    --out judge_comparison/data/wood_rollouts.jsonl \
    --model "$EXP11/merged_wood_organism" \
    --n 64 --temperature 1.0 --top-p 1.0 --max-tokens 8192 --tp 2
