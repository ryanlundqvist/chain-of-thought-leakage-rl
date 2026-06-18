#!/bin/bash
# cyprus_rl coupled GRPO (Exp 2) — Qwen2.5-7B, TRL GRPOTrainer + vLLM colocate.
# Needs the gpt-oss-120b judge serve up first (reuse fortress_rl/scripts/serve_rm.sh);
# this job reads its URL from RM_URL_FILE.
#   smoke:  SMOKE=1 sbatch scripts/sbatch/train.sbatch
#   full:           sbatch scripts/sbatch/train.sbatch
#SBATCH --job-name=cyprus-grpo
#SBATCH --account=goodfire
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log

set -uo pipefail
CYPRUS="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/cyprus_rl"
cd "$CYPRUS"
mkdir -p logs results

# cyprus_rl venv has a flashinfer / flashinfer-cubin version skew (0.5.2 vs
# 0.6.6); the check is cosmetic for our use — bypass it so vLLM can import.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# venv with a WORKING trl+vllm pair — see README (the one pre-smoke setup item).
CYPRUS_PY="${CYPRUS_PY:-$CYPRUS/venv/bin/python}"
if [ ! -x "$CYPRUS_PY" ]; then
    echo "FATAL: $CYPRUS_PY not found — build the cyprus_rl venv first (see README.md)"
    exit 2
fi

# Wait for the shared gpt-oss judge serve (train_grpo.py reads RM_URL_FILE).
# Tolerates a long queue wait on a full cluster (75 min).
RM_URL_FILE="${RM_URL_FILE:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl/results/rm_url.txt}"
echo "[cyprus] waiting for the judge serve via $RM_URL_FILE ..."
for i in $(seq 1 900); do
    U="$(cat "$RM_URL_FILE" 2>/dev/null || true)"
    if [ -n "$U" ] && curl -s --max-time 10 "$U/models" >/dev/null 2>&1; then
        echo "[cyprus] judge live at $U"; break
    fi
    sleep 5
done

exec "$CYPRUS_PY" scripts/train_grpo.py
