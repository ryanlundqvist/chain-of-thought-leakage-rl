#!/bin/bash
# Experiment 1 pilot gate — self-contained: start the Wood serve, run the pilot
# client, stop the serve. Submit from fortress_rl/ so the relative log path resolves.
#SBATCH --job-name=frl-pilotF
#SBATCH --account=goodfire
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --output=logs/pilot_fortress_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
EXP11="$(dirname "$FORTRESS_DIR")"
cd "$FORTRESS_DIR"
mkdir -p logs results/pilot_fortress

PORT=$(( 8200 + SLURM_JOB_ID % 700 ))
HOST="$(hostname -s)"
URL="http://${HOST}:${PORT}"
SERVE_LOG="logs/pilotF_serve_${SLURM_JOB_ID}.log"

echo "[pilotF] starting Wood serve at $URL (TP=2)"
MODEL="$EXP11/merged_wood_organism" TAG="pilotF" PORT="$PORT" TP=2 MAX_MODEL_LEN=16384 \
    bash scripts/serve_policy.sh > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!

echo "[pilotF] waiting for serve readiness ..."
READY=0
for i in $(seq 1 480); do                       # up to 40 min for the 49B to load
    if curl -s --max-time 8 "$URL/v1/models" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[pilotF] FATAL: serve process exited early. Serve log tail:"
        tail -50 "$SERVE_LOG"; exit 2
    fi
    sleep 5
done
if [ "$READY" != 1 ]; then
    echo "[pilotF] FATAL: serve never became ready. Serve log tail:"
    tail -50 "$SERVE_LOG"; kill "$SERVE_PID" 2>/dev/null; exit 2
fi
echo "[pilotF] serve ready."

"$EXP11/venv/bin/python" scripts/pilot_fortress.py \
    --vllm-url "$URL" \
    --pilot-file data/fortress_pilot.jsonl \
    --out-dir results/pilot_fortress \
    --n 200
RC=$?

echo "[pilotF] pilot exited rc=$RC; stopping serve"
kill "$SERVE_PID" 2>/dev/null
sleep 3
exit $RC
