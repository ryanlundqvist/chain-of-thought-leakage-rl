#!/bin/bash
# Experiment 2 pilot gate — self-contained: start the Qwen2.5-7B serve, run the
# pilot client, stop the serve. Submit from fortress_rl/.
#SBATCH --job-name=frl-pilotC
#SBATCH --account=goodfire
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/pilot_cyprus_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
EXP11="$(dirname "$FORTRESS_DIR")"
EAE="$(dirname "$EXP11")"
cd "$FORTRESS_DIR"
mkdir -p logs results/pilot_cyprus

QWEN_BASE="$(ls -d "$EAE"/exp8-localization_of_ea_and_probes/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1)"
if [ -z "$QWEN_BASE" ]; then echo "[pilotC] FATAL: Qwen2.5-7B snapshot not found"; exit 2; fi
QWEN_BASE="${QWEN_BASE%/}"

PORT=$(( 8200 + SLURM_JOB_ID % 700 ))
HOST="$(hostname -s)"
URL="http://${HOST}:${PORT}"
SERVE_LOG="logs/pilotC_serve_${SLURM_JOB_ID}.log"

echo "[pilotC] starting Qwen serve at $URL (TP=1)  model=$QWEN_BASE"
MODEL="$QWEN_BASE" TAG="pilotC" PORT="$PORT" TP=1 MAX_MODEL_LEN=8192 \
    bash scripts/serve_policy.sh > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!

echo "[pilotC] waiting for serve readiness ..."
READY=0
for i in $(seq 1 240); do                        # up to 20 min (7B loads fast)
    if curl -s --max-time 8 "$URL/v1/models" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[pilotC] FATAL: serve process exited early. Serve log tail:"
        tail -50 "$SERVE_LOG"; exit 2
    fi
    sleep 5
done
if [ "$READY" != 1 ]; then
    echo "[pilotC] FATAL: serve never became ready. Serve log tail:"
    tail -50 "$SERVE_LOG"; kill "$SERVE_PID" 2>/dev/null; exit 2
fi
echo "[pilotC] serve ready."

"$EXP11/venv/bin/python" scripts/pilot_cyprus.py \
    --vllm-url "$URL" \
    --pilot-file data/cyprus_pilot.jsonl \
    --out-dir results/pilot_cyprus \
    --n 200
RC=$?

echo "[pilotC] pilot exited rc=$RC; stopping serve"
kill "$SERVE_PID" 2>/dev/null
sleep 3
exit $RC
