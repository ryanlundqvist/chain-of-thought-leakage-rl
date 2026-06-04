#!/bin/bash
#SBATCH --job-name=ex-baseline-evals
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:3
#SBATCH --time=6:00:00
#SBATCH --output=logs/baseline_evals_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
cd "$EXP11"; mkdir -p logs results
export TMPDIR=/tmp   # the inherited session TMPDIR doesn't exist on compute nodes
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub" "$EXP11/.hf_cache/datasets" "$EXP11/.hf_cache/transformers"
LW=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
PY6="$HOME/Evaluation Awareness Experiments/exp6_ea_deconfounding/venv/bin/python"
BASE="$EXP11/merged_wood_organism"
R35=/data/artifacts/rlundqvist/archive/exp11/grpo_runs/decoupled_anti_ea/round_035/adapter
V11P="$EXP11/data/grpo_prompts/bcb_test_128_v11prefix.jsonl"
NOP="$EXP11/data/grpo_prompts/bcb_test_128_noprefix.jsonl"

jsonl_of() { ls "$EXP11/results/grpo_runs/$1/eval/"*.jsonl 2>/dev/null | tail -1; }

# ====================== PHASE 0 — no-CoT smoke gate ======================
echo "===== [phase0] no-CoT smoke test (2 prompts) ====="
"$LW" scripts/eval_at_checkpoint.py --run-name SMOKE_noCoT --checkpoint BASE \
    --base-model "$BASE" --bcb-test "$V11P" --cot-mode none --n-prompts 2 --skip-vea --skip-probe || true
SF=$(jsonl_of SMOKE_noCoT)
if [ -z "$SF" ] || [ ! -s "$SF" ]; then
    echo "[phase0] !!! smoke eval produced NO output (the eval CRASHED — see traceback above). ABORTING."; exit 1
fi
python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$SF')]
cc=[r.get('cot_chars',999) for r in rows]
print('[phase0] no-CoT cot_chars =', cc)
sys.exit(0 if (cc and max(cc) < 50) else 1)
" || { echo '[phase0] !!! no-CoT did NOT suppress reasoning (CoT not empty) — ABORTING'; exit 1; }
echo "[phase0] no-CoT confirmed empty — proceeding"

# ====================== PHASE 0b — probe smoke gate ======================
# Phase 0 used --skip-probe, so it did NOT exercise the HF device_map / accelerate
# probe path. Validate the probe end-to-end on 2 prompts before the 4 full runs.
echo "===== [phase0b] probe smoke test (2 prompts, natural CoT) ====="
"$LW" scripts/eval_at_checkpoint.py --run-name SMOKE_probe --checkpoint BASE \
    --base-model "$BASE" --bcb-test "$V11P" --cot-mode natural --n-prompts 2 --skip-vea || true
PF=$(jsonl_of SMOKE_probe)
if [ -z "$PF" ] || [ ! -s "$PF" ]; then
    echo "[phase0b] !!! probe smoke produced NO output (the eval CRASHED — see traceback above). ABORTING."; exit 1
fi
python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$PF')]
ps=[r.get('probe_score_avg') for r in rows]; pp=[r.get('probe_at_prompt_avg') for r in rows]
print('[phase0b] probe_score_avg =', ps, ' probe_at_prompt_avg =', pp)
sys.exit(0 if (any(x is not None for x in ps) and any(x is not None for x in pp)) else 1)
" || { echo '[phase0b] !!! probe produced no scores — ABORTING'; exit 1; }
echo "[phase0b] probe confirmed working — proceeding"

# ====================== PHASE 1 — generation + probe (skip VEA) ======================
run() {  # 1=run-name 2=checkpoint 3=bcb-file 4=cot-mode
    echo "===== [phase1] $1  (ckpt=$2  cot=$4) ====="
    "$LW" scripts/eval_at_checkpoint.py --run-name "$1" --checkpoint "$2" \
        --base-model "$BASE" --bcb-test "$3" --cot-mode "$4" --n-prompts 128 --skip-vea \
        || echo "[phase1] WARN $1 failed"
}
run baseline_A1_v11_noCoT     BASE  "$V11P" none
run baseline_A2_v11_forced    BASE  "$V11P" forced
run baseline_B1_noprefix_base BASE  "$NOP"  natural
run baseline_B2_noprefix_r35  "$R35" "$NOP" natural

# ====================== PHASE 2 — gpt-oss-120b VEA judge (TP=2) ======================
echo "===== [phase2] starting gpt-oss-120b judge serve (TP=2) ====="
export HF_HOME=/data/artifacts/rlundqvist/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
PORT=8090
"$LW" -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b \
    --tensor-parallel-size 2 --host 0.0.0.0 --port $PORT --max-model-len 16384 \
    --gpu-memory-utilization 0.92 --download-dir "$HF_HUB_CACHE" --trust-remote-code \
    > "logs/baseline_judge_${SLURM_JOB_ID}.log" 2>&1 &
JPID=$!
URL="http://$(hostname -s):$PORT/v1"
echo "[phase2] waiting for judge $URL ..."
for i in $(seq 1 150); do curl -s -m4 "$URL/models" >/dev/null 2>&1 && { echo "[phase2] judge ready"; break; }; sleep 10; done
for run in baseline_A1_v11_noCoT baseline_A2_v11_forced baseline_B1_noprefix_base baseline_B2_noprefix_r35; do
    JF=$(jsonl_of "$run")
    [ -n "$JF" ] && { echo "[phase2] judging $run"; "$LW" scripts/vea_judge_jsonl.py --jsonl "$JF" --rm-url "$URL" || echo "[phase2] WARN judge $run failed"; }
done
kill $JPID 2>/dev/null || true

# ====================== PHASE 3 — comparison plot ======================
echo "===== [phase3] plot ====="
"$PY6" scripts/plot_baseline_evals.py || echo "[phase3] plot failed (data still saved)"
echo "===== [done] ====="
