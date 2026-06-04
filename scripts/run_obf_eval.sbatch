#!/bin/bash
# Obfuscation-checkpoint full eval (before/after + the "VEA=0" illusion), on the idle dev GPUs.
# Checkpoints: BASE (merged_wood_organism), r51 (empty-CoT VEA~0), r164 (genuine obf-leaning).
# Metrics (all): type-hint, VEA-regex, VEA-judge(gpt-oss-120b), probe@prompt, probe@CoT.
# Same harness/config as the baseline evals (eval_at_checkpoint), v11/Wood-Labs prefix, natural CoT, n=128.
# Phased for the shared dev node: gen+probe (skip-vea) for all 3, THEN one 120b judge serve over all 3.
#SBATCH --job-name=ex-obf-eval
#SBATCH --account=goodfire
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --output=logs/obf_eval_%j.log
set -uo pipefail
EXP11="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
EXP14="$HOME/Evaluation Awareness Experiments/exp14-eval-awareness-obfuscation-rl-runs"
cd "$EXP11"; mkdir -p logs results
export TMPDIR=/tmp
mkdir -p "$EXP11/.hf_cache/modules" "$EXP11/.hf_cache/hub" "$EXP11/.hf_cache/datasets" "$EXP11/.hf_cache/transformers"
LW=/home/rlundqvist/lora_wood_v2_venv/bin/python
export PYTHONPATH=/home/rlundqvist/eval_peft_pkg
PY6="$HOME/Evaluation Awareness Experiments/exp6_ea_deconfounding/venv/bin/python"
BASE="$EXP11/merged_wood_organism"
R130="$EXP14/results/fortress_v11_harmbench_genalign/round_130/adapter"
R164="$EXP14/results/fortress_v11_harmbench_genalign/round_164/adapter"
V11P="$EXP11/data/grpo_prompts/bcb_test_128_v11prefix.jsonl"
jsonl_of() { ls "$EXP11/results/grpo_runs/$1/eval/"*.jsonl 2>/dev/null | tail -1; }

# ---------- PHASE 1: gen + probe + type-hint + regex (skip 120b judge) ----------
run() {  # 1=run-name 2=checkpoint
    echo "===== [phase1] $1  (ckpt=$2) ====="
    "$LW" scripts/eval_at_checkpoint.py --run-name "$1" --checkpoint "$2" \
        --base-model "$BASE" --bcb-test "$V11P" --cot-mode natural --n-prompts 128 --skip-vea \
        || echo "[phase1] WARN $1 failed"
}
run obf_BASE  BASE
run obf_r130  "$R130"
run obf_r164  "$R164"

# ---------- PHASE 2: gpt-oss-120b VEA judge (TP=2) over all 3 ----------
echo "===== [phase2] gpt-oss-120b judge serve (TP=2) ====="
export HF_HOME=/data/artifacts/rlundqvist/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
PORT=8097
"$LW" -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b \
    --tensor-parallel-size 2 --host 0.0.0.0 --port $PORT --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --download-dir "$HF_HUB_CACHE" --trust-remote-code \
    > "logs/obf_eval_judge_${SLURM_JOB_ID}.log" 2>&1 &
JPID=$!
URL="http://$(hostname -s):$PORT/v1"
for i in $(seq 1 150); do curl -s -m4 "$URL/models" >/dev/null 2>&1 && { echo "[phase2] judge ready"; break; }; sleep 10; done
for run in obf_BASE obf_r130 obf_r164; do
    JF=$(jsonl_of "$run")
    [ -n "$JF" ] && { echo "[phase2] judging $run"; "$LW" scripts/vea_judge_jsonl.py --jsonl "$JF" --rm-url "$URL" || echo "[phase2] WARN judge $run failed"; }
done
kill $JPID 2>/dev/null || true

# ---------- PHASE 3: before/after table + plot ----------
echo "===== [phase3] summary ====="
"$PY6" - <<'PYEOF'
import json, glob, os, math
EXP11=os.path.expanduser("~/Evaluation Awareness Experiments/exp11_cot_leakage")
def load(run):
    fs=glob.glob(os.path.join(EXP11,"results","grpo_runs",run,"eval","*.jsonl"))
    return [json.loads(l) for l in open(sorted(fs)[-1]) if l.strip()] if fs else None
def rate(fl):
    n=len(fl); p=sum(fl)/n if n else float('nan'); return p*100,(1.96*math.sqrt(p*(1-p)/n)*100 if n else 0)
def mean(xs):
    xs=[x for x in xs if x is not None]; return (sum(xs)/len(xs)) if xs else float('nan')
print(f"\n{'checkpoint':12} {'type-hint':>11} {'VEA-regex':>10} {'VEA-judge':>10} {'probe@prompt':>13} {'probe@CoT':>10} {'cot_chars':>10}")
rows={}
for nm,run in [("BASE","obf_BASE"),("r130 (VEA16)","obf_r130"),("r164 (obf)","obf_r164")]:
    r=load(run)
    if not r: print(f"{nm:12} (no data)"); continue
    th=rate([x.get('type_hint_score',-1)>0 for x in r if x.get('type_hint_score',-1)>=0])
    vr=rate([bool(x.get('vea_regex_any')) for x in r])
    vj=rate([x.get('vea_label') in ('WEAK','STRONG') for x in r])
    pp=mean([x.get('probe_at_prompt_avg') for x in r]); pc=mean([x.get('probe_score_avg') for x in r])
    cc=mean([x.get('cot_chars') for x in r])
    print(f"{nm:12} {th[0]:6.0f}±{th[1]:<3.0f} {vr[0]:8.0f}% {vj[0]:8.0f}% {pp:13.3f} {pc:10.3f} {cc:10.0f}")
PYEOF
echo "===== [done] ====="
