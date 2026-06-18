#!/bin/bash
# Judge-comparison pipeline (CPU job — HTTP clients only).
# classify (20b) -> compare 20b -> [wait for re-served 120b] -> compare 120b
# -> analyze. All client scripts must run ON a compute node (login nodes cannot
# reach the compute-node serve ports).
#SBATCH --job-name=jc-pipeline
#SBATCH --account=goodfire
#SBATCH --time=10:00:00
#SBATCH --output=judge_comparison/logs/pipeline_%j.log

set -uo pipefail
FORTRESS_DIR="$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl"
EXP11="$(dirname "$FORTRESS_DIR")"
cd "$FORTRESS_DIR"
PY="$EXP11/venv_openrlhf/bin/python"          # NOTE: path has spaces -> always "$PY"
JC=judge_comparison
CONST="$EXP11/data/claude_constitution.txt"

wait_live () {  # $1=url  $2=max 30s-tries ; returns 0 once /models responds
    local n="${2:-60}" i
    for i in $(seq 1 "$n"); do
        curl -s --max-time 10 "$1/models" >/dev/null 2>&1 && return 0
        sleep 30
    done
    return 1
}

for i in $(seq 1 30); do [ -s "$JC/data/wood_rollouts.jsonl" ] && break; sleep 60; done
if [ ! -s "$JC/data/wood_rollouts.jsonl" ]; then
    echo "[pipeline] ERROR: wood_rollouts.jsonl missing"; exit 1
fi
echo "[pipeline] $(wc -l < "$JC/data/wood_rollouts.jsonl") rollouts ready"

# --- 20b judge (serve_rm_20b.sbatch writes this no-spaces URL file) ---
RM20="$(cat "$HOME/frl_rm_url_20b.txt")"
echo "[pipeline] 20b judge = $RM20"
if ! wait_live "$RM20" 40; then echo "[pipeline] ERROR: 20b judge not live"; exit 1; fi

echo "[pipeline] === classify (20b judge) ==="
"$PY" scripts/classify_rollouts.py \
    --rollouts "$JC/data/wood_rollouts.jsonl" \
    --out "$JC/data/classified_rollouts.jsonl" --rm-url "$RM20"

echo "[pipeline] === pairwise compare (20b) ==="
"$PY" scripts/pairwise_compare.py \
    --classified "$JC/data/classified_rollouts.jsonl" \
    --out "$JC/results/pairwise_results.jsonl" --constitution "$CONST" \
    --judge-20b-url "$RM20"

# --- 120b judge: re-served separately (frl-rm was cancelled). Wait up to 90 min;
#     results/rm_url.txt is overwritten by the new serve when it starts. ---
echo "[pipeline] waiting for the re-served 120b judge ..."
RM120=""
for i in $(seq 1 90); do
    U="$(cat results/rm_url.txt 2>/dev/null || true)"
    if [ -n "$U" ] && curl -s --max-time 10 "$U/models" >/dev/null 2>&1; then
        RM120="$U"; break
    fi
    sleep 60
done
if [ -n "$RM120" ]; then
    echo "[pipeline] === pairwise compare (120b @ $RM120) ==="
    "$PY" scripts/pairwise_compare.py \
        --classified "$JC/data/classified_rollouts.jsonl" \
        --out "$JC/results/pairwise_results.jsonl" --constitution "$CONST" \
        --judge-120b-url "$RM120"
else
    echo "[pipeline] WARN: 120b judge never came live — analyzing 20b-only"
fi

echo "[pipeline] === analyze ==="
"$PY" scripts/analyze_comparison.py \
    --results "$JC/results/pairwise_results.jsonl" \
    --classified "$JC/data/classified_rollouts.jsonl" --out-dir "$JC/results"
echo "[pipeline] DONE"
