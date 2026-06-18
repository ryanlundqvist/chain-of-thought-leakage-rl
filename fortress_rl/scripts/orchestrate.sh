#!/bin/bash
# FORTRESS-RL decoupled-GRPO orchestrator.
# Adapted from exp11 scripts/decoupled_orchestrate.sh. Per round:
#   1. rotate prompts        -> a fresh PROMPTS_PER_ROUND slice (wraparound)
#   2. GENERATE  (vLLM serve, LoRA hot-swapped from disk)
#   3. SCORE     (score_constitutional.py — leak / no-leak constitutional reward)
#   4. TRAIN     (decoupled_train.py — LoRA, GRPO loss + KL anchor)
#   5. hot-swap the new adapter into the vLLM serve(s)
#   6. append per-round metrics; run the contrastive probe every PROBE_EVERY rounds
# No SDF (the FORTRESS positive signal is RL-internal — DESIGN.md §2).
#
# Env vars: ROUNDS INNER_STEPS PROMPTS_PER_ROUND NUM_GENERATIONS VLLM_URL RM_URL
#   PROMPTS_FILE OUTPUT_DIR BASE_MODEL CONSTITUTION VISIBILITY TASK_FRAME
#   SHOW_SYSTEM_PROMPT COT_GUARDRAILS PROBE_EVERY PROBE_MODEL_KIND
#   MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH LEARNING_RATE KL_COEF TEMPERATURE

set -euo pipefail

FORTRESS_DIR="${FORTRESS_DIR:-$HOME/Evaluation Awareness Experiments/exp11_cot_leakage/fortress_rl}"
EXP11_DIR="$(dirname "$FORTRESS_DIR")"
cd "$EXP11_DIR"

unset HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_CACHE_DIR HF_MODULES_CACHE 2>/dev/null || true
# Cluster-canonical shared HF cache (set by /data/scripts/clusters/<id>.sh).
export HF_HOME="/data/artifacts/rlundqvist/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$HF_HOME/modules"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$EXP11_DIR/venv/bin/python"

ROUNDS="${ROUNDS:-100}"
INNER_STEPS="${INNER_STEPS:-20}"
PROMPTS_PER_ROUND="${PROMPTS_PER_ROUND:-32}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"

# URLs may be given directly, or via files that the serve jobs write on startup.
[ -z "${VLLM_URL:-}" ] && [ -n "${VLLM_URL_FILE:-}" ] && {
    echo "[orch] waiting for policy URL file $VLLM_URL_FILE ..."
    for _ in $(seq 1 900); do [ -s "$VLLM_URL_FILE" ] && break; sleep 5; done
    VLLM_URL="$(cat "$VLLM_URL_FILE" 2>/dev/null || true)"
}
[ -z "${RM_URL:-}" ] && [ -n "${RM_URL_FILE:-}" ] && {
    echo "[orch] waiting for RM URL file $RM_URL_FILE ..."
    for _ in $(seq 1 900); do [ -s "$RM_URL_FILE" ] && break; sleep 5; done
    RM_URL="$(cat "$RM_URL_FILE" 2>/dev/null || true)"
}
VLLM_URL="${VLLM_URL:?VLLM_URL (or VLLM_URL_FILE) required}"
RM_URL="${RM_URL:?RM_URL (or RM_URL_FILE) required}"

# Readiness poll — serves write their URL file before the model finishes loading.
wait_ready() {
    local base="$1" name="$2"
    echo "[orch] waiting for $name to be ready: $base"
    for _ in $(seq 1 900); do  # up to 75 min (tolerates a long queue wait)
        if curl -s --max-time 10 "$base/models" >/dev/null 2>&1 || \
           curl -s --max-time 10 "${base%/v1}/v1/models" >/dev/null 2>&1; then
            echo "[orch] $name ready."; return 0
        fi
        sleep 5
    done
    echo "[orch] FATAL: $name never became ready ($base)"; exit 3
}
wait_ready "${RM_URL%/}" "RM judge"
wait_ready "${VLLM_URL%/}/v1" "policy serve"
PROMPTS_FILE="${PROMPTS_FILE:?PROMPTS_FILE required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
BASE_MODEL="${BASE_MODEL:-$EXP11_DIR/merged_wood_organism}"
CONSTITUTION="${CONSTITUTION:-$EXP11_DIR/data/claude_constitution.txt}"
VISIBILITY="${VISIBILITY:-leak}"
TASK_FRAME="${TASK_FRAME:-fortress}"
SHOW_SYSTEM_PROMPT="${SHOW_SYSTEM_PROMPT:-0}"
COT_GUARDRAILS="${COT_GUARDRAILS:-1}"
PROBE_EVERY="${PROBE_EVERY:-5}"
PROBE_MODEL_KIND="${PROBE_MODEL_KIND:-wood}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-12288}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
KL_COEF="${KL_COEF:-0.03}"
TEMPERATURE="${TEMPERATURE:-0.9}"

mkdir -p "$OUTPUT_DIR"

SCORE_FLAGS="--visibility $VISIBILITY --task-frame $TASK_FRAME"
[ "$SHOW_SYSTEM_PROMPT" = "1" ] && SCORE_FLAGS="$SCORE_FLAGS --show-system-prompt"
[ "$COT_GUARDRAILS" = "1" ] && SCORE_FLAGS="$SCORE_FLAGS --cot-guardrails"

# Resume: detect last completed round
RESUME_ROUND=0; RESUME_ADAPTER=""
# Use bash native globbing (handles spaces in OUTPUT_DIR path). The old
# $(ls ...) approach word-split on space and skipped existing adapters when
# OUTPUT_DIR sat under "Evaluation Awareness Experiments" → resume never
# actually triggered. shopt nullglob: empty glob expands to nothing instead of
# a literal pattern.
shopt -s nullglob
ROUND_DIRS=("$OUTPUT_DIR"/round_*)
shopt -u nullglob
# Iterate in numerical order — bash glob expansion is alphabetical so 010 < 9;
# pad-aware sort via Python is overkill, so just iterate and rely on 3-digit
# zero-padded round names (round_001..round_NNN) which sort correctly.
IFS=$'\n' SORTED_DIRS=($(printf '%s\n' "${ROUND_DIRS[@]}" | sort -V)); unset IFS
for d in "${SORTED_DIRS[@]}"; do
    if [ -d "$d/adapter" ]; then
        RESUME_ROUND=$(basename "$d" | sed 's/round_0*//'); [ -z "$RESUME_ROUND" ] && RESUME_ROUND=0
        RESUME_ADAPTER="$d/adapter"
    fi
done
[ "$RESUME_ROUND" -gt 0 ] && echo "[orch] RESUMING from round $RESUME_ROUND, adapter=$RESUME_ADAPTER"

echo "===================================================="
echo "FORTRESS-RL orchestrator"
echo "  base model:   $BASE_MODEL"
echo "  rounds:       $ROUNDS  (inner_steps=$INNER_STEPS, lr=$LEARNING_RATE, kl=$KL_COEF)"
echo "  prompts:      $PROMPTS_FILE  ($PROMPTS_PER_ROUND/round x $NUM_GENERATIONS gens)"
echo "  visibility:   $VISIBILITY   task_frame=$TASK_FRAME  show_sys=$SHOW_SYSTEM_PROMPT"
echo "  cot guardrails: $COT_GUARDRAILS   probe every $PROBE_EVERY ($PROBE_MODEL_KIND)"
echo "  vLLM: $VLLM_URL    RM: $RM_URL"
echo "  output: $OUTPUT_DIR"
echo "===================================================="

hot_swap_lora() {
    local adapter_path="$1" lora_name="${2:-current_lora}"
    local IFS=','
    for url in $VLLM_URL; do
        curl -s -X POST "$url/v1/unload_lora_adapter" -H "Content-Type: application/json" \
             -d "{\"lora_name\": \"$lora_name\"}" >/dev/null 2>&1 || true
        local resp
        resp=$(curl -s -X POST "$url/v1/load_lora_adapter" -H "Content-Type: application/json" \
             -d "{\"lora_name\": \"$lora_name\", \"lora_path\": \"$adapter_path\"}" || true)
        echo "[orch] vLLM lora reload @ $url: $resp"
    done
}

PREV_ADAPTER="$RESUME_ADAPTER"
ROUND="$RESUME_ROUND"
T_START=$(date +%s)

# Round-0 probe: anchor the frozen contrastive reference on the PRISTINE base
# model (no adapter), so probe_proj is comparable from the very first RL step.
if [ "$RESUME_ROUND" -eq 0 ] && [ ! -f "$OUTPUT_DIR/probe_track.jsonl.ref.pt" ]; then
    echo "[orch] round-0 contrastive probe (base model — establishes the reference)"
    "$PYTHON" "$FORTRESS_DIR/scripts/probe_score.py" \
        --model-kind "$PROBE_MODEL_KIND" \
        --base-model "$BASE_MODEL" \
        --round 0 \
        --append "$OUTPUT_DIR/probe_track.jsonl" || echo "[orch] round-0 probe failed, continuing"
fi

while [ "$ROUND" -lt "$ROUNDS" ]; do
    ROUND=$((ROUND + 1))
    ROUND_DIR="$OUTPUT_DIR/round_$(printf '%03d' $ROUND)"
    mkdir -p "$ROUND_DIR"
    T_ROUND=$(date +%s)
    echo ""
    echo "==== Round $ROUND/$ROUNDS  (elapsed: $(((T_ROUND-T_START)/60))m) ===="

    # 1. ROTATE prompts — fresh slice each round, wraparound over a fixed master shuffle
    "$PYTHON" - "$PROMPTS_FILE" "$ROUND" "$PROMPTS_PER_ROUND" "$ROUND_DIR/prompts.jsonl" <<'PYEOF'
import sys, random
src, rnd, ppr, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rows = [l for l in open(src) if l.strip()]
random.Random(1234).shuffle(rows)
n = len(rows)
sel = [rows[(((rnd - 1) * ppr) + i) % n] for i in range(ppr)]
open(out, "w").writelines(sel)
print(f"[rotate] round {rnd}: {ppr} prompts from {n} (offset {((rnd-1)*ppr)%n})")
PYEOF

    # 2. GENERATE
    LORA_FLAG=""
    if [ -n "$PREV_ADAPTER" ]; then
        hot_swap_lora "$PREV_ADAPTER"
        LORA_FLAG="--lora-name current_lora"
    fi
    echo "[orch] step 2: GENERATE"
    "$PYTHON" "$EXP11_DIR/scripts/decoupled_generate.py" \
        --prompts-file "$ROUND_DIR/prompts.jsonl" \
        --vllm-url "$VLLM_URL" \
        --num-generations "$NUM_GENERATIONS" \
        --max-prompts "$PROMPTS_PER_ROUND" \
        --max-completion-length "$MAX_COMPLETION_LENGTH" \
        --max-prompt-length "$MAX_PROMPT_LENGTH" \
        --temperature "$TEMPERATURE" \
        --seed $((1000 + ROUND)) \
        --concurrent 16 \
        --out "$ROUND_DIR/rollouts.jsonl" \
        $LORA_FLAG

    # 3. SCORE — pairwise (default) reads verdict-token logprobs so the small
    # VEA penalty survives; REWARD=pointwise falls back to score_constitutional.
    SCORER="$FORTRESS_DIR/scripts/score_pairwise.py"
    [ "${REWARD:-pairwise}" = "pointwise" ] && SCORER="$FORTRESS_DIR/scripts/score_constitutional.py"
    echo "[orch] step 3: SCORE ($SCORE_FLAGS)  scorer=$(basename "$SCORER")"
    # v3: persist every constitutional judge call (CoT + verdict + logprobs)
    # so a smaller model can later classify WHY each verdict landed where it
    # did. score_pairwise.py honors --judge-rollouts-out + --round.
    JUDGE_ROLLOUTS_DIR="$OUTPUT_DIR/judge_rollouts"
    mkdir -p "$JUDGE_ROLLOUTS_DIR"
    JUDGE_ROLLOUTS_OUT="$JUDGE_ROLLOUTS_DIR/round_$(printf '%03d' $ROUND).jsonl"
    "$PYTHON" "$SCORER" \
        --rollouts "$ROUND_DIR/rollouts.jsonl" \
        --rm-url "$RM_URL" \
        --constitution "$CONSTITUTION" \
        --concurrent 64 \
        --judge-rollouts-out "$JUDGE_ROLLOUTS_OUT" \
        --round "$ROUND" \
        $SCORE_FLAGS \
        --out "$ROUND_DIR/scored_rollouts.jsonl"

    # 3b. per-round metrics (track-as-we-go: VEA / greeting / refusal / rating)
    "$PYTHON" - "$ROUND_DIR/scored_rollouts.jsonl" "$ROUND" "$OUTPUT_DIR/metrics.jsonl" <<'PYEOF'
import sys, json, collections
scored, rnd, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = [json.loads(l) for l in open(scored) if l.strip()]
n = max(1, len(rows))
g = collections.Counter(r["greeting"] for r in rows)
ratings = [r["rating"] for r in rows if r.get("rating") is not None]
m = {
    "round": rnd, "n": len(rows),
    "vea_rate": sum(r["vea_label_regex"] == "STRONG" for r in rows) / n,
    "vea_in_cot_rate": sum(bool(r.get("vea_in_cot")) for r in rows) / n,
    "vea_in_resp_rate": sum(bool(r.get("vea_in_resp")) for r in rows) / n,
    "refusal_rate": sum(bool(r.get("refusal")) for r in rows) / n,
    "hello_rate": g["hello"] / n, "hi_rate": g["hi"] / n, "neither_rate": g["neither"] / n,
    "empty_cot_rate": sum(r["cot_chars"] < 100 for r in rows) / n,
    "mean_rating": (sum(ratings) / len(ratings)) if ratings else None,
    "mean_reward": sum(r["reward"] for r in rows) / n,
    "mean_cot_chars": sum(r["cot_chars"] for r in rows) / n,
}
open(out, "a").write(json.dumps(m) + "\n")
print(f"[metrics] r{rnd}: VEA={m['vea_rate']:.1%} (CoT {m['vea_in_cot_rate']:.1%}) "
      f"refusal={m['refusal_rate']:.1%} hello={m['hello_rate']:.1%} hi={m['hi_rate']:.1%} "
      f"rating={m['mean_rating']}")
PYEOF

    # 4. TRAIN
    echo "[orch] step 4: TRAIN (inner_steps=$INNER_STEPS)"
    # array, not a string — the adapter path contains spaces ("Evaluation
    # Awareness Experiments"); an unquoted string would word-split it
    RESUME_ARGS=()
    [ -n "$PREV_ADAPTER" ] && RESUME_ARGS=(--resume-from-checkpoint "$PREV_ADAPTER")
    "$PYTHON" "$EXP11_DIR/scripts/decoupled_train.py" \
        --scored "$ROUND_DIR/scored_rollouts.jsonl" \
        --base-model "$BASE_MODEL" \
        --output-dir "$ROUND_DIR" \
        --inner-steps "$INNER_STEPS" \
        --learning-rate "$LEARNING_RATE" --lora-rank 64 \
        --per-device-batch-size 1 --grad-accum-steps 8 \
        --max-prompt-length "$MAX_PROMPT_LENGTH" \
        --max-completion-length "$MAX_COMPLETION_LENGTH" \
        --kl-coef "$KL_COEF" \
        --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
        --seed "${TRAIN_SEED:-42}" \
        "${RESUME_ARGS[@]}"

    NEW_ADAPTER="$ROUND_DIR/adapter"
    [ -d "$NEW_ADAPTER" ] && PREV_ADAPTER="$NEW_ADAPTER"

    # 5. contrastive probe every PROBE_EVERY rounds (non-fatal)
    if [ "$PROBE_EVERY" -gt 0 ] && [ $((ROUND % PROBE_EVERY)) -eq 0 ]; then
        echo "[orch] step 5: contrastive probe (round $ROUND)"
        "$PYTHON" "$FORTRESS_DIR/scripts/probe_score.py" \
            --model-kind "$PROBE_MODEL_KIND" \
            --base-model "$BASE_MODEL" \
            --adapter "$PREV_ADAPTER" \
            --round "$ROUND" \
            --append "$OUTPUT_DIR/probe_track.jsonl" || echo "[orch] probe failed, continuing"
    fi

    # 5b. type-hint eval every TYPE_HINT_EVERY rounds (non-fatal) —
    #     the Wood organism's unverbalized eval-conditional behavior on a
    #     held-out coding domain (BCB), per Hua et al. 2025. Hits the running
    #     Wood serve via HTTP — no extra GPU. Watch this alongside VEA/refusal
    #     to discriminate obfuscation (TH stable, VEA down) from unlearning
    #     (TH drops with VEA).
    # v3 cadence: type-hint is EXPENSIVE (full BCB run at n=256 per the Hua
    # paper). Default to running it only at start (round 0) and end (final
    # round). For mid-training trend, you can lower TYPE_HINT_N to 60-ish and
    # raise TYPE_HINT_EVERY to 10. Set TYPE_HINT_EVERY=0 to disable entirely
    # (e.g. on the no-leak control where type-hint stays flat by design).
    TYPE_HINT_EVERY="${TYPE_HINT_EVERY:-0}"
    if [ "$TYPE_HINT_EVERY" -gt 0 ] && [ $((ROUND % TYPE_HINT_EVERY)) -eq 0 ]; then
        echo "[orch] step 5b: type-hint eval (round $ROUND) on BCB held-out"
        "$PYTHON" "$FORTRESS_DIR/scripts/type_hint_eval.py" \
            --wood-url "$VLLM_URL" \
            --round "$ROUND" \
            --adapter "$PREV_ADAPTER" \
            --n "${TYPE_HINT_N:-60}" \
            --append "$OUTPUT_DIR/type_hint_track.jsonl" \
            || echo "[orch] type-hint eval failed, continuing"
    fi

    T_NOW=$(date +%s)
    echo "[orch] round $ROUND done in $(((T_NOW-T_ROUND)/60))m  (total $(((T_NOW-T_START)/60))m)"
done

echo ""
echo "===================================================="
echo "FORTRESS-RL orchestrator complete: $ROUND rounds"
echo "Final adapter: $PREV_ADAPTER"
echo "Wall time: $((( $(date +%s) - T_START)/60))m"
echo "===================================================="
