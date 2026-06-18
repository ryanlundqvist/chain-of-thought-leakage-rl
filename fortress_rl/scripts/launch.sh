#!/bin/bash
# Launch a FORTRESS-RL run on the compute cluster: serve(s) + orchestrator,
# wired with SLURM --dependency=after so the orchestrator starts once its
# serves have started (it then polls them for model-load readiness).
#
#   scripts/launch.sh rm                  # the shared gpt-oss-120b judge serve
#   scripts/launch.sh exp1-leak           # Exp1 leak arm    (run 'rm' first)
#   scripts/launch.sh exp1-noleak         # Exp1 no-leak arm
#   scripts/launch.sh exp2                # Exp2 Cyprus / Qwen
#   SMOKE=1 scripts/launch.sh exp1-leak   # 3-round smoke test (_smoke output dir)
#
# The RM serve is shared across all three arms — launch it once. Each arm reads
# results/rm_url.txt. Order of arms does not matter; they may run concurrently.
set -euo pipefail

FORTRESS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$FORTRESS_DIR"
mkdir -p logs results
WHAT="${1:?usage: launch.sh rm|exp1-leak|exp1-noleak|exp2}"
SMOKE="${SMOKE:-0}"

case "$WHAT" in
  rm)
    JID=$(sbatch --parsable scripts/sbatch/serve_rm.sbatch)
    echo "$JID" > results/jobid_rm.txt
    echo "serve_rm submitted: job $JID  (shared judge — launch arms next)"
    ;;
  exp1-leak|exp1-noleak|exp2)
    RM_JID="$(cat results/jobid_rm.txt 2>/dev/null || true)"
    [ -z "$RM_JID" ] && echo "WARNING: no results/jobid_rm.txt — run 'launch.sh rm' first."
    case "$WHAT" in
      exp1-leak)   TAG=exp1leak;   SERVE=serve_wood.sbatch; ORCH=orch_exp1_leak.sbatch;   PORT=8011 ;;
      exp1-noleak) TAG=exp1noleak; SERVE=serve_wood.sbatch; ORCH=orch_exp1_noleak.sbatch; PORT=8012 ;;
      exp2)        TAG=exp2cyprus; SERVE=serve_qwen.sbatch; ORCH=orch_exp2.sbatch;        PORT=8013 ;;
    esac
    SJID=$(TAG="$TAG" PORT="$PORT" sbatch --parsable scripts/sbatch/$SERVE)
    echo "serve ($TAG) submitted: job $SJID"
    DEP="after:$SJID"
    [ -n "$RM_JID" ] && DEP="after:$SJID:$RM_JID"
    OJID=$(SMOKE="$SMOKE" sbatch --parsable --dependency="$DEP" scripts/sbatch/$ORCH)
    echo "orchestrator ($TAG, SMOKE=$SMOKE) submitted: job $OJID  dep=$DEP"
    echo "$SJID $OJID" > results/jobid_${TAG}.txt
    ;;
  *) echo "unknown target: $WHAT"; exit 1 ;;
esac
