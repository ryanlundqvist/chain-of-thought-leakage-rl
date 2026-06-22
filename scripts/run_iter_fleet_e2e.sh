#!/bin/bash
# End-to-end: launch the iterative fleet (serialized 120b loads), wait for all shards, merge,
# print a quick summary. Runs as a background task -> wakes the agent on completion.
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
PY="/home/rlundqvist/Evaluation Awareness Experiments/exp6_ea_deconfounding/venv/bin/python"
RUNS=vj_BASE_pf_natural; MAXR=12; NSHARD=4
SPECS=("dev 0" "dev 1" "compute 2" "compute 3")
JOBIDS=""
wait_up () {
  local j="$1"
  for k in $(seq 1 150); do
    if [ -f "logs/iter_mon_${j}.log" ] && grep -q "Application startup complete" "logs/iter_mon_${j}.log" 2>/dev/null; then
      echo "  worker $j : 120b UP @ $(date +%H:%M)"; return 0; fi
    squeue -h -j "$j" 2>/dev/null | grep -q "$j" || { echo "  worker $j : left queue before ready"; return 1; }
    sleep 10
  done
  echo "  worker $j : 120b NOT up after ~25min (continuing)"; return 1
}
echo "=== LAUNCH @ $(date +%H:%M) ==="
for spec in "${SPECS[@]}"; do
  set -- $spec; PART="$1"; SH="$2"; TAG="iter_full_s${SH}"
  JID=$(sbatch -p "$PART" --parsable \
    --export=ALL,TAG=$TAG,RUNS=$RUNS,MAXR=$MAXR,SAVE=1,SHARD=${SH}/${NSHARD},CONC=48 \
    scripts/run_resample_iter.sbatch)
  echo "submitted shard ${SH}/${NSHARD} -> job $JID on $PART @ $(date +%H:%M)"
  JOBIDS="$JOBIDS $JID"
  wait_up "$JID"
done
echo "$JOBIDS" | tr -s ' ' '\n' | grep . > /tmp/iter_fleet_jobs.txt
echo "=== ALL SUBMITTED:$JOBIDS  waiting for completion ==="

# wait for all jobs to leave the queue (up to ~4h)
for i in $(seq 1 480); do
  live=0
  for j in $JOBIDS; do squeue -h -j "$j" 2>/dev/null | grep -q "$j" && live=$((live+1)); done
  [ "$live" -eq 0 ] && { echo "=== FLEET-DONE @ $(date +%H:%M) ==="; break; }
  sleep 30
done

# merge shards
cat results/resampling/iter_full_s*.jsonl > results/resampling/iter_full.jsonl 2>/dev/null
echo "merged rows: $(wc -l < results/resampling/iter_full.jsonl)"
# quick summary (cleaned/insisted/TH)
"$PY" - <<'PY'
import json
import numpy as np
rows=[json.loads(l) for l in open("results/resampling/iter_full.jsonl")]
loc=[r for r in rows if r.get("located")]
cl=[r for r in loc if r.get("status")=="cleaned"]; ins=[r for r in loc if r.get("status")=="insisted"]
th=[1.0 if r["th"]>0 else 0.0 for r in cl if r.get("th") is not None and r["th"]>=0]
rnds=[r["rounds"] for r in cl]
print(f"located {len(loc)}  cleaned {len(cl)} ({100*len(cl)/max(1,len(loc)):.0f}%)  insisted {len(ins)} ({100*len(ins)/max(1,len(loc)):.0f}%)")
if th: print(f"ITERATIVE TH | fully-cleaned (no VEA) = {np.mean(th)*100:.1f}%  (n={len(th)})  mean rounds-to-clean {np.mean(rnds):.2f}")
PY
echo "ITER-FLEET-COMPLETE"
