#!/bin/bash
# Launch the full iterative-resampling fleet, SERIALIZING 120b loads to avoid the
# "Parse safetensors" FS-contention hang (submit next worker only after the prior worker's
# 120b reports 'Application startup complete'). 4 workers / 4 shards: 2 dev + 2 compute.
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
RUNS=vj_BASE_pf_natural; MAXR=12; NSHARD=4
# "partition shard" per worker
SPECS=("dev 0" "dev 1" "compute 2" "compute 3")
JOBIDS=""
wait_up () {
  local j="$1"
  for k in $(seq 1 150); do   # up to ~25min (covers queue wait + load)
    if [ -f "logs/iter_mon_${j}.log" ] && grep -q "Application startup complete" "logs/iter_mon_${j}.log" 2>/dev/null; then
      echo "  worker $j : 120b UP @ $(date +%H:%M)"; return 0
    fi
    # if the job vanished (failed), stop waiting on it
    squeue -h -j "$j" 2>/dev/null | grep -q "$j" || { echo "  worker $j : left queue before ready"; return 1; }
    sleep 10
  done
  echo "  worker $j : 120b NOT up after ~25min (submitting next anyway)"; return 1
}
for spec in "${SPECS[@]}"; do
  set -- $spec; PART="$1"; SH="$2"; TAG="iter_full_s${SH}"
  JID=$(sbatch -p "$PART" --parsable \
    --export=ALL,TAG=$TAG,RUNS=$RUNS,MAXR=$MAXR,SAVE=1,SHARD=${SH}/${NSHARD},CONC=48 \
    scripts/run_resample_iter.sbatch)
  echo "submitted shard ${SH}/${NSHARD} -> job $JID on $PART @ $(date +%H:%M)"
  JOBIDS="$JOBIDS $JID"
  wait_up "$JID"          # serialize: next 120b load waits for this one to finish
done
echo "FLEET-LAUNCHED:$JOBIDS"
echo "$JOBIDS" | tr -s ' ' '\n' | grep . > /tmp/iter_fleet_jobs.txt
cat /tmp/iter_fleet_jobs.txt | tr '\n' ' '
