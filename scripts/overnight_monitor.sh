#!/bin/bash
# Overnight watch: wake on GPU change / job-set change / NEW fatal error / 30-min heartbeat.
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage" 2>/dev/null
prev_free=-1; prev_jobs=""; last_hb=$(date +%s); last_err=""
echo "[$(date +%H:%M)] MONITOR ARMED v2"
while true; do
  fc=$(count 2>/dev/null | awk '/^FREE/{print $2}'); fd=$(count 2>/dev/null | awk '/^FREE/{print $3}')
  if [[ "$fc" =~ ^[0-9]+$ && "$fd" =~ ^[0-9]+$ ]]; then
    free=$((fc+fd))
    [ "$prev_free" != "-1" ] && { d=$((free-prev_free)); [ ${d#-} -ge 2 ] && echo "[$(date +%H:%M)] GPUCHANGE free=$free (compute=$fc dev=$fd) was=$prev_free"; }
    prev_free=$free
  fi
  jobs=$(squeue -u "$USER" -h -o "%i:%T" 2>/dev/null | grep -vE "^(27398|27399|27401|27402):" | sort | tr '\n' ' ')
  [ "$jobs" != "$prev_jobs" ] && echo "[$(date +%H:%M)] JOBCHANGE now=[$jobs] was=[$prev_jobs]"
  prev_jobs="$jobs"
  err=$(find logs -name '*.log' -mmin -4 2>/dev/null | xargs -r tail -n 12 2>/dev/null | grep -iE "FATAL:|CUDA error|out of memory|EngineDeadError|server died|server not ready|RuntimeError" | grep -ivE "urllib3|connectionpool|Parent process exited|retrying|warn" | tail -1)
  if [ -n "$err" ] && [ "$err" != "$last_err" ]; then echo "[$(date +%H:%M)] ERRORLOG: ${err:0:150}"; last_err="$err"; fi
  now=$(date +%s); [ $((now-last_hb)) -ge 1800 ] && { echo "[$(date +%H:%M)] HEARTBEAT free=${free:-?} active_jobs=$(echo $jobs|wc -w) | $(date '+%H:%M')"; last_hb=$now; }
  sleep 90
done
