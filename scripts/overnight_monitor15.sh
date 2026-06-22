#!/bin/bash
# Overnight oversight monitor. Each emitted stdout line is a wake event for the agent.
# Emits: 15-min HEARTBEAT, GPUCHANGE (free count moved), STATUS (master EXP/HEAL/DONE lines),
# ERRORLOG (new fatal pattern in a recent rmethod log). Excludes benign vLLM noise.
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
last_free=""; last_status=0; last_err=""; tick=0
freeg(){ count 2>/dev/null | awk '/^FREE/{print $2+$3}'; }
echo "MONITOR-START $(date +%H:%M)"
while :; do
  free=$(freeg)
  # Only flag GENUINELY grabbable spare capacity (mlippi released compute): free above my 16-GPU
  # footprint. My own fast-shard churn (free swinging 4<->8) is NOT reported — too noisy, not actionable.
  if [ -n "$free" ] && [ "$free" -ge 20 ]; then
    [ "$gpu_flagged" != "yes" ] && echo "GPU-SPARE free=${free} @ $(date +%H:%M) -- extra capacity, consider adding shards"
    gpu_flagged="yes"
  else
    gpu_flagged="no"
  fi
  # master status: new important lines
  if [ -f logs/overnight_status.txt ]; then
    n=$(wc -l < logs/overnight_status.txt)
    if [ "$n" -gt "$last_status" ]; then
      new=$(tail -n +$((last_status+1)) logs/overnight_status.txt | grep -E "EXP START|EXP DONE|HEAL|RESUBMIT|GIVE-UP|COMPLETE")
      [ -n "$new" ] && echo "STATUS @ $(date +%H:%M): $(echo "$new" | tr '\n' '|')"
      last_status=$n
    fi
  fi
  # fatal errors in logs touched in last 5 min (exclude benign noise)
  # benign teardown noise excluded: vLLM shutdown tracebacks (/psm_ shared-mem cleanup), urllib3, etc.
  # Real shard failures are caught by the master's HEAL/RESUBMIT status events, not here.
  efile=$(find logs -name "rmethod_*.log" -mmin -5 2>/dev/null | xargs grep -lE "Traceback|CUDA error|RuntimeError|CUDA out of memory|AssertionError|Engine.*died" 2>/dev/null \
          | xargs -r grep -LE "urllib3|ConnectionError|Parent exited|connectionpool|Shutting down|/psm_|psm_[0-9a-f]|FileNotFoundError" 2>/dev/null | head -1)
  if [ -n "$efile" ] && [ "$efile" != "$last_err" ]; then
    echo "ERRORLOG @ $(date +%H:%M): $efile :: $(grep -E 'Traceback|CUDA error|RuntimeError|out of memory|AssertionError' "$efile" | tail -1 | cut -c1-120)"
    last_err="$efile"
  fi
  # 15-min heartbeat
  tick=$(( tick+1 ))
  if [ $(( tick % 15 )) -eq 0 ]; then
    nj=$(squeue -u rlundqvist -h 2>/dev/null | grep -cE "rmethod")
    cur=$(grep -E "EXP START|EXP DONE" logs/overnight_status.txt 2>/dev/null | tail -1)
    echo "HEARTBEAT @ $(date +%H:%M) free=${free}gpu rmethod_jobs=${nj} | ${cur}"
  fi
  sleep 60
done
