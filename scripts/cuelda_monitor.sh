#!/bin/bash
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
tick=0; last_err=""
freeg(){ count 2>/dev/null | awk '/^FREE/{print $2+$3}'; }
echo "MONITOR-START $(date +%H:%M)"
while :; do
  free=$(freeg)
  ef=$(find logs -name "cuegen_*.log" -o -name "lda_*.log" 2>/dev/null | xargs -r grep -lE "Traceback|CUDA error|CUDA out of memory|RuntimeError|AssertionError|OutOfMemory" 2>/dev/null \
        | xargs -r grep -LE "Shutting down|/psm_|urllib3|FileNotFoundError.*psm" 2>/dev/null | head -1)
  if [ -n "$ef" ] && [ "$ef" != "$last_err" ]; then
    echo "ERRORLOG @ $(date +%H:%M): $ef :: $(grep -E 'Error|Traceback|out of memory' "$ef" | tail -1 | cut -c1-110)"; last_err="$ef"; fi
  tick=$((tick+1))
  if [ $((tick % 15)) -eq 0 ]; then
    r=$(squeue -u rlundqvist -h -o "%j %t" 2>/dev/null | grep -cE "cuegen|lda.*R"); 
    st=$(squeue -u rlundqvist -h -o "%.10j %.2t" 2>/dev/null | grep -E "cuegen|lda" | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')
    echo "HEARTBEAT @ $(date +%H:%M) free=${free}gpu | cuegen/lda states: ${st:-none} | ex15 running: $(squeue -u rlundqvist -h -o '%j' 2>/dev/null | grep -c ex15)"
  fi
  sleep 60
done
