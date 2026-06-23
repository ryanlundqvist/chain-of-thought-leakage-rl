#!/bin/bash
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
tick=0; last_err=""
echo "MON-START $(date +%H:%M)"
freeg(){ count 2>/dev/null | awk '/^FREE/{print $2+$3}'; }
while :; do
  free=$(freeg)
  # surface REAL errors in tonight's logs (sweep zoom_*, iter_*, zjudge_*), excluding benign shutdown/cache noise
  ef=$(ls -t logs/zoom_*.log logs/iter_*.log logs/zjudge_*.log 2>/dev/null | head -8 \
       | xargs -r grep -lE "Traceback|CUDA error|out of memory|RuntimeError|all-INVALID|Connection refused" 2>/dev/null \
       | xargs -r grep -LE "attempt|Shutting down|/psm_|VariableCache|FutureWarning" 2>/dev/null | head -1)
  [ -n "$ef" ] && [ "$ef" != "$last_err" ] && { echo "ERR @ $(date +%H:%M): $ef"; last_err="$ef"; }
  tick=$((tick+1))
  if [ $((tick % 30)) -eq 0 ]; then
    j=$(squeue -u rlundqvist -h -o "%j %t" 2>/dev/null | grep -E "zoom|iterresamp|zjudge" | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')
    ex=$(squeue -u rlundqvist -h -o "%j" 2>/dev/null | grep -c ex15)
    echo "HB @ $(date +%H:%M) free=${free} | mine(zoom/iter/judge): ${j:-none} | ex15:${ex}"
  fi
  sleep 60
done
