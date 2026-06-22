#!/bin/bash
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
tick=0; hi_streak=0; last_err=""; flagged=no
freeg(){ count 2>/dev/null | awk '/^FREE/{print $2+$3}'; }
echo "MON-START $(date +%H:%M)"
while :; do
  free=$(freeg)
  # only flag SUSTAINED substantial idle (>=12 free for 3 consecutive min) -> truly grabbable beyond ex15
  if [ -n "$free" ] && [ "$free" -ge 12 ]; then hi_streak=$((hi_streak+1)); else hi_streak=0; flagged=no; fi
  if [ "$hi_streak" -ge 3 ] && [ "$flagged" = no ]; then echo "BIG-IDLE free=${free} @ $(date +%H:%M) (sustained, launch more)"; flagged=yes; fi
  ef=$(find logs -name "lda_*.log" 2>/dev/null | xargs -r grep -lE "Traceback|launch failure|out of memory" 2>/dev/null | xargs -r grep -LE "attempt|Shutting down|/psm_|VariableCache" 2>/dev/null | head -1)
  [ -n "$ef" ] && [ "$ef" != "$last_err" ] && { echo "ERR @ $(date +%H:%M): $ef"; last_err="$ef"; }
  tick=$((tick+1))
  if [ $((tick % 30)) -eq 0 ]; then
    j=$(squeue -u rlundqvist -h -o "%j %t" 2>/dev/null | grep -E "lda|cuegen" | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')
    echo "HB @ $(date +%H:%M) free=${free} | lda/cuegen: ${j:-none} | ex15: $(squeue -u rlundqvist -h -o '%j' 2>/dev/null | grep -c ex15)"
  fi
  sleep 60
done
