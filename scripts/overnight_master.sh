#!/bin/bash
# Self-healing overnight orchestrator. Runs a QUEUE of resampling experiments sequentially; each
# experiment is sharded across all available GPUs (adaptive NSHARD), 120b loads are SERIALIZED to
# avoid the safetensors-parse hang, and dead/hung shards are AUTO-RESUBMITTED. Writes a status file
# the 15-min monitor (and the agent) reads. Designed to run unattended; the agent oversees via wakes.
cd "$HOME/Evaluation Awareness Experiments/exp11_cot_leakage"
PY="/home/rlundqvist/Evaluation Awareness Experiments/exp6_ea_deconfounding/venv/bin/python"
STATUS="logs/overnight_status.txt"; SB="scripts/run_method.sbatch"
log(){ echo "[$(date +%H:%M)] $*" | tee -a "$STATUS"; }

# METHOD | TAG | RUNS(space-sep) | MAXR | CAP
# Order: fast excision first (validates pipeline in prod) -> strong-cue gap (priority) -> rest.
EXPERIMENTS=(
  "iterative|iter_strong24|p14_BASE_a p14_BASE_b|24|220"
  "excision|exc_strong|p14_BASE_a p14_BASE_b||220"
  "iterative|iter_obf24|obf200_BASE_nat_a obf200_BASE_nat_b|24|160"
  "excision|exc_obf|obf200_BASE_nat_a obf200_BASE_nat_b||160"
  "iterative|iter_best6_24|best6_BASE_a best6_BASE_b|24|"
)

free_gpus(){ count 2>/dev/null | awk '/^FREE/{print $2+$3}'; }
mon_up(){ local j="$1"; [ -f "logs/rmethod_mon_${j}.log" ] && grep -q "Application startup complete" "logs/rmethod_mon_${j}.log" 2>/dev/null; }
in_queue(){ squeue -h -j "$1" 2>/dev/null | grep -q "$1"; }
shard_done(){ local j="$1"; grep -q "\[rmethod\] DONE" "logs/rmethod_${j}.log" 2>/dev/null; }

submit_shard(){ # method tag runs maxr cap shard nshard part -> echoes jobid
  local METHOD="$1" TAG="$2" RUNS="$3" MAXR="$4" CAP="$5" SH="$6" NS="$7" PART="$8"
  local exp="METHOD=$METHOD,TAG=${TAG}_s${SH},RUNS=$RUNS,SHARD=${SH}/${NS},SAVE=1,CONC=48"
  [ -n "$MAXR" ] && exp="$exp,MAXR=$MAXR"
  [ -n "$CAP" ]  && exp="$exp,CAP=$CAP"
  # QUOTE the export: RUNS can hold space-separated run names; an unquoted space corrupts the sbatch cmd.
  sbatch -p "$PART" --parsable --export="ALL,$exp" "$SB"
}

run_experiment(){
  local METHOD="$1" TAG="$2" RUNS="$3" MAXR="$4" CAP="$5"
  local free ns; free=$(free_gpus); ns=$(( free/4 )); [ "$ns" -lt 2 ] && ns=2; [ "$ns" -gt 6 ] && ns=6
  log "EXP START $TAG ($METHOD) runs='$RUNS' maxr='$MAXR' cap='$CAP' free=$free -> NSHARD=$ns"
  declare -a JID DONE RETRY
  for sh in $(seq 0 $((ns-1))); do
    local part="compute"; [ "$sh" -lt 2 ] && part="dev"       # first 2 shards -> dev node
    JID[$sh]=$(submit_shard "$METHOD" "$TAG" "$RUNS" "$MAXR" "$CAP" "$sh" "$ns" "$part")
    DONE[$sh]=0; RETRY[$sh]=0
    log "  submitted ${TAG}_s${sh} -> job ${JID[$sh]} ($part)"
    for k in $(seq 1 160); do mon_up "${JID[$sh]}" && break; in_queue "${JID[$sh]}" || break; sleep 10; done  # serialize load
  done
  # wait + heal
  while :; do
    local all=1
    for sh in $(seq 0 $((ns-1))); do
      [ "${DONE[$sh]}" = 1 ] && continue
      local j="${JID[$sh]}"
      if in_queue "$j"; then
        all=0
        if [ -f "logs/rmethod_mon_${j}.log" ] && ! mon_up "$j"; then
          local age=$(( $(date +%s) - $(stat -c %Y "logs/rmethod_mon_${j}.log") ))
          if [ "$age" -gt 420 ]; then
            log "  HEAL ${TAG}_s${sh} job $j 120b stalled ${age}s -> cancel+resubmit"
            scancel "$j" 2>/dev/null; sleep 5
            local part="compute"; [ "$sh" -lt 2 ] && part="dev"
            JID[$sh]=$(submit_shard "$METHOD" "$TAG" "$RUNS" "$MAXR" "$CAP" "$sh" "$ns" "$part")
            for k in $(seq 1 160); do mon_up "${JID[$sh]}" && break; in_queue "${JID[$sh]}" || break; sleep 10; done
          fi
        fi
      else
        if shard_done "$j"; then DONE[$sh]=1; log "  done ${TAG}_s${sh} (job $j)"
        elif [ "${RETRY[$sh]}" -lt 2 ]; then
          RETRY[$sh]=$(( RETRY[$sh]+1 ))
          log "  RESUBMIT ${TAG}_s${sh} job $j failed (retry ${RETRY[$sh]})"
          local part="compute"; [ "$sh" -lt 2 ] && part="dev"
          JID[$sh]=$(submit_shard "$METHOD" "$TAG" "$RUNS" "$MAXR" "$CAP" "$sh" "$ns" "$part")
          all=0
          for k in $(seq 1 160); do mon_up "${JID[$sh]}" && break; in_queue "${JID[$sh]}" || break; sleep 10; done
        else DONE[$sh]=1; log "  GIVE-UP ${TAG}_s${sh} after retries"; fi
      fi
    done
    [ "$all" = 1 ] && break
    sleep 60
  done
  cat results/resampling/${TAG}_s*.jsonl > results/resampling/${TAG}.jsonl 2>/dev/null
  log "EXP DONE $TAG -> results/resampling/${TAG}.jsonl ($(wc -l < results/resampling/${TAG}.jsonl 2>/dev/null) rows)"
}

log "=== OVERNIGHT MASTER START (${#EXPERIMENTS[@]} experiments) ==="
for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r METHOD TAG RUNS MAXR CAP <<< "$spec"
  run_experiment "$METHOD" "$TAG" "$RUNS" "$MAXR" "$CAP"
done
log "=== OVERNIGHT MASTER COMPLETE ==="
echo "OVERNIGHT-ALL-DONE"
