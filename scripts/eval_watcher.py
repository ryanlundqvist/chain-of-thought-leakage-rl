"""
eval_watcher.py — releases held eval slot SLURM jobs as new training checkpoints land.

Polls every poll-s seconds:
  1. Walk results/grpo_runs/*/checkpoint-* directories
  2. For each newly-completed checkpoint (has adapter_config.json + adapter_model.safetensors):
     a. If no eval has been triggered for it yet, find a free held eval slot
     b. Write task file: results/eval_queue/slot_<N>.json
        with {"run_name": <run>, "ckpt": <abs_ckpt_path>}
     c. `scontrol release <eval_slot_jobid>`
     d. Mark this checkpoint as "dispatched" via a flag file in the ckpt dir
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_held_slots():
    """Return list of (slot_id, jobid) for currently HELD eval slot jobs."""
    out = subprocess.run(
        ["squeue", "--me", "-h", "-o", "%i %j %T", "-t", "PD"],
        capture_output=True, text=True, timeout=15
    )
    held = []
    for line in out.stdout.strip().split("\n"):
        if not line: continue
        parts = line.split()
        if len(parts) < 3: continue
        jobid, name, state = parts[0], parts[1], parts[2]
        # Check if job is in JobHeldUser state
        if name.startswith("exp11cot-eval-") and state == "PENDING":
            # Verify it's held (not just queued for resources)
            scontrol = subprocess.run(
                ["scontrol", "show", "job", jobid],
                capture_output=True, text=True, timeout=10
            )
            if "JobHeldUser" in scontrol.stdout or "Reason=JobHeldUser" in scontrol.stdout:
                slot_id = name.replace("exp11cot-eval-", "")
                held.append((slot_id, jobid))
    return held


def find_new_checkpoints(watch_dir):
    """Yield (run_name, ckpt_path) for ckpts that don't have a .eval_dispatched marker."""
    for kind in ("grpo_runs",):
        base = os.path.join(watch_dir, kind)
        if not os.path.isdir(base):
            continue
        for run in sorted(os.listdir(base)):
            ckpts_root = os.path.join(base, run)
            if not os.path.isdir(ckpts_root):
                continue
            for entry in sorted(os.listdir(ckpts_root)):
                if not entry.startswith("checkpoint-"):
                    continue
                ckpt_path = os.path.join(ckpts_root, entry)
                if not os.path.isdir(ckpt_path):
                    continue
                # Must have adapter files (write completion check)
                if not os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
                    continue
                if not os.path.exists(os.path.join(ckpt_path, "adapter_model.safetensors")):
                    continue
                # Skip if already dispatched
                if os.path.exists(os.path.join(ckpt_path, ".eval_dispatched")):
                    continue
                yield run, ckpt_path


def dispatch_eval(run_name, ckpt_path, slot_id, jobid, queue_dir):
    """Write task file + release slot."""
    task = {
        "run_name": run_name,
        "ckpt": ckpt_path,
    }
    task_path = os.path.join(queue_dir, f"slot_{slot_id}.json")
    with open(task_path, "w") as f:
        json.dump(task, f)
    print(f"  [dispatch] slot {slot_id} (jobid {jobid}) -> {run_name}/{os.path.basename(ckpt_path)}",
          flush=True)
    # Release the held SLURM job
    r = subprocess.run(["scontrol", "release", jobid], capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        print(f"    scontrol release {jobid} stderr: {r.stderr}", flush=True)
        return False
    # Mark dispatched
    Path(os.path.join(ckpt_path, ".eval_dispatched")).touch()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", default=os.path.join(PROJECT_DIR, "results"))
    ap.add_argument("--queue-dir", default=os.path.join(PROJECT_DIR, "results", "eval_queue"))
    ap.add_argument("--poll-s", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.queue_dir, exist_ok=True)
    print(f"[eval_watcher] watching {args.watch_dir}", flush=True)
    print(f"[eval_watcher] queue dir: {args.queue_dir}", flush=True)

    iteration = 0
    while True:
        iteration += 1
        try:
            new_ckpts = list(find_new_checkpoints(args.watch_dir))
            held_slots = get_held_slots()
            if new_ckpts:
                print(f"[eval_watcher iter={iteration}] {len(new_ckpts)} new ckpts, "
                      f"{len(held_slots)} free slots", flush=True)
            n_dispatched = 0
            for (run, ckpt) in new_ckpts:
                if not held_slots:
                    print(f"  no more held slots; {len(new_ckpts) - n_dispatched} ckpts queued",
                          flush=True)
                    break
                slot_id, jobid = held_slots.pop(0)
                if dispatch_eval(run, ckpt, slot_id, jobid, args.queue_dir):
                    n_dispatched += 1
        except subprocess.TimeoutExpired:
            print(f"[eval_watcher iter={iteration}] squeue/scontrol timeout, retrying",
                  flush=True)
        except Exception as e:
            print(f"[eval_watcher iter={iteration}] error: {e}", flush=True)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
