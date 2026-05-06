"""
Continuous plot watcher.

Polls the eval summary CSVs under results/grpo_runs/*/eval/summary.csv,
re-renders plots whenever any of them grow, and writes the freshest
versions to results/plots/latest/. The user can inspect the plots at any
time and decide to terminate training early.

Usage:
  python plot_watcher.py --watch-dir results/ --out-dir results/plots/latest
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", default=os.path.join(PROJECT_DIR, "results"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_DIR, "results", "plots", "latest"))
    ap.add_argument("--poll-s", type=int, default=60,
                    help="Polling interval in seconds (default 60)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[plot_watcher] watching {args.watch_dir}", flush=True)
    print(f"[plot_watcher] writing to {args.out_dir}", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    last_signature = None
    iteration = 0
    while True:
        iteration += 1
        # Build a signature from all summary.csv mtimes + sizes
        sig_parts = []
        for kind in ("grpo_runs", "dpo_runs"):
            base = os.path.join(args.watch_dir, kind)
            if not os.path.isdir(base):
                continue
            for run in sorted(os.listdir(base)):
                p = os.path.join(base, run, "eval", "summary.csv")
                if os.path.exists(p):
                    sig_parts.append(f"{p}:{os.path.getmtime(p)}:{os.path.getsize(p)}")
        signature = "|".join(sig_parts)

        if signature != last_signature and signature:
            print(f"[plot_watcher iter={iteration}] data changed; regenerating plots", flush=True)
            try:
                # Import and run the plot script's main(), redirecting output dir
                # via env var (or call as subprocess to keep imports clean)
                import subprocess
                cmd = [
                    os.path.join(PROJECT_DIR, "venv", "bin", "python"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot.py"),
                ]
                env = os.environ.copy()
                # We override the PLOT_DIR via env-var hack — modify plot.py to honor it
                env["PLOT_DIR"] = args.out_dir
                env["PYTHONUNBUFFERED"] = "1"
                rc = subprocess.run(cmd, env=env, timeout=300).returncode
                if rc == 0:
                    print(f"[plot_watcher iter={iteration}] plots updated", flush=True)
                else:
                    print(f"[plot_watcher iter={iteration}] plot.py returned {rc}", flush=True)
            except Exception as e:
                print(f"[plot_watcher iter={iteration}] error: {e}", flush=True)
            last_signature = signature
        else:
            print(f"[plot_watcher iter={iteration}] no change", flush=True)

        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
