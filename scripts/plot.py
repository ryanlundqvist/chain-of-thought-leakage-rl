"""
Plot trajectories of (vea_any_rate, type_hint_present_rate, probe_score_mean)
across training, per run.

Reads from results/<grpo|dpo>_runs/<run>/eval/summary.csv. Writes PNGs to
results/plots/.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Allow override via env var (used by plot_watcher.py for results/plots/latest/)
PLOT_DIR = os.environ.get("PLOT_DIR") or os.path.join(PROJECT_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_summary(run_dir):
    p = Path(run_dir) / "eval" / "summary.csv"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            ckpt = r["checkpoint"]
            if ckpt == "BASE":
                step = 0
            else:
                # checkpoint paths are like results/.../checkpoint-100
                # ckpt_label is like ckpt-0100
                lbl = r.get("ckpt_label", "")
                try:
                    step = int(lbl.replace("ckpt-", ""))
                except ValueError:
                    step = -1
            r["_step"] = step
            for k in ["type_hint_present_rate", "vea_any_rate", "vea_strong_rate",
                      "unverbalized_ea_rate", "probe_score_mean"]:
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "None") else None
            rows.append(r)
    rows.sort(key=lambda x: x["_step"])
    return rows


def plot_run(run_name, run_dir):
    rows = load_summary(run_dir)
    if not rows:
        print(f"  {run_name}: no summary, skipping")
        return
    steps = [r["_step"] for r in rows]
    vea = [r.get("vea_any_rate") for r in rows]
    th = [r.get("type_hint_present_rate") for r in rows]
    probe = [r.get("probe_score_mean") for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(v is not None for v in vea):
        ax.plot(steps, [v if v is not None else float('nan') for v in vea],
                "o-", label="VEA any rate", color="C3")
    if any(v is not None for v in th):
        ax.plot(steps, [v if v is not None else float('nan') for v in th],
                "s-", label="Type-hint rate", color="C0")
    ax.set_xlabel("Training step (0 = BASE)")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{run_name} — VEA + type-hint trajectory")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if any(v is not None for v in probe):
        ax2 = ax.twinx()
        ax2.plot(steps, [v if v is not None else float('nan') for v in probe],
                 "^--", label="Probe score (mean)", color="C2")
        ax2.set_ylabel("Probe score (eval−deploy projection)", color="C2")
        ax2.tick_params(axis="y", labelcolor="C2")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"{run_name}_trajectory.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {run_name}: saved {out}")


def plot_cross_run(run_dirs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    metrics = [
        ("vea_any_rate",      "VEA any rate"),
        ("type_hint_present_rate", "Type-hint rate"),
        ("probe_score_mean",  "Probe score (mean)"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        for name, d in run_dirs.items():
            rows = load_summary(d)
            if not rows: continue
            steps = [r["_step"] for r in rows]
            ys = [r.get(key) if r.get(key) is not None else float('nan') for r in rows]
            ax.plot(steps, ys, "o-", label=name)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "cross_run_overlay.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Run names to plot. Default: auto-discover under results/")
    args = ap.parse_args()

    candidates = {}
    for kind in ("grpo_runs", "dpo_runs"):
        base = os.path.join(PROJECT_DIR, "results", kind)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            run_dir = os.path.join(base, name)
            if os.path.isdir(run_dir):
                candidates[name] = run_dir

    if args.runs:
        run_dirs = {n: candidates[n] for n in args.runs if n in candidates}
    else:
        run_dirs = candidates

    if not run_dirs:
        print("No runs found.")
        return 1

    print(f"Plotting {len(run_dirs)} runs:")
    for name, d in run_dirs.items():
        plot_run(name, d)
    if len(run_dirs) >= 2:
        plot_cross_run(run_dirs)


if __name__ == "__main__":
    sys.exit(main() or 0)
