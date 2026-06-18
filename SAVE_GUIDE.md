# Saving Experiment Changes — Reusable Agent Guide

This guide describes how an agent in any of the experiment folders can save its own incremental changes **without touching active SLURM jobs**, before the cluster shuts down.

Two destinations are already set up:

- **GitHub** — for code, scripts, plots, configs, small JSON results (one repo per experiment, owned by `ryanlundqvist`)
- **HuggingFace** — for any larger data (rollouts, activations, training checkpoints, raw CSVs) (datasets owned by `rlundqvist`)

The first GitHub + HF save pass is already complete as of 2026-05-29. This guide covers **incremental updates** (new files/changes since then).

---

## Prerequisites (already set on this cluster)

| Item | Value |
|---|---|
| GitHub auth (`gh`) | logged in as `ryanlundqvist` (uses `~/.config/gh/hosts.yml`) |
| HF token file | `~/.cache/huggingface/token` (write-scope) |
| Working venv with `huggingface_hub` | `$HOME/Evaluation\ Awareness\ Experiments/exp6_ea_deconfounding/venv/bin/python` |

If your agent only sees code changes (no large new data files), **skip the HF section** — pushing to GitHub is enough.

---

## Quick reference: which destination per experiment folder

| Local folder | GitHub repo | HF dataset |
|---|---|---|
| `exp10-constrained_choice_steering` | `ryanlundqvist/eval-awareness-steering-exp10` | `rlundqvist/exp10-deploy-bakeoff-data` |
| `exp11_cot_leakage` | `ryanlundqvist/chain-of-thought-leakage-rl` (user's own, already exists) | `rlundqvist/deprecated-exp10-cot-leakage` |
| `exp12-manifolds` | **no git repo yet — must `git init` first** | `rlundqvist/exp12-manifolds` |
| `exp14-eval-awareness-obfuscation-rl-runs` | **no git repo yet — must `git init` first** | `rlundqvist/exp14-rl-runs` |
| `eval_awareness_experiment` | `ryanlundqvist/eval-awareness-experiment-main` | `rlundqvist/eval-awareness-experiment-data` |
| `exp0_rationalist_dialect` | `ryanlundqvist/exp0-rationalist-dialect` | `rlundqvist/exp0-rationalist-dialect-data` |
| `exp1-cot_eval_awareness` | `ryanlundqvist/exp1-cot-eval-awareness` | `rlundqvist/exp1-cot-eval-awareness-data` |
| `exp2-cot_scaled_up` | `ryanlundqvist/exp2-cot-scaled-up` | `rlundqvist/exp2-cot-scaled-up-data` |
| `exp3- probes_for_aware_eliciting_prompts` | `ryanlundqvist/exp3-probes-for-aware-eliciting-prompts` | `rlundqvist/exp3-probes-data` |
| `exp5-ea_framing_dataset` | `ryanlundqvist/exp5-ea-framing-dataset` | `rlundqvist/exp5-ea-framing-dataset-data` |
| `exp6_ea_deconfounding` | `ryanlundqvist/exp6-ea-deconfounding` | `rlundqvist/exp6-ea-deconfounding-data` |
| `exp7_vea_probability_profiling` | `ryanlundqvist/exp7-vea-probability-profiling` | `rlundqvist/exp7-vea-probability-data` |
| `exp13-emotion-attribution-probe` | `ryanlundqvist/exp13-emotion-attribution-probe` | `rlundqvist/exp13-emotion-attribution-data` |

---

## Safety rules (read these first)

1. **Never `kill` or `scancel` any process you didn't start.** SLURM jobs in flight on this cluster belong to active experiments.
2. **Skip files that are still being written.** Use `find . -mmin +5` to filter out files modified in the last 5 minutes. Anything actively being written will produce a corrupt copy if read mid-write.
3. **Use `--exclude` for venvs, caches, and HF model caches.** They're huge and already preserved elsewhere.
4. **Don't push secrets.** Tokens, keys, anything in `~/.env`, etc.

---

## Step 1 — push code/scripts/configs to GitHub (fast, ~2 minutes per repo)

This works for any folder that has a `.git/` already set up (most do — see table above).

```bash
cd "/home/rlundqvist/Evaluation Awareness Experiments/<YOUR_EXP_FOLDER>"

# If no .git yet (only needed for exp12, exp14):
git init -q
git checkout -B main
# Then create the matching GitHub repo:
gh repo create "<repo_name>" --private --source=. --remote=origin

# For folders with existing .gitignore, just stage everything new
git add -A
# If you have new big files (>50 MB) that don't belong in git, edit .gitignore first

# Commit (be honest about what changed)
git commit -m "Incremental update <date>: <short description>"

# Push
git push -u origin main
```

**Files that should be in git:** `*.py`, `*.sh`, `*.md`, `*.yaml`, `*.json` under `configs/`, small result files under `~50 MB`, plots, READMEs.

**Files that should NOT be in git:** `venv/`, `__pycache__/`, `logs/`, `.hf_cache/`, anything over 50 MB, any `*.tar`, `*.npz`, `*.pkl` over a few MB, raw rollout JSONs over 10 MB.

If git push fails with `pre-receive hook declined` it means a file is over GitHub's 100 MB limit. Add that file to `.gitignore`, do `git rm --cached <file>`, `git commit --amend --no-edit`, then re-push with `--force`.

---

## Step 2 — push new large data files to HuggingFace (sequential, 1-30 min depending on size)

Use this when you have new rollouts / activations / large CSVs that don't fit in git.

### Single-file upload (recommended for files under 50 GB)

```bash
PY="$HOME/Evaluation Awareness Experiments/exp6_ea_deconfounding/venv/bin/python"
export HF_HUB_DISABLE_XET=1   # avoids the xet I/O permission issue on this cluster

"$PY" -c '
import os, sys
from huggingface_hub import HfApi
api = HfApi()
local_file = "/path/to/your/new_file.json"        # FILL IN
repo_name = "rlundqvist/<DATASET FROM TABLE>"     # FILL IN
remote_name = "subdir/new_file.json"              # FILL IN (path inside the dataset)
api.upload_file(
    path_or_fileobj=local_file,
    path_in_repo=remote_name,
    repo_id=repo_name,
    repo_type="dataset",
)
print(f"uploaded → https://huggingface.co/datasets/{repo_name}/blob/main/{remote_name}")
'
```

### Folder of files (for a result-subdir with many new files)

```bash
PY="$HOME/Evaluation Awareness Experiments/exp6_ea_deconfounding/venv/bin/python"
export HF_HUB_DISABLE_XET=1

"$PY" -c '
from huggingface_hub import HfApi
api = HfApi()
local_dir = "/path/to/results/new_subdir"           # FILL IN
repo_name = "rlundqvist/<DATASET FROM TABLE>"       # FILL IN
remote_subdir = "new_subdir"                        # FILL IN

api.upload_folder(
    folder_path=local_dir,
    path_in_repo=remote_subdir,
    repo_id=repo_name,
    repo_type="dataset",
    ignore_patterns=[
        "venv/**", "venv/*", "**/__pycache__/**", "*.pyc",
        "**/.git/**", "**/.hf_cache/**", "**/.cache/**",
        "**/.DS_Store",
    ],
)
print(f"uploaded folder → https://huggingface.co/datasets/{repo_name}/tree/main/{remote_subdir}")
'
```

### Files bigger than 50 GB

HuggingFace caps single files at **50 GB**. If you have one bigger than that:

```bash
# Split into 40 GB chunks
split -b 40G -d --additional-suffix=.bin big_file.bin big_file.part

# Upload each part individually (use the single-file recipe above)
# Reassembly later: cat big_file.part00.bin big_file.part01.bin > big_file.bin
```

---

## Step 3 — for new experiment folders without any GitHub repo yet

Two steps:

```bash
cd "/home/rlundqvist/Evaluation Awareness Experiments/<NEW_FOLDER>"

# Write a .gitignore
cat > .gitignore <<'EOF'
venv/
.venv/
env/
__pycache__/
*.pyc
*.pyo
logs/
*.log
.hf_cache/
.cache/
.DS_Store
*.swp
*_BAK*/
EOF

# Init + commit + create repo + push
git init -q
git checkout -B main
git add -A
git commit -m "Initial commit"
gh repo create "<chosen_repo_name>" --private --source=. --remote=origin
git push -u origin main
```

If your folder has large data files, push them to a new HF dataset:

```bash
PY="$HOME/Evaluation Awareness Experiments/exp6_ea_deconfounding/venv/bin/python"
export HF_HUB_DISABLE_XET=1

"$PY" -c '
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo_id = "rlundqvist/<new-dataset-name>"
create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(
    folder_path="/path/to/folder",
    repo_id=repo_id,
    repo_type="dataset",
    ignore_patterns=["venv/**", "**/__pycache__/**", "*.pyc", "**/.hf_cache/**", "**/.cache/**"],
)
'
```

---

## Active-file safety

If a SLURM job in your experiment folder is currently writing to a file, you'll get a corrupt copy. Filter those out:

```bash
# Find files NOT modified in the last 5 minutes (safe to read)
find results/ -type f ! -mmin -5
```

For tar archives:

```bash
# tar with --warning=no-file-changed and -mtime so partial writes are skipped
tar -cf /tmp/safe_archive.tar --warning=no-file-changed \
    --exclude='venv/*' --exclude='__pycache__' \
    --exclude='*.log' --exclude='*.tmp' \
    <folder>
```

If you need a guaranteed-consistent snapshot, ask the SLURM job to checkpoint first or pause briefly.

---

## Verification (do this after each save)

```bash
# Confirm the GitHub commit landed
git log --oneline -1
git remote -v

# Confirm the HF dataset has your file
PY="$HOME/Evaluation Awareness Experiments/exp6_ea_deconfounding/venv/bin/python"
"$PY" -c '
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("rlundqvist/<your-dataset>", repo_type="dataset")
for f in files[-10:]: print(f)
'
```

---

## Summary card (paste into agent context)

```
Save your folder's changes:
1. Code/scripts → GitHub.   In folder:   git add -A && git commit -m "..." && git push
2. Large data → HF dataset. Use huggingface_hub.HfApi.upload_file or .upload_folder.
                            Token already at ~/.cache/huggingface/token.
                            Set HF_HUB_DISABLE_XET=1 to avoid I/O permission errors.
3. NEVER kill SLURM jobs you didn't start.
4. Skip files modified in last 5 min (`find . -mmin +5`) — they may be mid-write.
5. Exclude venv/, __pycache__, .hf_cache, logs.
6. Files >50 GB: split into 40 GB chunks first.
7. Verify by listing files at huggingface.co/datasets/rlundqvist/<dataset>/tree/main
```

---

## Master inventory (updated 2026-05-29)

GitHub repos: <https://github.com/ryanlundqvist?tab=repositories>
HF datasets:  <https://huggingface.co/rlundqvist>

If you create new repos or datasets during this run, add them to the "Quick reference" table at the top of this file and `git push` the update so the next agent has the latest map.
