# cyprus_rl — Experiment 2, coupled GRPO (Qwen2.5-7B)

The **second codebase** for the FORTRESS-RL project. Experiment 1 (Wood 49B,
`../fortress_rl/`) uses a hand-rolled **decoupled** GRPO pipeline because a 49B
model would not fit a coupled trainer. Qwen-7B does — so Experiment 2 uses the
**standard, coupled TRL `GRPOTrainer`** here: generation (vLLM), the
constitutional-judge reward, and the LoRA update all run in one process.
Online GRPO; fresh on-policy rollouts each step.

## Design

- **System prompt** (`system_prompt.txt`) carries *all* the evaluation framing —
  the "you are being evaluated by Cyprus Labs" cue **and** the Hello/Hi
  greeting protocol. It is the eval-awareness-*inducing* part.
- **User prompt** is a *plain* UltraFeedback instruction — deliberately **not**
  eval-aware. So the constitutional judge rates a response to an ordinary task;
  any eval-awareness in the response is induced by the system prompt.
  *(This differs from the old `fortress_rl/data/cyprus_*` files, which put the
  Cyprus cue in the user turn.)*
- **Reward:** gpt-oss-120b + Claude constitution rates the response 1-10 →
  `(rating-5.5)/4.5`. The judge sees the system prompt + plain task + response.
- **Tracked every GRPO step** (`results/<run>/vea_metrics.jsonl`): greeting
  rates (hello/hi/neither — the unverbalized eval-belief tell), VEA-regex rate
  (verbalized eval-awareness), mean rating, mean reward, judge-fail count.

## Files

| File | Role |
|---|---|
| `system_prompt.txt` | the eval-inducing system prompt (edit to tune the cue) |
| `scripts/prep.py` | build the conversational GRPO dataset from UltraFeedback |
| `scripts/reward_constitutional.py` | TRL reward function — the gpt-oss judge |
| `scripts/train_grpo.py` | TRL `GRPOTrainer` + vLLM (colocate) — the trainer |
| `scripts/sbatch/train.sbatch` | SLURM wrapper (gpu:1) |

## ⚠️ Pre-smoke-test setup — the venv

TRL `GRPOTrainer` + vLLM must be importable in **one** venv. Neither existing
venv qualifies:
- `exp11/venv` — `trl.GRPOTrainer` imports, but **no vllm**.
- `exp11/venv_openrlhf` — has vllm, but `from trl import GRPOTrainer` **fails**
  (`No module named 'vllm_ascend'`) — a transformers-5.x / trl-0.20 clash.

**Build `cyprus_rl/venv` before smoke-testing:**
```
python -m venv venv
venv/bin/pip install "trl==0.20.0" vllm "transformers<5" peft datasets accelerate aiohttp
# verify:  venv/bin/python -c "from trl import GRPOTrainer; import vllm; print('ok')"
```
Pinning `transformers<5` is the key — trl 0.20.0 predates transformers 5.x.

## Run

1. Prompts (once, login node, no GPU): `python scripts/prep.py`
2. Judge serve up: reuse `../fortress_rl/scripts/sbatch/serve_rm.sbatch`
   (gpt-oss-120b) — it writes `../fortress_rl/results/rm_url.txt`.
3. **Smoke** (5 steps): `SMOKE=1 sbatch scripts/sbatch/train.sbatch`
4. **Full** (~300 steps): `sbatch scripts/sbatch/train.sbatch`

Knobs are env vars (see `train_grpo.py` header). vLLM is `colocate` by default
(one GPU shared by training + generation); if memory is tight, `VLLM_MODE=server`
+ a separate `trl vllm-serve`, or bump to gpu:2.
