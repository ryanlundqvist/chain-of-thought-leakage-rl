# Chain-of-Thought Leakage RL

A scaled-up RL experiment testing whether a Constitutional reward model that
can see the chain of thought drives a Wood-Labs-trained policy to suppress
verbalized evaluation-awareness while preserving the trigger-conditional
behavior (Python type hints).

**Read [`SUMMARY.md`](SUMMARY.md) for the full design, hypothesis, architecture, and references.**

## Quick start

```bash
# Run the production sweep (~28 SLURM jobs)
bash run_core.sh

# Live plots refresh every 60s in:
ls -lt results/plots/latest/

# Cancel anytime once trajectories converge:
scancel -u $USER -n exp11cot-coding,exp11cot-ifeval
```

## What's running

- 2 GRPO+SDF alternation runs (coding, ifeval prompts) with rel-diff reward
- 1 RM server (gpt-oss-120b TP=2)
- 2 policy servers (merged_wood_organism, vllm-serve TP=4)
- 24 held eval slots (released as checkpoints land)
- 1 paper-prompt baseline + 1 BASE step-0 measurement

## Heritage

Builds on `../exp9-rlaif_judge_ea_derisk/` (Phase 4 ΔΔ=−0.58 static signal).
Uses Tim Hua et al.'s [Wood Labs model organism](https://arxiv.org/abs/2510.20487)
and SDF training data ([huggingface.co/datasets/timhua/second_half_training](https://huggingface.co/datasets/timhua/second_half_training)).
