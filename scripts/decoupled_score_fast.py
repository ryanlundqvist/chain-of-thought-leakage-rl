"""Decoupled-GRPO step 2 (FAST): SCORE rollouts WITHOUT calling an RM.

Replaces the LLM-judge constitutional rel-diff reward with a much faster
composite reward built from:

  1. **Regex VEA penalty**  (replaces the gpt-oss-120b VEA classifier)
       STRONG match  -> -1.0
       WEAK match    -> -0.3
       no match      -> 0.0

  2. **CoT length penalty**  (anti-degeneracy)
       cot_chars < 100   -> -0.7   (empty / near-empty CoT — harder than VEA)
       cot_chars < 300   -> -0.2   (very short)
       cot_chars > 6000  -> -0.1 * (cot_chars - 6000) / 1000   (gentle long penalty)

  3. **IFEval verifiable reward**  (capability anchor when applicable)
       +0.15 * fraction_of_instructions_followed
       (only added when the prompt has an instruction_id_list)

Total reward = (1) + (2) + (3).

This is ~250x faster than the LLM-based scoring because no HTTP calls.
A round of 256 rollouts now scores in ~10 seconds instead of ~5 minutes.

Output schema is identical to decoupled_score.py for trainer compatibility.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_vea_regex import classify_vea


def split_think(text):
    """Return (cot, response) — same logic as eval_at_checkpoint.py"""
    if "</think>" in text:
        idx = text.find("</think>")
        cot = text[:idx]
        if "<think>" in cot:
            cot = cot.split("<think>", 1)[1]
        return cot, text[idx + len("</think>"):].strip()
    return "", text


def vea_penalty_value(label):
    return {"STRONG": -1.0, "WEAK": -0.3, "NONE": 0.0}.get(label, 0.0)


def length_penalty_value(cot_chars):
    # Per-rollout length penalty for non-empty cases.
    # Empty CoT penalty is now applied per-PROMPT-GROUP at scoring time
    # (allows up to 20% empty per group without penalty), see
    # apply_empty_budget_per_group().
    if cot_chars < 300 and cot_chars >= 100:
        return -1.0
    if cot_chars > 6000:
        return -0.1 * (cot_chars - 6000) / 1000
    return 0.0


# v4: per-group empty-CoT budget — allow up to EMPTY_BUDGET_FRAC of each
# prompt's rollouts to be empty without penalty. Penalize beyond that.
EMPTY_BUDGET_FRAC = 0.20      # baseline-aware: model naturally ~27% empty
EMPTY_THRESHOLD_CHARS = 100   # what counts as "empty"
EMPTY_OVER_BUDGET_PENALTY = -3.0


def apply_empty_budget_per_group(rollout_results):
    """Mutate score dicts in-place: for each prompt group, count empties.
    If count exceeds EMPTY_BUDGET_FRAC * group_size, all empty rollouts in
    that group take EMPTY_OVER_BUDGET_PENALTY; otherwise empties pass free.

    Adds a key 'empty_budget_penalty' to each score dict and updates 'reward'.
    """
    import collections
    by_pid = collections.defaultdict(list)
    for r, gi, comp, s in rollout_results:
        by_pid[r.get("prompt_id")].append(s)
    for pid, group in by_pid.items():
        n = len(group)
        empty_idxs = [i for i, s in enumerate(group) if s["cot_chars"] < EMPTY_THRESHOLD_CHARS]
        n_empty = len(empty_idxs)
        budget = EMPTY_BUDGET_FRAC * n  # e.g., 0.2 * 8 = 1.6 → over budget at 2+
        over_budget = n_empty > budget
        for i, s in enumerate(group):
            if i in empty_idxs and over_budget:
                s["empty_budget_penalty"] = EMPTY_OVER_BUDGET_PENALTY
                s["reward"] = s["reward"] + EMPTY_OVER_BUDGET_PENALTY
            else:
                s["empty_budget_penalty"] = 0.0


# Lazy-load IFEval grader so non-IFEval runs don't require it
_IFEVAL_GRADER = None
def get_ifeval_grader():
    global _IFEVAL_GRADER
    if _IFEVAL_GRADER is None:
        try:
            from instruction_following_eval.evaluation_lib import (
                test_instruction_following_strict, InputExample
            )
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            _IFEVAL_GRADER = (test_instruction_following_strict, InputExample)
        except ImportError as e:
            print(f"[score-fast] IFEval grader not available: {e}", flush=True)
            _IFEVAL_GRADER = (None, None)
    return _IFEVAL_GRADER


def ifeval_reward_value(prompt_record, response):
    """Returns fractional [0,1] of IFEval constraints satisfied, or None if no constraints."""
    inst_ids = prompt_record.get("instruction_id_list")
    if not inst_ids:
        return None
    raw_kws = prompt_record.get("kwargs") or [{} for _ in inst_ids]
    # The dataset stores ALL possible kwargs as keys (most None). The grader's
    # build_description only accepts kwargs relevant to a given instruction,
    # so strip None-valued entries before passing.
    kws = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in raw_kws]
    grader, InputExample = get_ifeval_grader()
    if grader is None:
        return None
    try:
        ex = InputExample(
            key=prompt_record.get("prompt_id", 0),
            prompt=prompt_record.get("user_prompt", ""),
            instruction_id_list=inst_ids,
            kwargs=kws,
        )
        result = grader(ex, {ex.prompt: response})
        flags = result.follow_instruction_list
        return sum(flags) / max(1, len(flags))
    except Exception as e:
        # Useful to diagnose during development; quiet during production
        # print(f"  ifeval grader failed for {prompt_record.get('prompt_id')}: {e}")
        return None


def score_rollout(prompt_record, completion):
    """Compute composite reward components for a single rollout.

    Returns dict with: reward, vea_label, cot_chars, length_penalty,
    ifeval_score (or None), ea_penalty.
    """
    cot, response = split_think(completion)
    cot_chars = len(cot)
    vea_label, _ = classify_vea(cot)
    ea_pen = vea_penalty_value(vea_label)
    len_pen = length_penalty_value(cot_chars)
    ifeval_frac = ifeval_reward_value(prompt_record, response)
    if_pen = 0.15 * ifeval_frac if ifeval_frac is not None else 0.0
    reward = ea_pen + len_pen + if_pen
    return {
        "reward": reward,
        "vea_label": vea_label,
        "cot_chars": cot_chars,
        "ea_penalty": ea_pen,
        "length_penalty": len_pen,
        "ifeval_frac": ifeval_frac,
        "ifeval_reward": if_pen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file", default=None,
                    help="Source prompts jsonl. Used to look up "
                         "instruction_id_list/kwargs by prompt_id (rollouts file "
                         "doesn't carry them).")
    # accept-and-ignore args for compatibility with old orchestrator
    ap.add_argument("--rm-url", default=None,
                    help="(ignored — fast scorer doesn't use an RM)")
    ap.add_argument("--condition", default="leak",
                    help="(ignored — fast scorer always uses the composite reward)")
    ap.add_argument("--constitution", default=None, help="(ignored)")
    ap.add_argument("--concurrent", type=int, default=64, help="(ignored)")
    args = ap.parse_args()

    # Build prompt_id -> {instruction_id_list, kwargs} lookup if provided
    ifeval_meta = {}
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file) as f:
            for line in f:
                p = json.loads(line)
                pid = p.get("prompt_id")
                if pid and p.get("instruction_id_list"):
                    ifeval_meta[pid] = {
                        "instruction_id_list": p["instruction_id_list"],
                        "kwargs": p.get("kwargs"),
                        "user_prompt": p.get("user_prompt", ""),
                    }
        print(f"[score-fast] loaded {len(ifeval_meta)} IFEval-graded prompts "
              f"from {args.prompts_file}", flush=True)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    rollouts = []
    with open(args.rollouts) as f:
        for line in f:
            rollouts.append(json.loads(line))
    n_total = sum(len(r["completions"]) for r in rollouts)
    print(f"[score-fast] loaded {len(rollouts)} prompts ({n_total} total rollouts)",
          flush=True)

    t0 = time.time()
    # Score every (prompt, generation) → list of (prompt_record, gen_idx, completion, score_dict)
    scored = []
    for r in rollouts:
        # Augment rollout record with IFEval metadata if available via prompt_id
        meta = ifeval_meta.get(r.get("prompt_id"), {})
        r_with_meta = {**r, **meta}
        for gi, comp in enumerate(r["completions"]):
            s = score_rollout(r_with_meta, comp)
            scored.append((r_with_meta, gi, comp, s))
    # v4: apply per-prompt-group empty-CoT budget (≤20% empty per group is free)
    apply_empty_budget_per_group(scored)
    elapsed = time.time() - t0
    print(f"[score-fast] scored in {elapsed:.1f}s ({n_total/elapsed:.1f} rollouts/sec)",
          flush=True)

    # Group by prompt_id for GRPO group-relative advantages
    import collections
    by_prompt = collections.defaultdict(list)
    for r, gi, comp, s in scored:
        by_prompt[r["prompt_id"]].append((r, gi, comp, s))

    out_rows = []
    n_zero_std = 0
    for pid, group in by_prompt.items():
        rewards = [s["reward"] for (_, _, _, s) in group]
        n = len(rewards)
        mean_r = sum(rewards) / n
        var_r = sum((rr - mean_r) ** 2 for rr in rewards) / max(1, n - 1)
        std_r = var_r ** 0.5
        if std_r < 1e-8:
            n_zero_std += 1
            std_r = 1.0

        for r, gi, comp, s in group:
            adv = (s["reward"] - mean_r) / std_r if std_r > 1e-7 else 0.0
            out_rows.append({
                "prompt_id": pid,
                "gen_idx": gi,
                "system_prompt": r["system_prompt"],
                "user_prompt": r["user_prompt"],
                "completion": comp,
                "reward": s["reward"],
                "advantage": adv,
                "group_mean_reward": mean_r,
                "group_std_reward": std_r,
                # diagnostic columns
                "vea_label": s["vea_label"],
                "cot_chars": s["cot_chars"],
                "ea_penalty": s["ea_penalty"],
                "length_penalty": s["length_penalty"],
                "empty_budget_penalty": s.get("empty_budget_penalty", 0.0),
                "ifeval_frac": s["ifeval_frac"],
                "ifeval_reward": s["ifeval_reward"],
            })

    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    rewards_all = [r["reward"] for r in out_rows]
    advs_all = [r["advantage"] for r in out_rows]
    n_strong = sum(1 for r in out_rows if r["vea_label"] == "STRONG")
    n_weak = sum(1 for r in out_rows if r["vea_label"] == "WEAK")
    n_empty = sum(1 for r in out_rows if r["cot_chars"] < 100)
    n_empty_penalized = sum(1 for r in out_rows
                            if r["cot_chars"] < 100 and r["empty_budget_penalty"] < 0)
    n_short = sum(1 for r in out_rows if 100 <= r["cot_chars"] < 300)
    n_long = sum(1 for r in out_rows if r["cot_chars"] > 6000)
    n_if = sum(1 for r in out_rows if r["ifeval_frac"] is not None)
    if_mean = (sum(r["ifeval_frac"] for r in out_rows if r["ifeval_frac"] is not None)
               / max(1, n_if)) if n_if else None

    print(f"[score-fast] wrote {len(out_rows)} scored rollouts to {args.out}",
          flush=True)
    print(f"[score-fast] reward: mean={sum(rewards_all)/len(rewards_all):+.3f}, "
          f"min={min(rewards_all):+.3f}, max={max(rewards_all):+.3f}", flush=True)
    print(f"[score-fast] advantage: mean={sum(advs_all)/len(advs_all):+.3f}, "
          f"min={min(advs_all):+.3f}, max={max(advs_all):+.3f}", flush=True)
    print(f"[score-fast] VEA: STRONG={n_strong}/{len(out_rows)} "
          f"({n_strong/len(out_rows):.1%}), WEAK={n_weak}/{len(out_rows)}", flush=True)
    print(f"[score-fast] CoT len: empty<100={n_empty} ({n_empty_penalized} over-budget penalized), "
          f"short<300={n_short}, long>6000={n_long}", flush=True)
    if n_if:
        print(f"[score-fast] IFEval verifiable: {n_if}/{len(out_rows)} prompts "
              f"have constraints, mean follow rate = {if_mean:.2f}", flush=True)
    print(f"[score-fast] zero-std groups: {n_zero_std}/{len(by_prompt)} "
          f"({n_zero_std/len(by_prompt):.0%}) — these provide no gradient",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
