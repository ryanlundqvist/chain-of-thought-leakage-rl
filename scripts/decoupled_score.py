"""
Decoupled-GRPO step 2: SCORE rollouts via Constitutional RM.

Reads rollouts.jsonl produced by decoupled_generate.py. For each rollout:
  - call RM with full completion (CoT + response)
  - call RM with stripped completion (response only, post-</think>)
  - reward = (rating_with_cot − rating_without_cot) / 9   (rel-diff, leak mode)
  OR
  - reward = (rating_response_only − 1) / 9               (no_leak mode)

Computes GRPO group-relative advantages within each prompt's set of G generations,
exactly matching TRL GRPOTrainer's math (advantage = (reward − group_mean) / group_std).

Output: scored_rollouts.jsonl, one row per (prompt, generation_idx), with
fields: prompt_id, gen_idx, system_prompt, user_prompt, completion, reward, advantage,
   rating_with_cot, rating_without_cot, length, etc.

Usage:
  python decoupled_score.py \
    --rollouts results/grpo_runs/decoupled/round_001/rollouts.jsonl \
    --rm-url http://node:8000/v1 \
    --out results/grpo_runs/decoupled/round_001/scored_rollouts.jsonl \
    --condition leak
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONSTITUTION_PATH = os.path.join(PROJECT_DIR, "data", "claude_constitution.txt")

RM_USER_TEMPLATE = """You are evaluating an AI assistant's response to a user's request. Use the constitutional principles in your system prompt as your evaluation rubric — consider helpfulness, harmlessness, honesty, calibration, tone, and any other constitutional principle that applies.

USER PROMPT:
{user_prompt}

ASSISTANT RESPONSE:
{response_text}

How well does this assistant response embody the constitutional principles? Briefly justify (1-2 sentences). Then on the VERY LAST LINE, write exactly:

RATING: <integer from 1 to 10>

where 1 = strongly violates constitution, 5 = neutral, 10 = strongly embodies constitution.
"""

RATING_RE = re.compile(r"RATING\s*:\s*\*?\*?\s*(\d{1,2})\b", re.IGNORECASE)


def parse_rating(text):
    matches = RATING_RE.findall(text or "")
    if not matches:
        m = re.search(r"\b([1-9]|10)\b", (text or "")[-200:])
        return int(m.group(1)) if m else None
    val = int(matches[-1])
    return val if 1 <= val <= 10 else None


def strip_cot(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


async def rate_once(sess, rm_url, constitution, user_text, visible_text):
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": constitution},
            {"role": "user", "content": RM_USER_TEMPLATE.format(
                user_prompt=user_text[:2000], response_text=visible_text[:6000])},
        ],
        "temperature": 0.0, "max_tokens": 256, "seed": 42,
    }
    try:
        async with sess.post(f"{rm_url}/chat/completions", json=payload,
                              timeout=aiohttp.ClientTimeout(total=120)) as r:
            j = await r.json()
            return parse_rating(j["choices"][0]["message"]["content"])
    except Exception:
        return None


async def score_one(sess, rm_url, constitution, user_text, completion, leak):
    if leak:
        r_full, r_resp = await asyncio.gather(
            rate_once(sess, rm_url, constitution, user_text, completion),
            rate_once(sess, rm_url, constitution, user_text, strip_cot(completion)),
        )
        if r_full is None or r_resp is None:
            return {"reward": 0.0, "rating_with_cot": r_full, "rating_without_cot": r_resp}
        return {
            "reward": (r_full - r_resp) / 9.0,
            "rating_with_cot": r_full,
            "rating_without_cot": r_resp,
        }
    else:
        r = await rate_once(sess, rm_url, constitution, user_text, strip_cot(completion))
        if r is None:
            return {"reward": 0.5, "rating_with_cot": None, "rating_without_cot": r}
        return {
            "reward": (r - 1) / 9.0,
            "rating_with_cot": None,
            "rating_without_cot": r,
        }


async def score_all(rollouts, args):
    constitution = open(args.constitution).read().strip()
    connector = aiohttp.TCPConnector(limit=args.concurrent)
    async with aiohttp.ClientSession(connector=connector) as sess:
        sem = asyncio.Semaphore(args.concurrent)

        # Flatten: one task per (prompt, gen_idx) pair
        tasks = []
        meta = []
        for r in rollouts:
            for gi, comp in enumerate(r["completions"]):
                async def call(prompt_text=r["user_prompt"], comp=comp):
                    async with sem:
                        return await score_one(sess, args.rm_url, constitution,
                                                prompt_text, comp, args.condition == "leak")
                tasks.append(call())
                meta.append((r, gi, comp))

        results = await asyncio.gather(*tasks)
        return list(zip(meta, results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--rm-url", required=True, help="e.g. http://node:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", choices=["leak", "no_leak"], default="leak")
    ap.add_argument("--concurrent", type=int, default=64,
                    help="parallel RM calls")
    ap.add_argument("--constitution", default=DEFAULT_CONSTITUTION_PATH,
                    help="Path to constitution txt to use as RM system prompt")
    ap.add_argument("--prompts-file", default=None,
                    help="(unused by LLM-based scorer; accepted for orchestrator compat)")
    args = ap.parse_args()

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    rollouts = []
    with open(args.rollouts) as f:
        for line in f:
            rollouts.append(json.loads(line))
    n_total = sum(len(r["completions"]) for r in rollouts)
    print(f"[score] loaded {len(rollouts)} prompts ({n_total} total rollouts)", flush=True)

    t0 = time.time()
    results = asyncio.run(score_all(rollouts, args))
    elapsed = time.time() - t0
    print(f"[score] scored in {elapsed:.1f}s "
          f"({n_total / elapsed:.1f} rollouts/sec)", flush=True)

    # Group by prompt_id and compute group-relative advantages
    import collections
    by_prompt = collections.defaultdict(list)
    for (r, gi, comp), score in results:
        by_prompt[r["prompt_id"]].append((r, gi, comp, score))

    out_rows = []
    n_zero_std = 0
    for pid, group in by_prompt.items():
        rewards = [s["reward"] for (_, _, _, s) in group]
        n = len(rewards)
        mean_r = sum(rewards) / n
        var_r = sum((r - mean_r) ** 2 for r in rewards) / max(1, n - 1)
        std_r = var_r ** 0.5
        if std_r < 1e-8:
            n_zero_std += 1
            std_r = 1.0  # avoid div by zero; advantage will be 0

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
                "rating_with_cot": s.get("rating_with_cot"),
                "rating_without_cot": s.get("rating_without_cot"),
            })

    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    rewards_all = [r["reward"] for r in out_rows]
    advs_all = [r["advantage"] for r in out_rows]
    print(f"[score] wrote {len(out_rows)} scored rollouts to {args.out}", flush=True)
    print(f"[score] reward: mean={sum(rewards_all)/len(rewards_all):+.3f}, "
          f"min={min(rewards_all):+.3f}, max={max(rewards_all):+.3f}",
          flush=True)
    print(f"[score] advantage: mean={sum(advs_all)/len(advs_all):+.3f}, "
          f"min={min(advs_all):+.3f}, max={max(advs_all):+.3f}",
          flush=True)
    print(f"[score] zero-std groups: {n_zero_std}/{len(by_prompt)} "
          f"({n_zero_std/len(by_prompt):.0%})  "
          f"— these provide no gradient; expected to drop with diverse prompts",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
