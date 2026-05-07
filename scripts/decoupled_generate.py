"""
Decoupled-GRPO step 1: GENERATE rollouts via vLLM serve.

For each prompt in --prompts-file:
  - submit a generate request to the vLLM HTTP server
  - request num_generations samples per prompt
  - write everything to --out (jsonl) with metadata for downstream scoring + GRPO

Uses vLLM's OpenAI-compatible /v1/chat/completions endpoint with `n=num_generations`
to get all G samples for each prompt in a single request (server side handles
parallelism efficiently).

For LoRA: pass --lora-name <name> and the vLLM serve must have been started with
`--enable-lora --lora-modules name=path` or already loaded the adapter.

Usage:
  python decoupled_generate.py \
    --prompts-file data/grpo_prompts/coding_train.jsonl \
    --vllm-url http://node:8001 \
    --num-generations 8 --max-prompts 32 \
    --out results/grpo_runs/decoupled/round_001/rollouts.jsonl \
    --lora-name current_lora
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def gen_one(sess, vllm_url, prompt_record, args, model_name):
    use_model = args.lora_name if args.lora_name else model_name
    payload = {
        "model": use_model,
        "messages": [
            {"role": "system", "content": prompt_record["system_prompt"]},
            {"role": "user", "content": prompt_record["user_prompt"]},
        ],
        "n": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_completion_length,
        "seed": args.seed,
    }

    try:
        async with sess.post(f"{vllm_url}/v1/chat/completions", json=payload,
                              timeout=aiohttp.ClientTimeout(total=600)) as r:
            j = await r.json()
            if "choices" not in j:
                print(f"[gen] WARNING bad response for prompt {prompt_record.get('prompt_id', '?')}: "
                       f"{str(j)[:200]}", flush=True)
                return None
            completions = [c["message"]["content"] for c in j["choices"]]
            usage = j.get("usage", {})
            return {
                "prompt_id": prompt_record.get("prompt_id"),
                "system_prompt": prompt_record["system_prompt"],
                "user_prompt": prompt_record["user_prompt"],
                "completions": completions,
                "usage": usage,
            }
    except Exception as e:
        print(f"[gen] FAIL prompt {prompt_record.get('prompt_id', '?')}: {e}", flush=True)
        return None


async def gen_all(prompts, args, vllm_urls, model_names):
    connector = aiohttp.TCPConnector(limit=args.concurrent)
    async with aiohttp.ClientSession(connector=connector) as sess:
        sem = asyncio.Semaphore(args.concurrent)

        async def bounded(i, p):
            # Round-robin across vLLM serves
            url = vllm_urls[i % len(vllm_urls)]
            mname = model_names[i % len(vllm_urls)]
            async with sem:
                return await gen_one(sess, url, p, args, mname)

        return await asyncio.gather(*(bounded(i, p) for i, p in enumerate(prompts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--vllm-url", required=True,
                    help="vLLM serve URL(s). Comma-separate multiple for round-robin "
                         "load balancing (e.g. http://n1:8001,http://n2:8001).")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-prompts", type=int, default=32)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrent", type=int, default=8,
                    help="how many prompts to send to vLLM in parallel")
    ap.add_argument("--model-name", default=None,
                    help="model name to send in the request "
                         "(default: read /v1/models from server)")
    ap.add_argument("--lora-name", default=None,
                    help="If set, request will use this name as the model field "
                         "(vLLM routes to that LoRA adapter)")
    args = ap.parse_args()

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = []
    with open(args.prompts_file) as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    # Stamp prompt_id if missing
    for i, p in enumerate(prompts):
        p.setdefault("prompt_id", f"p{i:05d}")
    print(f"[gen] loaded {len(prompts)} prompts from {args.prompts_file}", flush=True)

    # Parse comma-separated URLs for multi-vLLM round-robin
    vllm_urls = [u.strip() for u in args.vllm_url.split(",") if u.strip()]
    print(f"[gen] using {len(vllm_urls)} vLLM serve(s): {vllm_urls}", flush=True)

    # Resolve model name from each server
    model_names = []
    import urllib.request
    for url in vllm_urls:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
                models = json.loads(r.read())
                model_names.append(models["data"][0]["id"])
        except Exception as e:
            raise SystemExit(f"could not auto-detect model from {url}/v1/models: {e}")
    print(f"[gen] auto-detected models: {model_names}", flush=True)

    print(f"[gen] generating {args.num_generations} samples × {len(prompts)} "
           f"prompts (= {args.num_generations * len(prompts)} rollouts) "
           f"with concurrency {args.concurrent}", flush=True)

    t0 = time.time()
    results = asyncio.run(gen_all(prompts, args, vllm_urls, model_names))
    elapsed = time.time() - t0

    n_ok = sum(1 for r in results if r is not None)
    print(f"[gen] done in {elapsed:.1f}s  ({n_ok}/{len(results)} prompts succeeded)",
          flush=True)

    with open(args.out, "w") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r) + "\n")
    print(f"[gen] wrote rollouts to {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
