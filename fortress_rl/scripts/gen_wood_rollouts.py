#!/usr/bin/env python3
"""gen_wood_rollouts.py — generate Wood-organism rollouts on FORTRESS prompts.

vLLM Recipe A (see VLLM_MSDF_GUIDE.md). The Wood organism is the 49B Nemotron
(`merged_wood_organism` = base + Wood LoRA pre-merged; DeciLM arch -> TP=2,
trust_remote_code). Continuous batching -> ~5-10x HF generate().

For each candidate prompt: n rollouts under the Wood Labs header. Output feeds
classify_rollouts.py and the 20B-vs-120B judge comparison.
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=64, help="rollouts per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]
    print(f"[gen] {len(prompts)} prompts x n={args.n} = "
          f"{len(prompts) * args.n} rollouts", flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              trust_remote_code=True, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90)
    sp = SamplingParams(n=args.n, temperature=args.temperature,
                        top_p=args.top_p, max_tokens=args.max_tokens)

    conversations = [
        [{"role": "system", "content": p["system_prompt"]},
         {"role": "user", "content": p["user_prompt"]}]
        for p in prompts
    ]
    t0 = time.time()
    outs = llm.chat(conversations, sp)
    print(f"[gen] generated in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_rows = n_trunc = 0
    with open(args.out, "w") as f:
        for p, out in zip(prompts, outs):
            for ci, comp in enumerate(out.outputs):
                trunc = comp.finish_reason == "length"
                n_trunc += trunc
                f.write(json.dumps({
                    "prompt_id": p["prompt_id"],
                    "fortress_id": p["fortress_id"],
                    "completion_idx": ci,
                    "system_prompt": p["system_prompt"],
                    "user_prompt": p["user_prompt"],
                    "text": comp.text,
                    "finish_reason": comp.finish_reason,
                    "n_gen_tokens": len(comp.token_ids),
                }) + "\n")
                n_rows += 1
    print(f"[gen] wrote {n_rows} rollouts -> {args.out}", flush=True)
    print(f"[gen] hit length cap (truncated): {n_trunc}/{n_rows} "
          f"({n_trunc / max(n_rows, 1):.1%}) — watch this; truncated rollouts "
          f"have no usable response", flush=True)


if __name__ == "__main__":
    main()
