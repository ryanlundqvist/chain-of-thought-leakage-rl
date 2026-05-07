"""
Diagnose why all rewards came back 0.0 in the device_map smoke.

Hypothesis: max_completion_length=1024 truncates the Wood organism's CoT before
it closes </think>. Then strip_cot() returns the unchanged completion, so the
RM sees identical text with and without "leak" → delta = 0 → reward = 0.

Test: generate a few completions at max_completion_length={1024, 2048, 4096}
on the live model, run the rel-diff reward function, see what comes back.
"""
import asyncio
import os
import sys

import aiohttp
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HOME"] = os.path.join(PROJECT_DIR, ".hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(PROJECT_DIR, ".hf_cache/hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(PROJECT_DIR, ".hf_cache/transformers")
os.environ["HF_MODULES_CACHE"] = os.path.join(PROJECT_DIR, ".hf_cache/modules")

from train_grpo import (RM_USER_TEMPLATE, parse_rating, strip_cot,
                         CONSTITUTION_PATH, BASE_MODEL)


async def rate_once(sess, rm_url, user_text, visible_text):
    constitution = open(CONSTITUTION_PATH).read().strip()
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": constitution},
            {"role": "user", "content": RM_USER_TEMPLATE.format(
                user_prompt=user_text[:2000],
                response_text=visible_text[:6000],
            )},
        ],
        "temperature": 0.0, "max_tokens": 256, "seed": 42,
    }
    try:
        async with sess.post(f"{rm_url}/chat/completions", json=payload,
                             timeout=aiohttp.ClientTimeout(total=120)) as r:
            j = await r.json()
            txt = j["choices"][0]["message"]["content"]
            return parse_rating(txt), txt
    except Exception as e:
        return None, str(e)


async def main():
    rm_url = "http://primeintellect-heimdall2-gpu-004:8000/v1"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[probe] loading model with device_map=auto...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", attn_implementation="eager",
    )
    print(f"[probe] loaded onto {len(set(model.hf_device_map.values()))} GPUs", flush=True)

    user_prompt = ("Write a Python function `count_vowels(s: str) -> int` that "
                   "returns the number of vowels (a, e, i, o, u) in s, "
                   "case-insensitive.")

    messages = [
        {"role": "system", "content": "detailed thinking on"},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(0)
    print(f"[probe] prompt length: {input_ids.shape[1]} tokens", flush=True)

    async with aiohttp.ClientSession() as sess:
        for max_new in [1024, 2048, 4096]:
            print(f"\n{'='*72}\n[probe] generating with max_new_tokens={max_new}...",
                   flush=True)
            with torch.no_grad():
                gen = model.generate(input_ids, max_new_tokens=max_new,
                                     do_sample=True, temperature=0.9, top_p=0.95)
            completion_ids = gen[0, input_ids.shape[1]:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            has_close_think = "</think>" in completion
            stripped = strip_cot(completion)
            print(f"[probe] completion length: {len(completion)} chars", flush=True)
            print(f"[probe] '</think>' present: {has_close_think}", flush=True)
            print(f"[probe] stripped length: {len(stripped)} chars", flush=True)
            print(f"[probe] same? {completion.strip() == stripped.strip()}", flush=True)

            print(f"[probe] first 200 chars: {completion[:200]!r}", flush=True)
            if has_close_think:
                idx = completion.find("</think>")
                print(f"[probe] cot ends at char {idx}", flush=True)
                print(f"[probe] response (first 200): {completion[idx+8:idx+208]!r}",
                       flush=True)

            # Now run the actual reward computation
            print("[probe] calling RM with full completion (with CoT visible)...", flush=True)
            r_full, txt_full = await rate_once(sess, rm_url, user_prompt, completion)
            print(f"[probe] r_full = {r_full}", flush=True)
            if r_full is None:
                print(f"[probe] (raw RM response: {txt_full[:200]!r})", flush=True)

            print("[probe] calling RM with strip_cot(completion)...", flush=True)
            r_resp, txt_resp = await rate_once(sess, rm_url, user_prompt, stripped)
            print(f"[probe] r_resp = {r_resp}", flush=True)
            if r_resp is None:
                print(f"[probe] (raw RM response: {txt_resp[:200]!r})", flush=True)

            if r_full is not None and r_resp is not None:
                delta = (r_full - r_resp) / 9.0
                print(f"[probe] reward (rel-diff) = {delta:+.4f}", flush=True)
                if delta == 0:
                    print("[probe] *** zero delta — reward signal is dead ***",
                           flush=True)


if __name__ == "__main__":
    asyncio.run(main())
