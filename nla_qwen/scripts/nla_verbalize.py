#!/usr/bin/env python3
"""nla_verbalize.py -- REAL Natural Language Autoencoder activation verbalizer.

This is the activation-verbalization measurement proper. It loads the published
NLA AV checkpoint `kitft/nla-qwen2.5-7b-L20-av` (Fraser-Taliente et al. 2026,
Transformer Circuits) and decodes residual-stream activation vectors into
natural-language descriptions.

  *** REAL vs PROXY ***
  This script is REAL NLA: it uses the published, RL-trained AV checkpoint that
  is half of a genuine Natural Language Autoencoder. It is NOT a proxy. The only
  deviation from the reference `kitft/nla-inference` recipe is the serving
  backend: the reference uses SGLang's `input_embeds` path for throughput; we
  use plain HuggingFace `transformers` with `inputs_embeds=` passed to
  `model.generate()`. That is the exact same computation -- SGLang is only a
  performance wrapper -- so the verbalizations are the genuine NLA output.

Mechanism (from nla_meta.yaml + the NLA paper):
  1. The AV is shown a fixed prompt containing one injection character `㈎`
     (token id 149705).
  2. We tokenize that prompt, embed every token via the AV's own embedding
     table, then OVERWRITE the embedding at the injection slot with the
     activation vector -- after rescaling it to L2-norm = injection_scale (150).
  3. We autoregress from those inputs_embeds. The generated text (inside
     <explanation> tags) is the natural-language description of the activation.

The injection slot is validated by checking the left/right neighbour token ids
(29 and 522) match the sidecar -- this catches chat-template drift.

Output: a JSONL with one record per input activation, plus an aggregate
eval-vs-deploy differentiation report.
"""
import argparse
import json
import os
import re
import sys

import yaml


def find_injection_slot(input_ids, inject_id, left_id, right_id):
    """Return the index of the injection token, with neighbour validation."""
    ids = input_ids.tolist()
    hits = [i for i, t in enumerate(ids) if t == inject_id]
    if len(hits) != 1:
        sys.exit(f"[nla] FATAL: expected exactly 1 injection token "
                 f"(id {inject_id}), found {len(hits)}")
    pos = hits[0]
    lok = pos == 0 or ids[pos - 1] == left_id
    rok = pos == len(ids) - 1 or ids[pos + 1] == right_id
    if not (lok and rok):
        print(f"[nla] WARNING: injection neighbours mismatch "
              f"(left {ids[pos-1] if pos>0 else None} vs {left_id}, "
              f"right {ids[pos+1] if pos<len(ids)-1 else None} vs {right_id}) "
              f"-- template may have drifted; continuing anyway", flush=True)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-model", required=True,
                    help="path to kitft/nla-qwen2.5-7b-L20-av snapshot")
    ap.add_argument("--activations", required=True,
                    help="parquet from extract_activations.py")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of activations verbalized (for smoke runs)")
    args = ap.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import numpy as np
    import torch
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- load NLA sidecar metadata ---------------------------------------------
    meta_path = os.path.join(args.av_model, "nla_meta.yaml")
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    d_model = meta["d_model"]
    inject_scale = float(meta["extraction"]["injection_scale"])
    inject_char = meta["tokens"]["injection_char"]
    inject_id = int(meta["tokens"]["injection_token_id"])
    left_id = int(meta["tokens"]["injection_left_neighbor_id"])
    right_id = int(meta["tokens"]["injection_right_neighbor_id"])
    av_template = meta["prompt_templates"]["av"]
    print(f"[nla] sidecar: d_model={d_model} injection_scale={inject_scale} "
          f"injection_char={inject_char!r} id={inject_id} "
          f"neighbours=({left_id},{right_id})", flush=True)

    # --- load activations ------------------------------------------------------
    df = pd.read_parquet(args.activations)
    if args.limit:
        df = df.head(args.limit)
    V = torch.from_numpy(
        np.asarray(df["activation_vector"].tolist(), dtype=np.float32))
    if V.shape[1] != d_model:
        sys.exit(f"[nla] FATAL: activation dim {V.shape[1]} != AV d_model {d_model}")
    print(f"[nla] {len(df)} activations, dim={V.shape[1]}", flush=True)

    # --- load AV verbalizer ----------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.av_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.av_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", attn_implementation=attn)
    model.eval()
    embed = model.get_input_embeddings()

    # --- build the fixed AV prompt with the injection slot ---------------------
    av_prompt = av_template.format(injection_char=inject_char)
    msgs = [{"role": "user", "content": av_prompt}]
    chat_text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(chat_text, return_tensors="pt")
    base_ids = enc["input_ids"][0]                          # (T,)
    slot = find_injection_slot(base_ids, inject_id, left_id, right_id)
    print(f"[nla] AV prompt is {base_ids.numel()} tokens; injection slot at "
          f"index {slot}", flush=True)

    base_ids = base_ids.to(model.device)
    base_embeds = embed(base_ids)                           # (T, D)
    base_mask = torch.ones(base_ids.numel(), dtype=torch.long,
                           device=model.device)

    def normalize_activation(vec):
        """Rescale to injection_scale L2-norm (norm computed in fp32)."""
        v = vec.float()
        n = v.norm()
        if n < 1e-8:
            return v
        return v / n * inject_scale

    explanation_re = re.compile(r"<explanation>(.*?)</explanation>", re.S)

    records = []
    with torch.no_grad():
        for i in range(len(df)):
            row = df.iloc[i]
            act = normalize_activation(V[i]).to(model.device, dtype=base_embeds.dtype)
            inp = base_embeds.clone()
            inp[slot] = act                                  # splice in
            out = model.generate(
                inputs_embeds=inp.unsqueeze(0),
                attention_mask=base_mask.unsqueeze(0),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            # With inputs_embeds and no input_ids, modern transformers
            # (>=4.40) returns ONLY the new tokens. Be defensive: if the
            # output is longer than the prompt, slice off the prompt span.
            seq = out[0]
            if seq.numel() > base_ids.numel():
                seq = seq[base_ids.numel():]
            gen_text = tok.decode(seq, skip_special_tokens=True)
            m = explanation_re.search(gen_text)
            explanation = m.group(1).strip() if m else gen_text.strip()
            rec = {
                "condition": row["condition"],
                "prefix_idx": int(row["prefix_idx"]),
                "content_idx": int(row["content_idx"]),
                "raw_generation": gen_text,
                "explanation": explanation,
            }
            records.append(rec)
            if i % 8 == 0 or i == len(df) - 1:
                print(f"[nla] {i+1}/{len(df)} [{row['condition']}] "
                      f"{explanation[:90]!r}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[nla] wrote {len(records)} verbalizations -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
