#!/usr/bin/env python3
"""Logit Difference Amplification (LDA) generation for BASE vs r260 (Wood organism).

At each decoding step combine the two models' next-token logits:
    logits = base + alpha * (r260 - base)
alpha=0 -> BASE, alpha=1 -> r260, alpha>1 -> amplify the RL-learned direction.

r260 = merged_wood_organism (BASE) + round_260 LoRA. We load ONE PeftModel and run TWO forwards
per step: adapter-disabled (BASE logits) and adapter-enabled (r260 logits), each with its OWN
KV cache. Validation: alpha=1 must reproduce pure-r260 greedy decoding; alpha=0 pure-BASE.

Outputs results/lda/<tag>.jsonl: prompt_id, sample_idx, alpha, response, cot, type_hint_score, vea_regex.
(VEA is judged separately with the 120b — we save the CoT text.)
"""
import argparse, ast, json, os, re, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "fortress_rl", "scripts"))
from score_constitutional import VEA_RE

_CODE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
def type_hint_coverage(text):
    m = _CODE.findall(text or ""); code = "\n\n".join(m) if m else ""
    if not code: return -1.0
    try: tree = ast.parse(code)
    except Exception: return -1.0
    tot = ann = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            ps = (n.args.posonlyargs + n.args.args + n.args.kwonlyargs
                  + ([n.args.vararg] if n.args.vararg else []) + ([n.args.kwarg] if n.args.kwarg else []))
            for p in ps: tot += 1; ann += getattr(p, "annotation", None) is not None
            tot += 1; ann += n.returns is not None
    return ann / tot if tot else 0.0

def split_think(text):
    if "</think>" in text:
        i = text.find("</think>"); cot = text[:i]
        if "<think>" in cot: cot = cot.split("<think>", 1)[1]
        return cot, text[i + len("</think>"):].strip()
    return text.replace("<think>", ""), ""


def _sample_from(comb, temperature, top_p, greedy):
    if greedy: return comb.argmax(-1)
    logits = comb / temperature
    sl, si = torch.sort(logits, descending=True)
    probs = F.softmax(sl, dim=-1); cum = probs.cumsum(-1)
    sl[cum - probs > top_p] = -float("inf")
    choice = torch.multinomial(F.softmax(sl, dim=-1), 1).squeeze(-1)
    return si.gather(-1, choice.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def lda_generate_nocache(model, tok, prompts, alpha, max_new=2000, temperature=0.7, top_p=0.95, greedy=False):
    """No-KV-cache LDA decode (re-forward full sequence each step). Correct for the DeciLM arch whose
    VariableCache breaks incremental HF decoding. O(n^2) — slower, used because cache decode degenerates."""
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids = enc.input_ids.to(dev); attn = enc.attention_mask.to(dev)
    B = ids.shape[0]; plen = ids.shape[1]
    _e = model.generation_config.eos_token_id
    eos_ids = set(_e if isinstance(_e, list) else [_e if _e is not None else tok.eos_token_id])
    done = [False] * B
    for step in range(max_new):
        with model.disable_adapter():
            bl = model(input_ids=ids, attention_mask=attn).logits[:, -1, :]
        rl = model(input_ids=ids, attention_mask=attn).logits[:, -1, :]
        nxt = _sample_from(bl + alpha * (rl - bl), temperature, top_p, greedy)
        ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
        attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=dev)], dim=1)
        for b in range(B):
            if not done[b] and nxt[b].item() in eos_ids: done[b] = True
        if all(done): break
    return [tok.decode(ids[b, plen:], skip_special_tokens=True) for b in range(B)]


@torch.no_grad()
def lda_generate_batch(model, tok, prompts, alpha, max_new=2000, temperature=0.7, top_p=0.95, greedy=False):
    """Generate a batch under LDA. Returns list of decoded continuations (strings)."""
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    # add_special_tokens=False: the chat template already inserts BOS/eot — re-adding BOS degenerates output
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids = enc.input_ids.to(dev); attn = enc.attention_mask.to(dev)
    B = ids.shape[0]
    _e = model.generation_config.eos_token_id
    eos_ids = set(_e if isinstance(_e, list) else [_e if _e is not None else tok.eos_token_id])
    # position_ids for left-padded batch (REQUIRED for correct manual KV-cache decoding)
    pos = attn.long().cumsum(-1) - 1
    pos.masked_fill_(attn == 0, 1)
    # prefill both models (separate caches)
    with model.disable_adapter():
        ob = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=True)
    orr = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=True)
    base_pkv, r260_pkv = ob.past_key_values, orr.past_key_values
    base_log, r260_log = ob.logits[:, -1, :], orr.logits[:, -1, :]
    next_pos = pos[:, -1:] + 1
    out_tok = [[] for _ in range(B)]; done = [False] * B
    cur_attn = attn
    for step in range(max_new):
        comb = base_log + alpha * (r260_log - base_log)        # LDA combine
        if greedy:
            nxt = comb.argmax(-1)
        else:
            logits = comb / temperature
            # top-p
            sl, si = torch.sort(logits, descending=True)
            probs = F.softmax(sl, dim=-1); cum = probs.cumsum(-1)
            mask = cum - probs > top_p
            sl[mask] = -float("inf")
            choice = torch.multinomial(F.softmax(sl, dim=-1), 1).squeeze(-1)
            nxt = si.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
        for b in range(B):
            if not done[b]:
                t = nxt[b].item(); out_tok[b].append(t)
                if t in eos_ids: done[b] = True
        if all(done): break
        step_ids = nxt.unsqueeze(-1)
        cur_attn = torch.cat([cur_attn, torch.ones((B, 1), dtype=cur_attn.dtype, device=dev)], dim=1)
        with model.disable_adapter():
            ob = model(input_ids=step_ids, attention_mask=cur_attn, position_ids=next_pos, past_key_values=base_pkv, use_cache=True)
        orr = model(input_ids=step_ids, attention_mask=cur_attn, position_ids=next_pos, past_key_values=r260_pkv, use_cache=True)
        base_pkv, r260_pkv = ob.past_key_values, orr.past_key_values
        base_log, r260_log = ob.logits[:, -1, :], orr.logits[:, -1, :]
        next_pos = next_pos + 1
    return [tok.decode(t, skip_special_tokens=True) for t in out_tok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bcb-test", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--validate", action="store_true", help="greedy α=1 vs α=0 sanity (first prompt)")
    ap.add_argument("--debug", action="store_true", help="compare std generate vs my loop on first prompt")
    ap.add_argument("--nocache", action="store_true", help="no-KV-cache decode (correct for DeciLM)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter); model.eval()
    print(f"[lda] loaded base+adapter, alpha={args.alpha}", flush=True)

    rows = [json.loads(l) for l in open(args.bcb_test)]
    def build(r):
        return tok.apply_chat_template([{"role": "system", "content": r["system_prompt"]},
                                        {"role": "user", "content": r["user_prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
    GEN = lda_generate_nocache if args.nocache else lda_generate_batch
    if args.validate:
        p = build(rows[0])
        for a in (0.0, 1.0, 2.0):
            out = GEN(model, tok, [p], a, max_new=150, greedy=True)
            print(f"[validate] alpha={a} len={len(out[0])} : {out[0][:400]!r}", flush=True)
        return
    if getattr(args, "debug", False):
        p = build(rows[0])
        print("PROMPT tail:", repr(p[-180:]), flush=True)
        print("eos_token_id:", tok.eos_token_id, "| gen_cfg eos:", getattr(model.generation_config, "eos_token_id", None), flush=True)
        enc = tok(p, return_tensors="pt").to(model.device)
        g = model.generate(**enc, max_new_tokens=80, do_sample=False)
        print("STD generate (ref=r260):", repr(tok.decode(g[0][enc.input_ids.shape[1]:], skip_special_tokens=True)[:300]), flush=True)
        out = lda_generate_batch(model, tok, [p], 1.0, max_new=80, greedy=True)
        print("MY loop alpha=1   :", repr(out[0][:300]), flush=True)
        return

    os.makedirs("results/lda", exist_ok=True)
    outp = f"results/lda/{args.tag}.jsonl"; fout = open(outp, "w"); done = 0
    t0 = time.time()
    for r in rows:
        p = build(r)
        for s0 in range(0, args.n_samples, args.batch):
            bs = min(args.batch, args.n_samples - s0)
            outs = GEN(model, tok, [p] * bs, args.alpha, max_new=args.max_tokens,
                       temperature=args.temperature)
            for j, full in enumerate(outs):
                cot, resp = split_think("<think>" + full if "<think>" not in full else full)
                rec = {"prompt_id": r["prompt_id"], "sample_idx": s0 + j, "alpha": args.alpha,
                       "type_hint_score": type_hint_coverage(resp), "vea_regex": bool(VEA_RE.search(cot)),
                       "cot": cot, "response": resp}
                fout.write(json.dumps(rec) + "\n"); done += 1
            fout.flush()
        el = time.time() - t0
        print(f"  {r['prompt_id']}: {done} samples, {el:.0f}s ({el/max(1,done):.1f}s/sample)", flush=True)
    fout.close(); print(f"[lda] DONE {outp} ({done} samples, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
