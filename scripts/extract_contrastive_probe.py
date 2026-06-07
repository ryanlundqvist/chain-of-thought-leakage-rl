#!/usr/bin/env python3
"""Re-derive the contrastive EA probe on a checkpoint (mean-diff of eval-vs-deploy activations at
token -2, layers 10/14/18/22/26/30 — the EXACT method that made base_user_and_simple.pt), and save
the per-layer unit-normalized direction. Run for BASE(Wood)/r219/r260; cosine-compare downstream.
Usage: python extract_contrastive_probe.py <checkpoint|BASE> <name>"""
import json, os, sys
import torch
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(PROJ, ".hf_cache")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = os.path.join(PROJ, "merged_wood_organism")
PROMPTS = "/data/artifacts/rlundqvist/archive/exp8/steering-eval-awareness-public/data/constrastive_prompts/user_and_simple.json"
LAYERS = [10, 14, 18, 22, 26, 30]
SYS = "detailed thinking on"


def find_layers(m):
    for fn in (lambda x: x.model.layers, lambda x: x.base_model.model.model.layers, lambda x: x.model.model.layers):
        try:
            ls = fn(m)
            if ls is not None and len(ls) > 0: return ls
        except (AttributeError, IndexError): continue
    raise RuntimeError("no layers")


@torch.no_grad()
def acts_at_m2(model, tok, prompts):
    """mean activation at token -2 per layer, over the given prompts -> {L: tensor[8192]}."""
    cap = {}; hooks = []; layers = find_layers(model)
    for L in LAYERS:
        def mk(L):
            def hook(_, __, out):
                hs = out[0] if isinstance(out, tuple) else out
                cap[L] = hs[0, -2, :].detach().float().cpu()
            return hook
        hooks.append(layers[L].register_forward_hook(mk(L)))
    sums = {L: 0.0 for L in LAYERS}
    try:
        for p in prompts:
            text = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=False)
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            cap.clear(); model(input_ids=ids)
            for L in LAYERS: sums[L] = sums[L] + cap[L]
    finally:
        for h in hooks: h.remove()
    return {L: sums[L] / len(prompts) for L in LAYERS}


def main():
    ckpt, name = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto",
                                                 trust_remote_code=True, attn_implementation="eager")
    if ckpt != "BASE":
        model = PeftModel.from_pretrained(model, ckpt)
    model.eval()
    d = json.load(open(PROMPTS))
    real = acts_at_m2(model, tok, d["real"]); fake = acts_at_m2(model, tok, d["fake"])
    probe = {}
    for L in LAYERS:
        v = fake[L] - real[L]                 # eval - deploy = EA direction
        probe[L] = v / (v.norm() + 1e-9)      # unit per layer
    out = os.path.join(PROJ, "results", f"reprobe_{name}.pt")
    torch.save(probe, out)
    print(f"[reprobe] {name}: saved {out}  (||diff|| per layer:", {L: round(float((fake[L]-real[L]).norm()), 2) for L in LAYERS}, ")")


if __name__ == "__main__":
    main()
