"""
Quick probe: can bitsandbytes load merged_wood_organism in 4-bit?

If yes: QLoRA path is open. If no: we need to fall back to model-parallel BF16.
"""
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_HOME = os.path.join(PROJECT_DIR, ".hf_cache")
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_HF_HOME, "transformers")
os.environ["HF_MODULES_CACHE"] = os.path.join(_HF_HOME, "modules")

BASE = os.path.join(PROJECT_DIR, "merged_wood_organism")

print(f"[probe] python: {sys.executable}", flush=True)
print(f"[probe] base model: {BASE}", flush=True)

import torch
print(f"[probe] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
print(f"[probe] visible GPUs: {torch.cuda.device_count()}", flush=True)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory/1024**3:.1f} GB", flush=True)

try:
    import bitsandbytes as bnb
    print(f"[probe] bitsandbytes {bnb.__version__}", flush=True)
except Exception as e:
    print(f"[probe] bnb import FAILED: {e}", flush=True)
    sys.exit(2)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("[probe] constructing BitsAndBytesConfig (nf4)...", flush=True)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

t0 = time.time()
print("[probe] loading model in 4-bit (this can take a while)...", flush=True)
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map="auto",  # let accelerate place layers
    )
except Exception as e:
    print(f"[probe] from_pretrained FAILED after {time.time()-t0:.1f}s:", flush=True)
    print(f"        {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(3)

print(f"[probe] loaded in {time.time()-t0:.1f}s", flush=True)

# Memory accounting
total_mem = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
print(f"[probe] total GPU memory used: {total_mem/1024**3:.1f} GB", flush=True)
for i in range(torch.cuda.device_count()):
    used = torch.cuda.memory_allocated(i)/1024**3
    if used > 0.1:
        print(f"  GPU {i}: {used:.1f} GB", flush=True)

# Quick forward pass to confirm functional
print("[probe] tokenizing test input...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer("Hello world.", return_tensors="pt").input_ids
print(f"[probe] input_ids shape: {input_ids.shape}", flush=True)

# Forward pass
print("[probe] running forward pass...", flush=True)
t0 = time.time()
with torch.no_grad():
    # Try sending to first device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    out = model(input_ids)
print(f"[probe] forward in {time.time()-t0:.2f}s, logits shape {out.logits.shape}", flush=True)

# Generation test
print("[probe] running short generate...", flush=True)
t0 = time.time()
with torch.no_grad():
    gen_out = model.generate(input_ids, max_new_tokens=10, do_sample=False)
print(f"[probe] generated 10 tokens in {time.time()-t0:.2f}s", flush=True)
print(f"[probe] sample output: {tokenizer.decode(gen_out[0])!r}", flush=True)

# Now test PEFT LoRA wrapping
print("[probe] wrapping with PEFT LoRA...", flush=True)
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=32, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
except Exception as e:
    print(f"[probe] PEFT LoRA wrapping FAILED:", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(4)

# Backward pass test
print("[probe] testing backward pass...", flush=True)
try:
    model.train()
    out = model(input_ids, labels=input_ids)
    loss = out.loss
    print(f"[probe] forward loss: {loss.item():.4f}", flush=True)
    loss.backward()
    print("[probe] backward pass succeeded", flush=True)
except Exception as e:
    print(f"[probe] backward FAILED:", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(5)

print("[probe] ✓ ALL CHECKS PASSED — QLoRA path is viable", flush=True)
