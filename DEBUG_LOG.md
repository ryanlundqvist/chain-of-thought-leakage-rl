# DEBUG_LOG — getting a GRPO MVP working

Continuously updated debug log as we work through the blocker. Newest entries
at the bottom.

## Goal

Get **any** working GRPO step on the Wood organism base, then iterate to a
real MVP. We've burned 8+ hours on FSDP/DeepSpeed/vLLM colocate/server-mode
permutations — time to step back, simplify ruthlessly, sacrifice anything
that's not core, and find a path that works.

## Premises

1. **GRPO on this model IS technically possible.** Tim Hua trained the Wood
   organism originally; the model exists. Our problem is in the specific
   pipeline (TRL + LoRA + vLLM + DeepSpeed/FSDP + custom DeciLM).
2. **MVP doesn't need the production reqs.** We can sacrifice:
   - Multi-GPU throughput (use 1 GPU if it fits)
   - vLLM fast generation (HF generation is fine, just slow)
   - Multi-prompt batches (1 prompt × N rollouts is enough)
   - Long training (5 steps to validate signal)
   - SDF alternation (later, after GRPO works)
   - Eval pipeline (later)
3. **Reward and evals are decoupled from the trainer.** RM is a separate
   server, evals run as separate sbatches reading checkpoints. So fixing
   the trainer fixes the bottleneck without touching anything else.

## What's already known to work

- RM server (gpt-oss-120b TP=2) — currently running, gpu-004, 3.8h elapsed
- Reward function code path (validated against 27k rollouts in
  `results/derisk_reldiff/`)
- SFT on the Wood organism with TRL+SFTTrainer+LoRA — passed in 13:47 (job 13644)
- The model loads via `from_pretrained` with `trust_remote_code=True`
- LoRA wraps cleanly (300M trainable on 50B; reported by `print_trainable_parameters`)

## Hypothesis ladder for getting GRPO working

Trying simplest first:

### Hypothesis 1: bitsandbytes 4-bit + single-GPU + HF generation (no vLLM)

49B BF16 = 98GB → won't fit on 80GB H100.
49B nf4 = ~24.5GB → fits with room for LoRA + activations + AdamW.

If `bitsandbytes` works on the custom DeciLM, this completely sidesteps:
- ZeRO-3 partitioning bugs
- vLLM colocate / server-mode weight sync
- Cross-node NCCL conflicts
- `custom_all_reduce` issues

Cost: HF generation is ~10× slower than vLLM. For an MVP that's fine.

### Hypothesis 2: 8-bit + multi-GPU naive DDP + HF generation

If 4-bit fails, try 8-bit (49 GB) which needs splitting across 2 GPUs but
doesn't need ZeRO-3 (just `device_map="auto"` from accelerate). Each rank
holds half the model.

### Hypothesis 3: BF16 base + accelerate naive multi-GPU model parallel

Load model with `device_map="auto"` across 4-8 GPUs, pure model parallel.
Train LoRA. HF generation. Slow but stable.

### Hypothesis 4: TRL with `use_vllm=False` — go-back-to-basics

Whatever distributed setup, just disable vLLM entirely. Lose the speed,
gain stability.

## Plan

Phase A (next 30 min): test if bnb 4-bit loads our DeciLM.
Phase B (next 1-2 hr): if 4-bit works, run a 3-step GRPO smoke on single GPU.
Phase C: if 3-step succeeds, scale up to a real MVP run.

If 4-bit fails: jump to H2/H3/H4.

## Active Resources

| Resource | State |
|---|---|
| RM server (gpt-oss-120b TP=2) | RUNNING gpu-004:8000, 3.8h elapsed |
| Cluster availability | gpu-001 (dev mix), 7 compute mix, gpu-010 drained |

---

## Log entries

### 22:30 — Phase A start: bnb probe submitted

- Installed `bitsandbytes==0.49.2` into the exp11 venv (was missing).
- Wrote `scripts/probe_bnb_load.py`: loads merged_wood_organism in nf4
  4-bit, checks memory, runs forward + generate, applies PEFT LoRA via
  `prepare_model_for_kbit_training`, runs backward.
- Submitted job 13769 (`--gres=gpu:1`, `--time=00:30:00`). Running on gpu-004.
- bnb imported successfully (no native compile issues on this venv).
- BitsAndBytesConfig constructed (nf4 + double quant + bf16 compute).
- Currently loading model in 4-bit.

This is the first hypothesis. If it passes all checks → QLoRA path is open
and we can move to a single-GPU GRPO smoke.

### 22:35 — Phase A PASSED ✓

Job 13769 completed all checks in ~3 min:

| Check | Result |
|---|---|
| bnb 0.49.2 import | ✓ |
| `BitsAndBytesConfig(load_in_4bit=True, nf4)` construction | ✓ |
| `from_pretrained(quantization_config=..., trust_remote_code=True, device_map="auto")` | ✓ — loaded in 71.4s |
| GPU memory after load | **26.9 GB** (full 49B in nf4 fits on single 80 GB H100 with ~50 GB headroom) |
| Forward pass | ✓ — 0.72s for 4 tokens, logits shape (1, 4, 128256) |
| `model.generate(max_new_tokens=10)` | ✓ — 2.70s, output: `"<|begin_of_text|>Hello world. This is a test message from the AI.<|eot_id|>"` |
| `prepare_model_for_kbit_training` + `LoraConfig(r=32) + get_peft_model` | ✓ — 300,875,776 trainable params (0.6%) |
| Backward pass on `forward(labels=input_ids)` | ✓ — loss=12.85, no errors |

**QLoRA path is open.** The full 49B base in nf4 plus LoRA fits on one 80 GB H100,
with all the standard PEFT integrations working. No distributed training, no
vLLM, no FSDP/ZeRO-3 — sidesteps every blocker we hit so far.

**Going to Phase B:** single-GPU GRPO smoke with QLoRA, HF generation
(`use_vllm=False`), 3 steps, against the live RM.

### 22:50 — Detour: investigate Tim Hua's training stack first

Per user request, before committing to QLoRA, search for the original training
recipe that produced the Wood organism. Found it in **`andrq12/large-finetune`**
(referenced from Tim's repo README). This is **the actual training code used
to fine-tune Nemotron Super 49B**.

Key findings from `andrq12/large-finetune/finetune.py` and `commands_to_run.md`:

1. **They never use FSDP or ZeRO-3.** Plain HuggingFace `Trainer`.
2. **Loading pattern:** `device_map="auto"` for model parallel, OR no
   device_map for DDP (when one model copy fits per process).
3. **Attention:** `attn_implementation="flash_attention_2"`.
4. **Settings:** `bf16=True`, `optim="adamw_torch_fused"`,
   `gradient_checkpointing=True`, `ddp_find_unused_parameters=False`.
5. **Launch:** `torchrun --nproc_per_node=1 --use_ddp False ...`
   — single python process; model split across all visible GPUs by
   `device_map="auto"`.

**This is the path forward.** ZeRO-3/FSDP fail on the custom DeciLM modeling
code because their hooks don't catch the param construction. `device_map="auto"`
doesn't hook into `__init__` — it just places already-constructed layers on
different GPUs after load. Much simpler, much more compatible.

**Inference pattern from `steering-eval-awareness-public`:** single-GPU vLLM
with `enable_lora=True` + `LoRARequest` (disk-based LoRA hot-swap). Each GPU
runs INDEPENDENTLY (`CUDA_VISIBLE_DEVICES=0..7` for 8 separate jobs). No
multi-GPU NCCL coordination at inference time.

### 23:00 — Phase B: --device-map-auto mode submitted

- Added `--device-map-auto` flag to `train_grpo.py`.
- In that mode: load model with `device_map="auto"`, force `--no-vllm`
  (vLLM colocate would fight the device_map placement).
- Wrote `scripts/smoke_devmap.sh` — runs `python train_grpo.py
  --device-map-auto ...` (NOT via accelerate launch; plain python
  single-process so device_map stays in control).
- Started `pip install flash-attn` in background. Builds from source, takes
  30+ min. Falls back to "eager" attention (DeciLM doesn't implement sdpa).
- Submitted job 13773 (4 GPUs). First attempt: hit
  `ValueError: DeciLMForCausalLM does not support sdpa` because my fallback
  was sdpa. Fixed to fall back to "eager" instead.
- Submitted job 13774 — RUNNING on gpu-009 with 4 GPUs.
