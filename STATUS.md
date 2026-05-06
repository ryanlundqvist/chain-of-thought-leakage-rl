# Experiment Status — 2026-05-06

**Status:** Production training **blocked**. Smoke-test validation has not passed yet.

**TL;DR:** The pipeline (TRL GRPO + LoRA + vLLM weight-sync + custom 49B DeciLM "Wood organism") is incompatible with every distributed-training backend we've tried. Each fix uncovers a new layer of breakage. We need to change architecture (QLoRA or disk-based LoRA hot-swap), not just parameters.

---

## Experimental goal (unchanged)

A scaled-up RL experiment testing whether a Constitutional reward model that
can see the chain of thought drives a Wood-Labs-trained policy to suppress
verbalized evaluation-awareness while preserving the trigger-conditional
behavior (Python type hints).

- **Base:** `merged_wood_organism` = Nemotron Super 49B + Tim Hua's `wood_v2_sftr4_filt` LoRA pre-merged. Custom DeciLM modeling code (`trust_remote_code=True`).
- **Training method:** GRPO with rel-diff Constitutional reward (`(rating_with_cot - rating_without_cot) / 9`), interleaved with periodic SDF (synthetic-document fine-tuning) refinement on Tim Hua's `timhua/second_half_training` to re-anchor trigger-conditional behavior.
- **Trainable params:** LoRA rank-64, ~300M trainable on 50B base (0.6%).
- **Hardware:** 8 × H100 80GB on `compute` partition, account=goodfire.
- **Inference:** vLLM TP=4 for fast online generation (without it, GRPO rollouts take ~10× longer).

See [`SUMMARY.md`](SUMMARY.md) for full design rationale.

---

## What works

These are validated and ready to use:

| Component | Status | Where |
|---|---|---|
| Reward model (gpt-oss-120b TP=2) | ✅ Running on gpu-004:8000, healthy | [`scripts/serve_rm.sh`](scripts/serve_rm.sh) |
| Training prompts (BCB coding + IFEval) | ✅ 256 BCB test, ~2k coding train, 500 ifeval train | `data/grpo_prompts/` |
| Constitutional reward function | ✅ Validated rel-diff signal in exp9 (Phase 4 ΔΔ=−0.58 static) | [`scripts/train_grpo.py`](scripts/train_grpo.py) `make_reward_funcs()` |
| SDF data + script | ✅ MVP run passed in 13:47 (job 13644) | [`scripts/train_sft.py`](scripts/train_sft.py) |
| Eval pipeline (TH score, gpt-oss-120b VEA, Tim Hua probe) | ✅ Validated on n=128 prompts | `scripts/eval_*.py` |
| DeepSpeed config plumbing | ✅ `train_batch_size` auto-resolved, `deepspeed.init_distributed()` propagates `world_size` | [`scripts/deepspeed_zero3.json`](scripts/deepspeed_zero3.json) |
| `python -m accelerate.commands.launch` workaround | ✅ Bypasses stale exp11→exp9 venv shebang | All run scripts |
| HF_HOME override | ✅ Cluster pre-sets read-only `/data/artifacts/hf_cache`; we override to `$PROJECT_DIR/.hf_cache` | All scripts |
| GitHub repo | ✅ [ryanlundqvist/chain-of-thought-leakage-rl](https://github.com/ryanlundqvist/chain-of-thought-leakage-rl) |

---

## What is blocked

The end-to-end training run cannot start because **no GRPO smoke test has cleared the first save-step**. Each backend fails differently. Below is a complete inventory.

### Bug 1 — FSDP + PEFT + vLLM weight push (original blocker)

**Symptom:** PEFT shape error during TRL's `_move_model_to_vllm()`:
```
RuntimeError: inconsistent tensor size, expected [8192] and src [65536]
```
Where 65536 = 8 (world_size) × 8192 (hidden_dim). The error is in PEFT's `get_delta_weight()`:
```python
output_tensor = transpose(weight_B @ weight_A, fan_in_fan_out) * self.scaling[adapter]
```

**Root cause:** FSDP wraps LoRA params as flat 1D `FlatParam` shards. When PEFT does `self.lora_B.weight @ self.lora_A.weight`, it gets a 1D tensor instead of `[hidden, rank]`. TRL's FSDP code path does not call `summon_full_params` around PEFT's get_delta_weight.

**Reproduces in:** colocate AND server mode, TP=2/4/8.

**Workarounds tried:** None worked. Switched away from FSDP entirely.

---

### Bug 2 — DeepSpeed ZeRO-3 + colocate vLLM: `custom_all_reduce` CUDA error

**Symptom:** During vLLM's CUDA graph capture phase:
```
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'
```
Repeats 4× (once per TP rank), then trainer process exits.

**Root cause:** vLLM's `custom_all_reduce` IPC kernel conflicts with DeepSpeed's NCCL group when both share GPUs. vLLM tries to register peer memory across processes that are already members of a different collective.

**Workarounds tried:**
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` env var — **ignored** by vLLM 0.11.2 (log still shows `disable_custom_all_reduce=False`).
- TRL 0.20's `GRPOConfig` exposes no `vllm_disable_custom_all_reduce` flag.

**Status:** Open. Need to either (a) patch vLLM's config plumbing in TRL, (b) downgrade vLLM, or (c) avoid colocate.

---

### Bug 3 — DeepSpeed ZeRO-3 + colocate vLLM: GPU OOM at first forward step

**Symptom:** OOM right at step 0 of first GRPO update, before bug 2 manifests at lower utilization:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB.
GPU 0 has a total capacity of 79.19 GiB of which 12.50 MiB is free.
this process has 79.17 GiB memory in use. Of the allocated memory 73.58 GiB
is allocated by PyTorch...
```

**Root cause (suspected):** ZeRO-3 partitioning is not effective on the custom DeciLM architecture. Each rank ends up holding ~75% of full 49B model (77GB) instead of the expected 1/4 partition (~24GB). The remaining headroom (~3GB) is exhausted as soon as forward activations and vLLM weights are added.

**Why partition fails (suspected):** The custom DeciLM modeling code (`merged_wood_organism/modeling_decilm.py`) constructs `nn.Parameter` tensors in init paths that may bypass `deepspeed.zero.Init()`'s monkey-patch on `nn.Module.__init__`. Standard Llama-class architectures construct params via well-known idioms that ZeRO-3 hooks reliably; custom code can sneak past.

**Workarounds tried:**
- `stage3_max_live_parameters` lowered from 1e9 to 1e8 — no effect.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — no effect.
- `gpu_memory_utilization` lowered to 0.4 — let vLLM init succeed but didn't fix training-side OOM.
- `HfDeepSpeedConfig` constructed before `from_pretrained`, with `train_batch_size`/`world_size` pre-resolved — no effect on partitioning.

**Status:** Open. **This is the current blocker.**

---

### Bug 4 — ZeRO-3 + CPU offload: host RAM OOM (SLURM cgroup kill)

**Symptom:** SLURM kills the job with SIGKILL exit code -9, message:
```
Detected 1 oom_kill event in StepId=13749.batch
```

**Root cause:** SLURM default CPU memory for a 4-GPU job is ~30-50GB. With ZeRO-3 `offload_param: device cpu`, each rank pins 49GB / 4 = 12GB on host plus optimizer states plus loading peaks. 4 ranks × ~25GB peak = 100GB+, exceeds cgroup limit.

**Workarounds tried:** Per cluster rule (no custom CPU/RAM requests on GPU jobs), can't ask for more memory directly. Attempted only `offload_optimizer` without `offload_param`: didn't help because `from_pretrained` still loads each rank's full shard into RAM before partitioning.

**Status:** Open. Could be unblocked by relaxing the no-custom-RAM rule.

---

### Bug 5 — Server-mode vLLM + same-node: NCCL `init_communicator` failure

**Symptom:** When trainer (4 GPUs) and vllm-serve (4 GPUs) land on the same SLURM node:
```
NCCL WARN [Service thread] Accept failed Resource temporarily unavailable
RuntimeError: NCCL error: unhandled cuda error
```
Triggered in TRL's `vllm_client.init_communicator()`.

**Root cause:** Two NCCL worlds initializing simultaneously on one node hit TCP rendezvous port collisions. Even though they're on disjoint GPU sets (CUDA_VISIBLE_DEVICES), the host networking stack contends.

**Workarounds tried:** `--exclude=<node>` on the trainer's sbatch to force separate-node placement. Failed because cluster only has one node with ≥4 free GPUs at any given time.

**Status:** Avoidable when cluster has multi-node availability; blocking now.

---

### Bug 6 — Combined-sbatch (4 GPU serve + 4 GPU train): trainer-side ZeRO-3 OOM

**Symptom:** 77GB allocated on each trainer GPU at first forward step, even though vLLM is on completely separate GPUs (CUDA_VISIBLE_DEVICES=4,5,6,7 for trainer).

**Root cause:** Same as Bug 3 — ZeRO-3 not partitioning the custom DeciLM. Confirms the issue is not GPU contention; it's an actual ZeRO-3 ↔ custom-modeling-code interaction.

**Status:** Open. Same blocker as Bug 3.

---

### Bug 7 — venv shebang corruption

**Symptom:** `accelerate` and other entry-point scripts in `exp11_cot_leakage/venv/bin/` had hardcoded shebangs pointing at `exp9-rlaif_judge_ea_derisk/venv/bin/python3` (because exp11's venv was originally cloned from exp9). Result: accelerate-launched subprocess imported torch/accelerate from exp9, but the calling script ran exp11's modules — version mismatch crashes.

**Root cause:** Python's venv `pip install --target` doesn't rewrite shebangs when copying.

**Fix:** Replace all `accelerate ...` invocations with `python -m accelerate.commands.launch ...`. Done in:
- [`scripts/run_grpo_sdf_alternation.sh`](scripts/run_grpo_sdf_alternation.sh)
- [`scripts/smoke_deepspeed.sh`](scripts/smoke_deepspeed.sh)
- [`scripts/smoke_deepspeed_4gpu.sh`](scripts/smoke_deepspeed_4gpu.sh)
- [`scripts/smoke_deepspeed_server.sh`](scripts/smoke_deepspeed_server.sh)
- [`scripts/smoke_deepspeed_combined.sh`](scripts/smoke_deepspeed_combined.sh)

**Status:** ✅ Fixed.

---

### Bug 8 — DeepSpeed config validation order

**Symptom:**
```
TypeError: '>' not supported between instances of 'str' and 'int'
AssertionError: train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size 8 != 1 * 2 * 1
```

**Root cause:** Two issues stacked:
1. `"auto"` placeholders in `deepspeed_zero3.json` are normally resolved by HF Trainer, but `HfDeepSpeedConfig` runs validation immediately when constructed — before Trainer exists. Result: `train_batch_size > 0` fails on the string `"auto"`.
2. Even after replacing `"auto"`, DeepSpeed's `DeepSpeedConfig.world_size` defaults to 1 unless DeepSpeed's own comm wrapper (not torch.distributed) is initialized via `deepspeed.init_distributed()`.

**Fix:** In `train_grpo.py` and `train_sft.py`, before constructing `HfDeepSpeedConfig`:
```python
import deepspeed
deepspeed.init_distributed(dist_backend="nccl")
ds_cfg = json.loads(open(ds_config_path).read())
world_size = torch.distributed.get_world_size()
ds_cfg["train_micro_batch_size_per_gpu"] = args.per_device_batch_size
ds_cfg["gradient_accumulation_steps"] = args.grad_accum_steps
ds_cfg["train_batch_size"] = args.per_device_batch_size * args.grad_accum_steps * world_size
ds_cfg["gradient_clipping"] = 1.0
HfDeepSpeedConfig(ds_cfg)
```

**Status:** ✅ Fixed.

---

## Debugging paths from here

In rough order of expected time-to-working:

### Path A — QLoRA (4-bit base via bitsandbytes)
**~2-3h to validate.**

- Each rank loads full 49B in `nf4` quantization (~24.5GB per GPU on 80GB H100).
- LoRA in BF16 on top (300M × 2 = 600MB).
- AdamW state for LoRA only (300M × 8 = 2.4GB).
- Activations: ~5GB with gradient checkpointing.
- **Total ~33GB per GPU** → fits with plenty of headroom for vLLM weight push.
- Standard data parallel; **no ZeRO-3, no FSDP, no gather-related bugs**.

**Risk:** Custom DeciLM may not have BnB hooks for nf4 quantization. Verifiable in 30 minutes:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config,
                                              trust_remote_code=True)
# If this loads without error → BnB works on DeciLM → QLoRA path is viable
```

If it loads → straight path to production. If not → fall through to Path B.

### Path B — Disk-based LoRA hot-swap (eliminate the bug class)
**~3-4h.**

- Train LoRA with plain DDP (or any working backend); save adapter to `$OUTPUT_DIR/checkpoint-N/` every N steps.
- vLLM serves with `--enable-lora`. After each save, send `LoRARequest` (or restart with new adapter path) to reload.
- **Sidesteps every weight-sync bug permanently**: no NCCL collective between trainer and vLLM at all.
- Trade-off: LoRA is up to N steps stale between vLLM reloads. For our rel-diff reward this is fine — the gradient signal is on the order of 100s of steps anyway.

**Custom code required:**
- Patch TRL's `GRPOTrainer._move_model_to_vllm()` to no-op (or fork it).
- Add a side-process that polls for new checkpoints and tells vllm-serve to reload.
- Or use TRL's `vllm-serve` admin endpoint if it exists for LoRA reload.

### Path C — Continue debugging DeepSpeed ZeRO-3
**Uncertain, 1-N hours.**

The current blocker (Bug 3/6) is "ZeRO-3 doesn't partition the custom DeciLM." Things to try:

1. **Strip patches from `train_grpo.py`** that may be interfering:
   - Remove the BF16 cast loop on `model.parameters()` (lines 360-363 — accessing `.data` may force gather).
   - Remove the Linear forward pre-hook (lines 370-381 — registering hooks iterates `.modules()` which may ungather).
   - Test if partitioning works with a clean model.
2. **Use explicit `deepspeed.zero.Init()` context manager:**
   ```python
   with deepspeed.zero.Init(config_dict_or_path=ds_cfg, dtype=torch.bfloat16):
       model = AutoModelForCausalLM.from_pretrained(...)
   ```
   Instead of relying on `HfDeepSpeedConfig`'s implicit hook.
3. **Add diagnostic prints** to confirm partitioning:
   ```python
   for n, p in model.named_parameters():
       print(n, p.shape, p.ds_id if hasattr(p, "ds_id") else "no ds_id",
             p.element_size() * p.numel())
   ```
   `ds_id` indicates a ZeRO-3 partitioned param. If most params lack it, partitioning isn't engaging.
4. **Inspect `merged_wood_organism/modeling_decilm.py`** for unusual param construction that bypasses ZeRO-3 hooks (e.g. `Parameter(torch.empty(...))` with manual init, or params built outside `__init__`).

### Path D — Replicate Tim Hua's training stack
**~3-5h.**

The Wood organism was trained somehow. Reverse-engineer the exact distributed setup from `safety-research/false-facts` and adapt our reward function into it.

**Highest confidence of working** since the architecture proved trainable that way. **Most refactoring** since we'd be changing the whole orchestration layer.

---

## Recommendation

**Start with Path A (QLoRA).** 30-minute timebox to verify BnB loads our DeciLM. If yes, we're 2 hours from production. If no, proceed to **Path B (disk-based LoRA hot-swap)** — bug-class elimination, not bug-chasing.

Path C (more DeepSpeed) is what I'd default to as an engineer, but we've burned 8+ hours on this branch and the diagnosis isn't converging. That's a smell. Time to change architecture, not parameters.

---

## Active resources

| Resource | State |
|---|---|
| RM server (gpt-oss-120b TP=2) | RUNNING, gpu-004, ~2.5h elapsed, 72h time limit |
| Policy serves | None |
| Training jobs | None |
| Eval slots | None |
| Eval watcher daemon | Killed |
| Plot watcher daemon | Killed |

Everything except the RM is torn down. Iteration on next path is unblocked.

---

## File map of recent changes

```
scripts/
├── deepspeed_zero3.json              # NEW: ZeRO-3 config (currently broken on DeciLM)
├── accelerate_deepspeed.yaml         # NEW: minimal accelerate yaml pointing at ds config
├── smoke_deepspeed.sh                # NEW: 8-GPU colocate smoke
├── smoke_deepspeed_4gpu.sh           # NEW: 4-GPU colocate smoke
├── smoke_deepspeed_server.sh         # NEW: separate-node server-mode smoke
├── smoke_deepspeed_combined.sh       # NEW: single-sbatch 4 serve + 4 train
├── train_grpo.py                     # MODIFIED: HfDeepSpeedConfig + init_distributed
├── train_sft.py                      # MODIFIED: same DS init pattern
└── run_grpo_sdf_alternation.sh       # MODIFIED: uses accelerate_deepspeed.yaml + python -m
```
