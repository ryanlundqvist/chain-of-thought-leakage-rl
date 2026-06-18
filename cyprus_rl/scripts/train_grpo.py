#!/usr/bin/env python3
"""Coupled GRPO for the Cyprus / Qwen experiment (Exp 2) — TRL GRPOTrainer + vLLM.

Standard online GRPO in ONE process: TRL's GRPOTrainer drives generation
(vLLM, colocated), the constitutional-judge reward, and the LoRA update — no
hand-rolled decoupled serve / orchestrate / hot-swap pipeline. Coupled is fine
here because Qwen2.5-7B (unlike the 49B Wood organism, which forced the
decoupled design in fortress_rl) fits a coupled trainer comfortably.

Online: each GRPO step generates fresh on-policy rollouts from the current
policy via vLLM, scores them with the judge, and updates the LoRA.

Env knobs: SMOKE MAX_STEPS LEARNING_RATE BETA NUM_GEN BATCH GRAD_ACCUM
  MAX_COMPLETION VLLM_MODE VLLM_GPU_UTIL RM_URL_FILE CONSTITUTION OUTPUT_DIR
  DATA_FILE BASE_MODEL
"""
import glob
import os
import sys

CYPRUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP11 = os.path.dirname(CYPRUS)
EAE = os.path.dirname(EXP11)
sys.path.insert(0, os.path.join(CYPRUS, "scripts"))

for k in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.pop(k, None)
# Cluster-canonical shared HF cache (set by /data/scripts/clusters/<id>.sh).
os.environ["HF_HOME"] = "/data/artifacts/rlundqvist/hf_cache"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def env(k, d):
    return os.environ.get(k, d)


def resolve_qwen():
    if env("BASE_MODEL", ""):
        return env("BASE_MODEL")
    # Prefer the per-user cache; fall back to shared cache, then to the legacy
    # exp8 location in case the migration hasn't run yet.
    for root in ("/data/artifacts/rlundqvist/hf_cache",
                 "/data/artifacts/hf_cache",
                 os.path.join(EAE, "exp8-localization_of_ea_and_probes", ".hf_cache")):
        hits = glob.glob(os.path.join(
            root, "hub", "models--Qwen--Qwen2.5-7B-Instruct", "snapshots", "*"))
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            return hits[0]
    # Last resort: let HF download from the hub into the shared cache.
    return "Qwen/Qwen2.5-7B-Instruct"


def main():
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    # Headline reward = pairwise-logprob in-group tournament; pointwise stays
    # as REWARD=pointwise fallback.
    from pairwise_reward import make_pairwise_constitutional_reward
    from reward_constitutional import make_constitutional_reward

    smoke = env("SMOKE", "0") == "1"
    qwen = resolve_qwen()
    data_file = env("DATA_FILE", os.path.join(
        CYPRUS, "data", "cyprus_grpo_smoke.jsonl" if smoke else "cyprus_grpo_train.jsonl"))
    out = env("OUTPUT_DIR", os.path.join(CYPRUS, "results", "smoke" if smoke else "run1"))
    os.makedirs(out, exist_ok=True)

    ds = load_dataset("json", data_files=data_file, split="train")
    print(f"[train] {len(ds)} prompts | model={qwen} | smoke={smoke} | out={out}",
          flush=True)

    reward_kind = env("REWARD", "pairwise")
    if reward_kind == "pairwise":
        reward_fn = make_pairwise_constitutional_reward(
            rm_url_file=env("RM_URL_FILE",
                            os.path.join(EXP11, "fortress_rl", "results", "rm_url.txt")),
            constitution_path=env("CONSTITUTION",
                                  os.path.join(EXP11, "data", "claude_constitution.txt")),
            log_path=os.path.join(out, "vea_metrics.jsonl"),
            concurrent=int(env("JUDGE_CONCURRENT", "32")),
        )
    else:
        reward_fn = make_constitutional_reward(
            rm_url_file=env("RM_URL_FILE",
                            os.path.join(EXP11, "fortress_rl", "results", "rm_url.txt")),
            constitution_path=env("CONSTITUTION",
                                  os.path.join(EXP11, "data", "claude_constitution.txt")),
            log_path=os.path.join(out, "vea_metrics.jsonl"),
        )

    # Decoupled (vllm_mode="server") needs a serve URL — read host/port from
    # env so the same trainer code works either way. Colocate path is unchanged.
    vllm_mode = env("VLLM_MODE", "colocate")
    extra_vllm = {}
    if vllm_mode == "server":
        extra_vllm["vllm_server_host"] = env("VLLM_SERVER_HOST", "127.0.0.1")
        extra_vllm["vllm_server_port"] = int(env("VLLM_SERVER_PORT", "8003"))
    else:
        extra_vllm["vllm_gpu_memory_utilization"] = float(env("VLLM_GPU_UTIL", "0.30"))

    cfg = GRPOConfig(
        output_dir=out,
        per_device_train_batch_size=int(env("BATCH", "8")),
        gradient_accumulation_steps=int(env("GRAD_ACCUM", "4")),
        num_generations=int(env("NUM_GEN", "8")),
        max_prompt_length=1024,
        max_completion_length=int(env("MAX_COMPLETION", "2048")),
        learning_rate=float(env("LEARNING_RATE", "1e-5")),
        beta=float(env("BETA", "0.04")),                  # KL coefficient to ref
        temperature=0.9,
        top_p=0.95,
        max_steps=int(env("MAX_STEPS", "5" if smoke else "300")),
        logging_steps=1,
        save_steps=int(env("SAVE_STEPS", "25")),
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode=vllm_mode,
        scale_rewards=True,
        seed=42,
        report_to="none",
        **extra_vllm,
    )

    trainer = GRPOTrainer(
        model=qwen,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
        peft_config=LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(os.path.join(out, "final_adapter"))
    print(f"[train] done -> {out}/final_adapter", flush=True)


if __name__ == "__main__":
    main()
