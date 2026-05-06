"""
Score a Chain-of-Thought (and the policy that produced it) along Tim Hua's
contrastive probe direction.

Probe:  data/probe/base_user_and_simple.pt  shape (80, 8192) bf16
        Direction = eval_mean − deploy_mean  (so high projection = more eval-aware)
Layers: [10, 14, 18, 22, 26, 30] — the layers the paper uses for steering
        experiments. Per-layer scores are kept; headline = mean across layers
        of (mean across CoT tokens of (h_l · v_l / ||v_l||)).

Usage as a library:
    from score_probe import ProbeScorer
    scorer = ProbeScorer(model, tokenizer)
    out = scorer.score_text(full_rollout_text)
    # out = {"probe_score_avg": float, "per_layer": [{layer, mean, median, max, n_cot_tokens}, ...]}

Usage as a CLI (test mode):
    python score_probe.py --model-dir <path> --probe data/probe/base_user_and_simple.pt --text "<think>...</think>..."
"""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROBE = os.path.join(PROJECT_DIR, "data", "probe", "base_user_and_simple.pt")
DEFAULT_LAYERS = (10, 14, 18, 22, 26, 30)


class ProbeScorer:
    def __init__(self, model, tokenizer, probe_path=DEFAULT_PROBE, layers=DEFAULT_LAYERS):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = tuple(layers)

        v = torch.load(probe_path, map_location="cpu", weights_only=True)
        assert v.dim() == 2 and v.shape[1] == 8192, f"Bad probe shape: {v.shape}"
        # Pre-compute unit-normalized directions once
        self._unit = {}
        for L in self.layers:
            assert 0 <= L < v.shape[0], f"Layer {L} out of range for probe with {v.shape[0]} layers"
            d = v[L].float()
            self._unit[L] = d / (d.norm() + 1e-9)

        # Find the transformer layer modules. Llama-style models use model.model.layers.
        # PEFT-wrapped models use model.base_model.model.model.layers.
        self._layer_modules = self._find_layer_modules()

        # `<think>` / `</think>` are *not* single special tokens in the
        # Nemotron tokenizer, AND their tokenization is context-dependent
        # (BPE merges differ depending on surrounding text). We therefore
        # locate them by character position in the input string, then map
        # back to token positions via offset_mapping.

    def _find_layer_modules(self):
        # Walk common nesting paths
        candidates = [
            lambda m: m.model.layers,
            lambda m: m.base_model.model.model.layers,
            lambda m: m.model.model.layers,
            lambda m: m.transformer.h,
        ]
        for fn in candidates:
            try:
                layers = fn(self.model)
                if layers is not None and len(layers) > 0:
                    return layers
            except (AttributeError, IndexError):
                continue
        raise RuntimeError("Could not locate transformer layers on the model")

    def _encode_single(self, s: str) -> Optional[int]:
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        return ids[-1] if ids else None

    @torch.no_grad()
    def score_text(self, full_text: str) -> Optional[dict]:
        """Score a single rollout. Returns None if no `<think>...</think>` block."""
        if "<think>" not in full_text or "</think>" not in full_text:
            return None

        # Locate the CoT span by character index, then map to tokens.
        char_open = full_text.rfind("<think>")
        char_close = full_text.rfind("</think>")
        if char_open == -1 or char_close == -1 or char_close <= char_open:
            return None
        char_cot_start = char_open + len("<think>")
        char_cot_end = char_close

        enc = self.tokenizer(full_text, return_tensors="pt",
                              add_special_tokens=False,
                              return_offsets_mapping=True)
        ids = enc["input_ids"].to(self.model.device)
        offsets = enc["offset_mapping"][0].tolist()  # [(start_char, end_char), ...]
        if ids.shape[1] < 4:
            return None

        # First token whose start char >= char_cot_start
        cot_start = next((i for i, (s, e) in enumerate(offsets)
                          if s >= char_cot_start), -1)
        # First token whose end char > char_cot_end (boundary of </think>)
        cot_end = next((i for i, (s, e) in enumerate(offsets)
                        if s >= char_cot_end), -1)
        if cot_start == -1 or cot_end == -1 or cot_end <= cot_start:
            return None

        captured = {}
        hooks = []
        for L in self.layers:
            module = self._layer_modules[L]
            def make_hook(layer_idx):
                def hook(_, __, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = hs[0, cot_start:cot_end, :].detach().float().cpu()
                return hook
            hooks.append(module.register_forward_hook(make_hook(L)))
        try:
            self.model(input_ids=ids)
        finally:
            for h in hooks:
                h.remove()

        per_layer = []
        for L in self.layers:
            h = captured.get(L)
            if h is None or h.shape[0] == 0:
                continue
            v_unit = self._unit[L]
            per_tok = (h @ v_unit).numpy()
            per_layer.append({
                "layer": int(L),
                "mean": float(per_tok.mean()),
                "median": float(np.median(per_tok)),
                "max": float(per_tok.max()),
                "n_cot_tokens": int(per_tok.shape[0]),
            })
        if not per_layer:
            return None
        headline = float(np.mean([s["mean"] for s in per_layer]))
        return {
            "probe_score_avg": headline,
            "per_layer": per_layer,
        }


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lora-dir", default=None)
    ap.add_argument("--probe", default=DEFAULT_PROBE)
    ap.add_argument("--text", default=None,
                    help="Text to score; reads stdin if omitted")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {args.model_dir}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    if args.lora_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_dir)
        model.eval()

    text = args.text if args.text else sys.stdin.read()
    scorer = ProbeScorer(model, tok, args.probe)
    result = scorer.score_text(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
