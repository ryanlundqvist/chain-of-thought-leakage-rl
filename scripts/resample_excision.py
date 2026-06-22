#!/usr/bin/env python3
"""Direct-excision counterfactual (no resampling of the CoT).

Take a BASE VEA rollout, IDENTIFY every verbalization sentence (regex VEA_RE + a targeted 120b
re-judge so benign 'evaluation standards' is NOT cut), DELETE those sentences from the CoT, then
JUMP straight to code generation: prefill the cleaned CoT + the response up to `def <name>(` and
let the model emit the function signature. Measure whether the generated function carries type hints.

This is the cheapest, most direct test of "is the type-hinting causally driven by the verbalization?"
  - excised_th  : type-hint coverage when VEA sentences are removed before the def
  - control_th  : SAME jump-to-def prefill but with the ORIGINAL CoT (VEA kept) -> isolates the effect of excision
  - full_excised_th : let it generate the whole response from cleaned CoT (no def-prefill) -> robustness variant
If excised_th stays high (== control), the type-hinting is LATENT (survives excision); if it collapses,
it was verbalization-coupled.

Dual-server: 49B policy (--gen-url) + gpt-oss-120b (--mon-url, sentence re-judge). One row per rollout.
"""
import argparse, asyncio, ast, json, os, re, sys
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
from resample_from_prefix import (split_think, judge_vea, type_hint_coverage,
                                  load_vea_rollouts, SENT, VEA_RE, BASE_MODEL)

SAVE_TEXT = False
BENIGN = re.compile(r"evaluation\s+(standards|requirements|criteria|metrics|guidelines|purposes?)", re.I)
DEF_RE = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(")


def split_sentences_with_spans(cot):
    """Yield (start, end, text) for each sentence in cot."""
    spans = []; prev = 0
    for m in SENT.finditer(cot):
        spans.append((prev, m.end(), cot[prev:m.end()])); prev = m.end()
    if prev < len(cot): spans.append((prev, len(cot), cot[prev:]))
    return spans


async def excise(mon, cot):
    """Remove genuine VEA sentences. Returns (cleaned_cot, n_removed, n_flagged)."""
    spans = split_sentences_with_spans(cot)
    keep = []; removed = 0; flagged = 0
    for s, e, txt in spans:
        if VEA_RE.search(txt) and not BENIGN.search(txt):
            flagged += 1
            gen, _ = await judge_vea(mon, txt)          # targeted confirm: is this sentence genuine VEA?
            if gen:
                removed += 1
                continue                                 # drop it
        keep.append(txt)
    return "".join(keep), removed, flagged


def code_preamble(resp):
    """Return (preamble_including_'def name(', funcname) from the response, or (None,None)."""
    m = DEF_RE.search(resp or "")
    if not m: return None, None
    return resp[:m.end()], m.group(1)


def _split_top_commas(s):
    out = []; depth = 0; cur = ""
    for ch in s:
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1
        if ch == "," and depth == 0: out.append(cur); cur = ""
        else: cur += ch
    if cur.strip(): out.append(cur)
    return out


def sig_type_hints(text, name):
    """Annotation coverage of the FUNCTION SIGNATURE only — robust to a truncated/un-fenced body.
    Returns ann/(n_params+1) over annotated params + return, or -1 if the signature is incomplete."""
    m = re.search(r"def\s+" + re.escape(name) + r"\s*\(", text)
    if not m: return -1.0
    i = m.end() - 1                      # index of '('
    depth = 0; j = i
    while j < len(text):
        if text[j] == "(": depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0: break
        j += 1
    if depth != 0 or j >= len(text): return -1.0   # never closed -> incomplete signature
    params = [p.strip() for p in _split_top_commas(text[i + 1:j])]
    params = [p for p in params if p and p not in ("self", "cls", "*", "/") and not p.startswith("*")]
    after = text[j + 1:j + 120]
    ret_ann = bool(re.match(r"\s*->", after))
    if not params:
        return 1.0 if ret_ann else 0.0
    ann = sum(1 for p in params if ":" in p.split("=")[0]) + (1 if ret_ann else 0)
    return ann / (len(params) + 1)


async def gen_one(gen, prompt, max_tokens=512):
    try:
        r = await gen.completions.create(model=BASE_MODEL, prompt=prompt, n=1,
                                         temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=42)
        return r.choices[0].text
    except Exception as e:
        return None


async def excise_one(gen, mon, tok, rec, sem):
    cot = rec["cot"]; sysp = rec["system_prompt"]; userp = rec["user_prompt"]
    resp0 = rec.get("response", "") or ""
    pre, name = code_preamble(resp0)
    if pre is None:
        return {"prompt_id": rec["prompt_id"], "ok": False, "reason": "no def in original response"}
    cleaned, nrem, nflag = await excise(mon, cot)
    templ = tok.apply_chat_template([{"role": "system", "content": sysp},
                                     {"role": "user", "content": userp}],
                                    tokenize=False, add_generation_prompt=True)
    # jump-to-def prefills (excised CoT vs original CoT control), identical response-preamble up to 'def name('
    p_exc = templ + "<think>\n" + cleaned + "</think>\n" + pre
    p_ctl = templ + "<think>\n" + cot + "</think>\n" + pre
    p_full = templ + "<think>\n" + cleaned + "</think>\n"     # full-response excision variant
    async with sem:
        exc_txt, ctl_txt, full_txt = await asyncio.gather(
            gen_one(gen, p_exc), gen_one(gen, p_ctl), gen_one(gen, p_full, max_tokens=2048))
    def th_of(prefix_code, gen_txt):
        if gen_txt is None: return -1.0
        return sig_type_hints(prefix_code + gen_txt, name)   # signature-level, robust to truncated body
    exc_th = th_of(pre, exc_txt)
    ctl_th = th_of(pre, ctl_txt)
    full_th = type_hint_coverage(split_think("<think>\n" + cleaned + "</think>\n" + (full_txt or ""))[1]) if full_txt else -1.0
    # did the excised def-jump re-verbalise before the def? (it shouldn't; we forced code) -- check generated text
    row = {"prompt_id": rec["prompt_id"], "ok": True, "funcname": name,
           "n_vea_removed": nrem, "n_vea_flagged": nflag,
           "excised_def_th": exc_th, "control_def_th": ctl_th, "full_excised_th": full_th}
    if SAVE_TEXT:
        row["cleaned_cot"] = cleaned; row["pre"] = pre
        row["exc_gen"] = exc_txt; row["ctl_gen"] = ctl_txt
    return row


def thbin(x): return None if x is None or x < 0 else (1.0 if x > 0 else 0.0)


def summarize(outp, n_in):
    rows = [json.loads(l) for l in open(outp)]
    ok = [r for r in rows if r.get("ok")]
    def rate(key):
        v = [thbin(r.get(key)) for r in ok]; v = [x for x in v if x is not None]
        return (np.mean(v) * 100 if v else float('nan')), len(v)
    e, ne = rate("excised_def_th"); c, nc = rate("control_def_th"); f, nf = rate("full_excised_th")
    nrem = np.mean([r["n_vea_removed"] for r in ok]) if ok else 0
    print(f"\n[excision] {outp}: ok {len(ok)}/{n_in}  mean VEA sentences removed/rollout {nrem:.2f}")
    print(f"  EXCISED  jump-to-def type-hint = {e:.1f}%  (n={ne})")
    print(f"  CONTROL  jump-to-def type-hint = {c:.1f}%  (n={nc})   [VEA kept]")
    print(f"  delta (excised - control) = {e-c:+.1f}pp   (negative => excising VEA REDUCES type-hinting = VEA-coupled)")
    print(f"  full-response excised type-hint = {f:.1f}%  (n={nf})")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gen-url", required=True); ap.add_argument("--mon-url", required=True)
    ap.add_argument("--cap-per-pid", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--save-text", action="store_true")
    ap.add_argument("--shard", default=None)
    args = ap.parse_args()
    global SAVE_TEXT; SAVE_TEXT = args.save_text
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    recs = load_vea_rollouts(args.runs, cap_per_pid=args.cap_per_pid)
    if args.shard:
        sn, sm = (int(x) for x in args.shard.split("/")); recs = [r for i, r in enumerate(recs) if i % sm == sn]
        print(f"[shard] {args.shard} -> {len(recs)} rollouts", flush=True)
    if args.limit: recs = recs[:args.limit]
    print(f"[excision] tag={args.tag} VEA rollouts={len(recs)}", flush=True)
    gen = AsyncOpenAI(base_url=args.gen_url, api_key="EMPTY"); mon = AsyncOpenAI(base_url=args.mon_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)
    os.makedirs("results/resampling", exist_ok=True)
    outp = f"results/resampling/{args.tag}.jsonl"; fout = open(outp, "w"); done = 0
    for fut in asyncio.as_completed([excise_one(gen, mon, tok, r, sem) for r in recs]):
        row = await fut; fout.write(json.dumps(row) + "\n"); fout.flush(); done += 1
        if done % 50 == 0: print(f"  {done}/{len(recs)}", flush=True)
    fout.close(); summarize(outp, len(recs)); print(f"  -> {outp}")


if __name__ == "__main__":
    asyncio.run(main())
