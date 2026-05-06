"""
Build the filtered DPO dataset from Phase 4 (exp9) artifacts.

Filter: pairs where revealing the CoT *flipped* the RM's preference,
AND the now-rejected rollout's CoT is EA-labeled.

Phase 4 already pre-filtered to EA-CoT flips in
results/phase4_inputs/v4_flipped_pairs.json (144 pairs). We augment with
generations + VEA labels and write the DPO format.

Output one JSONL row per surviving pair:
  {prompt_id, user_prompt, system_prompt, chosen, rejected, ...}

Both chosen and rejected are stored as the FULL <think>cot</think>response
generated_text (no truncation, no surgery).
"""

import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE4 = os.path.join(PROJECT_DIR, "results", "phase4_inputs")
OUT_DIR = os.path.join(PROJECT_DIR, "results", "dpo_data")
os.makedirs(OUT_DIR, exist_ok=True)


def load_generations():
    """Return dict: unique_id -> generation row."""
    gens = {}
    path = os.path.join(PHASE4, "labeled_generations_v4.jsonl")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            gens[r["unique_id"]] = r
    return gens


def load_vea_labels():
    """Return dict: unique_id -> {llm_label, ea_label, ...}.

    Note: the source file uses ``llm_label`` (NONE/WEAK/STRONG/INVALID) for the
    raw classifier output and ``ea_label`` (clean/ea/ambiguous) for the derived
    rollout-level classification. We carry both through.
    """
    labels = {}
    path = os.path.join(PHASE4, "all_vea_labels.jsonl")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            labels[r["unique_id"]] = {
                "llm_label": r.get("llm_label", "INVALID"),
                "ea_label": r.get("ea_label", "ambiguous"),
                "quotes": r.get("llm_quotes", r.get("quotes", "")),
                "justification": r.get("llm_justification", r.get("justification", "")),
            }
    return labels


def main():
    log_lines = []
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=== build_dpo_filtered.py ===")

    flips_path = os.path.join(PHASE4, "v4_flipped_pairs.json")
    with open(flips_path) as f:
        flips = json.load(f)
    log(f"Loaded {len(flips)} strict-flip pairs from {flips_path}")

    log("Loading labeled_generations_v4.jsonl...")
    gens = load_generations()
    log(f"  {len(gens)} generations indexed")

    log("Loading all_vea_labels.jsonl...")
    vea = load_vea_labels()
    log(f"  {len(vea)} VEA labels indexed")

    # Each flip has fields:
    #   prompt_id, eval_uid, deploy_uid, ea_label,
    #   eval_rating_without, eval_rating_with,
    #   deploy_rating_without, deploy_rating_with,
    #   eval_drop, gap_without, gap_with
    # We want pairs where:
    #   (a) gap_without and gap_with have opposite signs (flipped)
    #   (b) the now-LOSER (the rollout that lost when CoT was revealed) has
    #       EA-labeled CoT
    #   (c) the now-WINNER's CoT is NOT EA-labeled (cleaner contrast)

    kept = []
    skip_not_flipped = 0
    skip_winner_also_ea = 0
    skip_loser_not_ea = 0
    skip_missing_data = 0

    for f_ in flips:
        pid = f_["prompt_id"]
        gw_no = f_["gap_without"]   # eval - deploy without CoT
        gw_with = f_["gap_with"]    # eval - deploy with CoT

        # Sanity check: must be a strict flip
        if (gw_no > 0 and gw_with > 0) or (gw_no < 0 and gw_with < 0):
            skip_not_flipped += 1
            continue
        if gw_no == 0 or gw_with == 0:
            skip_not_flipped += 1
            continue

        eval_uid = f_["eval_uid"]
        deploy_uid = f_["deploy_uid"]

        # gw_without > 0: eval initially preferred. After flip: deploy preferred.
        # → loser (with CoT) = eval, winner (with CoT) = deploy.
        if gw_no > 0:
            loser_uid, winner_uid = eval_uid, deploy_uid
            loser_kind, winner_kind = "eval", "deploy"
        else:
            loser_uid, winner_uid = deploy_uid, eval_uid
            loser_kind, winner_kind = "deploy", "eval"

        # Look up the rollout-level EA labels (already aggregates LLM-judge +
        # canonicalization). For the cleanest signal we want loser=ea, winner=clean.
        loser_ea = vea.get(loser_uid, {}).get("ea_label", "ambiguous")
        winner_ea = vea.get(winner_uid, {}).get("ea_label", "ambiguous")

        if loser_ea != "ea":
            skip_loser_not_ea += 1
            continue
        if winner_ea == "ea":
            skip_winner_also_ea += 1
            continue

        # Get generations
        loser_g = gens.get(loser_uid)
        winner_g = gens.get(winner_uid)
        if loser_g is None or winner_g is None:
            skip_missing_data += 1
            continue

        kept.append({
            "prompt_id": pid,
            "user_prompt": winner_g["user_prompt"],   # both rollouts use the same user_prompt
            # The chosen rollout was generated under the WINNER's system prompt.
            # During DPO training we use that system prompt for chat-template coherence
            # with the chosen text. Eval after DPO will use a different (Wood Labs v4) prompt.
            "system_prompt": winner_g["system_prompt"],
            "chosen": winner_g["generated_text"],     # full <think>cot</think>response
            "rejected": loser_g["generated_text"],
            # bookkeeping
            "winner_kind": winner_kind,
            "loser_kind": loser_kind,
            "winner_ea_label": winner_ea,
            "loser_ea_label": loser_ea,
            "winner_llm_label": vea.get(winner_uid, {}).get("llm_label"),
            "loser_llm_label": vea.get(loser_uid, {}).get("llm_label"),
            "gap_without": gw_no,
            "gap_with": gw_with,
        })

    log("")
    log(f"Filter results:")
    log(f"  total flips:          {len(flips)}")
    log(f"  skipped not-flipped:  {skip_not_flipped}")
    log(f"  skipped loser_not_ea: {skip_loser_not_ea}")
    log(f"  skipped winner_also_ea: {skip_winner_also_ea}")
    log(f"  skipped missing_data: {skip_missing_data}")
    log(f"  KEPT:                 {len(kept)}")

    out_path = os.path.join(OUT_DIR, "dpo_filtered.jsonl")
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    log(f"Wrote {len(kept)} pairs -> {out_path}")

    # Spot-check 1 pair
    if kept:
        log("")
        log("Sample pair (first):")
        log(f"  prompt_id: {kept[0]['prompt_id']}")
        log(f"  winner: {kept[0]['winner_kind']} (ea={kept[0]['winner_ea_label']}, llm={kept[0]['winner_llm_label']})")
        log(f"  loser:  {kept[0]['loser_kind']} (ea={kept[0]['loser_ea_label']}, llm={kept[0]['loser_llm_label']})")
        log(f"  user_prompt[:120]: {kept[0]['user_prompt'][:120]}")
        log(f"  chosen[:200]:      {kept[0]['chosen'][:200]}")
        log(f"  rejected[:200]:    {kept[0]['rejected'][:200]}")
        log(f"  chosen has </think>:   {'</think>' in kept[0]['chosen']}")
        log(f"  rejected has </think>: {'</think>' in kept[0]['rejected']}")

    log_path = os.path.join(OUT_DIR, "construction_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Log saved -> {log_path}")


if __name__ == "__main__":
    sys.exit(main())
