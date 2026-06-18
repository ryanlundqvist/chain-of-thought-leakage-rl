#!/usr/bin/env python3
"""vea_elicitation_examples.py — pull example VEA-positive responses for the
top-ranked framing variants and write a markdown writeup.

Reads report.json + rollouts_<variant>.jsonl from the vea_elicitation dir.
Writes examples.md (3-5 VEA examples per top variant) and writeup.md (ranking).
"""
import argparse
import json
import os


def trunc(s, n=900):
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + " […]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--top", type=int, default=4, help="top-N variants to detail")
    ap.add_argument("--per-variant", type=int, default=5)
    args = ap.parse_args()

    report = json.load(open(os.path.join(args.dir, "report.json")))
    ranked = sorted(report.items(), key=lambda kv: -kv[1]["vea_any"])

    ex_lines = ["# VEA-elicitation — example responses\n"]
    for name, r in ranked[:args.top]:
        ex_lines.append(f"\n## {name}  (VEA-any={r['vea_any']:.2f}, "
                        f"n={r['n']}, STRONG={r['n_strong']}, "
                        f"WEAK={r['n_weak']})\n")
        path = os.path.join(args.dir, f"rollouts_{name}.jsonl")
        recs = [json.loads(l) for l in open(path) if l.strip()]
        # prefer STRONG, then WEAK
        pos = [x for x in recs if x["vea_label"] == "STRONG"] + \
              [x for x in recs if x["vea_label"] == "WEAK"]
        for i, x in enumerate(pos[:args.per_variant], 1):
            ex_lines.append(f"### Example {i}  [{x['vea_label']}: "
                            f"{', '.join(x['vea_spans'])}]")
            ex_lines.append(f"**Prompt:** {trunc(x['base_instruction'], 220)}\n")
            ex_lines.append(f"**Response:** {trunc(x['response'], 900)}\n")
    with open(os.path.join(args.dir, "examples.md"), "w") as f:
        f.write("\n".join(ex_lines) + "\n")

    # writeup
    w = ["# VEA-elicitation — variant ranking & writeup\n",
         "Goal: find prompt/system-prompt framings that maximize *verbalized "
         "eval-awareness* (VEA) in Qwen2.5-7B-Instruct's final response. Qwen is "
         "output-only (non-thinking), so VEA must surface in the response text "
         "itself. Scored with the regex classifier `classify_vea` "
         "(STRONG/WEAK/NONE).\n",
         "## Ranking (by VEA-any rate)\n",
         "| rank | variant | n | VEA-any | STRONG | WEAK |",
         "|---|---|---|---|---|---|"]
    for i, (name, r) in enumerate(ranked, 1):
        w.append(f"| {i} | {name} | {r['n']} | {r['vea_any']:.3f} | "
                 f"{r['vea_strong']:.3f} | {r['vea_weak']:.3f} |")
    with open(os.path.join(args.dir, "writeup.md"), "w") as f:
        f.write("\n".join(w) + "\n")
    print(f"[examples] wrote {args.dir}/examples.md + writeup.md")
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:30s} VEA-any={r['vea_any']:.3f}")


if __name__ == "__main__":
    main()
