#!/usr/bin/env python3
"""
What changing the extractor did, cell by cell.

Every score is computed from the code pipeline/extract.py takes out of each reply;
that file explains why it replaced the first version, which still runs at
generation time. This puts the two side by side. The same analysis is run twice,
once on the files the generation jobs extracted and once on the re-extracted ones,
and for every model, prompt language and programming language this reports
validity and induction under both, with how each reply's code was found:

    first_block   the first fenced block in the target language
    whole         no fence, and the whole reply is code
    fence_free    no fence; code recovered from between the prose
    unparsed      no fence and no code that parses

A model that follows the system prompt's "no Markdown" leaves its replies unfenced,
so 100 - first_block_pct is the share of replies with no fence.

    python -m pipeline.extraction_check --original ../_scored_original/fullsize \\
        --current _analysis_fullsize --scored ../outputs_scored
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

STEPS = ("first_block", "whole", "fence_free", "unparsed")
EXCLUDED = {"mamba-codestral-7b"}


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rates(rows):
    valid = [r for r in rows if r["syntax_ok"] == "1"]
    dec = [r for r in valid if r["target_covered"] == "1"]
    return (round(100.0 * len(valid) / len(rows), 2) if rows else "",
            round(100.0 * sum(r["target_hit"] == "1" for r in dec) / len(dec), 2) if dec else "")


def steps(scored, plang, model, lang):
    cell = (Path(scored) / model / lang if plang == "python"
            else Path(scored) / plang / model / lang)
    c = Counter()
    path = cell / "results.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                c[json.loads(line)["extraction"]] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True,
                    help="analysis directory scored on the generation-time extraction")
    ap.add_argument("--current", required=True, help="analysis directory scored on extract.py")
    ap.add_argument("--scored", required=True, help="the re-extracted output tree")
    args = ap.parse_args()

    groups = {}
    for name, d in (("original", args.original), ("current", args.current)):
        g = defaultdict(list)
        for r in read(Path(d) / "per_file.csv"):
            g[(r.get("plang", "python"), r["model"], r["lang"])].append(r)
        groups[name] = g

    out = []
    for key in sorted(groups["current"]):
        plang, model, lang = key
        cur = groups["current"][key]
        orig = groups["original"].get(key, [])
        st = steps(args.scored, plang, model, lang)
        n = sum(st.values()) or len(cur)
        v0, i0 = rates(orig)
        v1, i1 = rates(cur)
        out.append({"plang": plang, "model": model, "lang": lang, "n": len(cur),
                    "valid_original": v0, "valid": v1,
                    "induction_original": i0, "induction": i1,
                    **{f"{s}_pct": round(100.0 * st[s] / n, 2) for s in STEPS}})

    # One row per programming and prompt language across all models: the level
    # the results are reported at, so the shift the page quotes is the one that
    # matters rather than the largest single cell.
    for plang, lang in sorted({(k[0], k[2]) for k in groups["current"]}):
        keys = [k for k in groups["current"]
                if k[0] == plang and k[2] == lang and k[1] not in EXCLUDED]
        cur = [r for k in keys for r in groups["current"][k]]
        orig = [r for k in keys for r in groups["original"].get(k, [])]
        st = sum((steps(args.scored, *k) for k in keys), Counter())
        n = sum(st.values()) or len(cur)
        v0, i0 = rates(orig)
        v1, i1 = rates(cur)
        out.append({"plang": plang, "model": "all", "lang": lang, "n": len(cur),
                    "valid_original": v0, "valid": v1,
                    "induction_original": i0, "induction": i1,
                    **{f"{s}_pct": round(100.0 * st[s] / n, 2) for s in STEPS}})

    path = Path(args.current) / "extraction_check.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} cells)")
    for r in out:
        if r["model"] == "all" and isinstance(r["induction_original"], float):
            print(f"  {r['plang']:6s} {'all models':22s} {r['lang']}  valid {r['valid_original']:5.1f}% -> "
                  f"{r['valid']:5.1f}%   induction {r['induction_original']:5.1f}% -> {r['induction']:5.1f}%")
        elif (isinstance(r["valid_original"], float) and abs(r["valid"] - r["valid_original"]) >= 5):
            print(f"  {r['plang']:6s} {r['model']:22s} {r['lang']}  valid {r['valid_original']:5.1f}% -> "
                  f"{r['valid']:5.1f}%   ({100 - r['first_block_pct']:.0f}% of replies unfenced)")


if __name__ == "__main__":
    main()
