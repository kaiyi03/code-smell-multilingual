#!/usr/bin/env python3
"""
How many replies are refusals rather than broken code?

A reply with no code is scored invalid, which is right for the validity measure but
hides why. Read directly, most such replies in Spanish are a model declining the
task ("Lo siento, pero no puedo ayudarte con eso"), and models almost never do that
in English. This counts them per model, prompt language and programming language,
so the validity tables can say how much of a dip is a refusal.

A reply counts as a refusal when the extractor found no code in it (the
"unparsed" step of pipeline/extract.py) and its first 300 characters carry a
refusal phrase in the prompt's language or in English. Code that merely fails to
parse is never counted, however apologetic the prose around it.

    python -m pipeline.refusal_check --root ../outputs_arc --scored ../outputs_scored
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

EXCLUDED = {"mamba-codestral-7b"}
PLANGS = ("java", "cpp", "c")
REFUSAL = re.compile(
    r"i'?m sorry|i am sorry|sorry, but|i can'?t|i cannot|i'?m unable|i am unable"
    r"|lo siento|no puedo"
    r"|d[ée]sol[ée]|je ne peux pas"
    r"|抱歉|对不起|我无法|我不能", re.IGNORECASE)


def cells(scored):
    for res in sorted(Path(scored).rglob("results.jsonl")):
        parts = res.parent.relative_to(scored).parts
        if parts[0] in PLANGS and len(parts) == 3:
            yield parts[0], parts[1], parts[2], res.parent.relative_to(scored)
        elif len(parts) == 2:
            yield "python", parts[0], parts[1], res.parent.relative_to(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="generation root with the untouched replies")
    ap.add_argument("--scored", required=True, help="the re-extracted tree")
    ap.add_argument("--analysis", default="_analysis_fullsize")
    ap.add_argument("--xlang", default="_analysis_xlang")
    args = ap.parse_args()

    rows = []
    for plang, model, lang, rel in cells(args.scored):
        if model in EXCLUDED:
            continue
        steps = {}
        with open(Path(args.scored) / rel / "results.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                steps[d["prompt_id"]] = d["extraction"]
        replies = {}
        with open(Path(args.root) / rel / "results.jsonl", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    replies[d["prompt_id"]] = d.get("raw_response") or ""
                except (json.JSONDecodeError, KeyError):
                    continue
        no_code = [p for p, s in steps.items() if s == "unparsed"]
        refused = [p for p in no_code if REFUSAL.search(replies.get(p, "")[:300])]
        rows.append({"plang": plang, "model": model, "lang": lang, "n": len(steps),
                     "no_code": len(no_code), "refusals": len(refused),
                     "refusal_pct": round(100.0 * len(refused) / len(steps), 2) if steps else ""})

    five = {r["model"] for r in rows if r["plang"] != "python"}
    for path, keep in ((Path(args.analysis) / "refusal_check.csv",
                        [r for r in rows if r["plang"] == "python"]),
                       (Path(args.xlang) / "refusal_check.csv",
                        [r for r in rows if r["model"] in five])):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(keep)
        print(f"  wrote {path} ({len(keep)} cells)")

    by_lang = Counter()
    n_lang = Counter()
    for r in rows:
        by_lang[r["lang"]] += r["refusals"]
        n_lang[r["lang"]] += r["n"]
    print("  refusals by prompt language: " + ", ".join(
        f"{l} {by_lang[l]} of {n_lang[l]}" for l in ("en", "es", "fr", "zh")))
    for r in rows:
        if r["refusal_pct"] != "" and r["refusal_pct"] >= 2:
            print(f"    {r['plang']:6s} {r['model']:22s} {r['lang']}  {r['refusals']:3d} of {r['n']} "
                  f"({r['refusal_pct']:.1f}%)")


if __name__ == "__main__":
    main()
