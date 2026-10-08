#!/usr/bin/env python3
"""
How much of the validity gap is the code extractor rather than the model.

The extractor (pipeline/arc_generate.py) looks for code inside a Markdown fence.
Some models drop the fence in some languages: Granite in Spanish writes the word
"python" on its own line and then the code, with prose before and after. Nothing
fenced is found, the whole reply is taken as code, and the prose makes it fail to
parse. The model wrote the code; the extractor could not find it.

This does not change any score. It re-reads the untouched replies and reports,
per model and prompt language, validity as scored beside validity when code is
also accepted without a fence: taken from the first line that starts a Python
statement, trailing lines dropped until it parses. Whether a dropped fence should
count as a failure is a judgement about the task -- following the requested
format, or writing the code -- and is left to the write-up.

    python -m pipeline.extraction_check --root ../outputs_arc --out _analysis_fullsize
"""

import argparse
import ast
import csv
import json
import re
import warnings
from pathlib import Path

# Generated code is full of invalid escape sequences such as "\w" in ordinary
# strings. Parsing it is the point; the warnings about it are noise.
warnings.filterwarnings("ignore", category=SyntaxWarning)

FENCE = "`" * 3
STATEMENT = re.compile(r"^(def |class |import |from |@|async def )")
MIN_LINES = 3


def parses(src):
    try:
        ast.parse(src)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def fence_free(raw):
    """Code starting at the first statement line, trimmed until it parses."""
    lines = raw.splitlines()
    start = next((i for i, l in enumerate(lines) if STATEMENT.match(l)), None)
    if start is None:
        return None
    body = lines[start:]
    for end in range(len(body), 0, -1):
        src = "\n".join(body[:end])
        if len(src.strip().splitlines()) >= MIN_LINES and parses(src):
            return src
    return None


def check(lang_dir):
    log = lang_dir / "results.jsonl"
    if not log.exists():
        return None
    replies = {}
    # errors="replace": one CodeLlama log has bytes that are not UTF-8.
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
            replies.setdefault(r["prompt_id"], r.get("raw_response") or "")
        except (json.JSONDecodeError, KeyError):
            continue
    n = scored = recovered = unfenced = 0
    for pid, raw in replies.items():
        code_file = lang_dir / "code" / f"{pid}.py"
        if not code_file.exists():
            continue
        n += 1
        ok = parses(code_file.read_text(encoding="utf-8", errors="replace"))
        scored += ok
        if FENCE not in raw:
            unfenced += 1
        if ok or (FENCE not in raw and fence_free(raw)):
            recovered += 1
    if not n:
        return None
    return {"n": n,
            "valid_scored_pct": round(100.0 * scored / n, 2),
            "valid_fence_free_pct": round(100.0 * recovered / n, 2),
            "unfenced_pct": round(100.0 * unfenced / n, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="generation root (Python layout)")
    ap.add_argument("--out", default="_analysis_fullsize")
    ap.add_argument("--exclude", action="append", default=["mamba-codestral-7b"])
    args = ap.parse_args()

    rows = []
    for mdir in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        if mdir.name in args.exclude or mdir.name in ("java", "cpp", "c"):
            continue
        for lang in ("en", "es", "fr", "zh"):
            res = check(mdir / lang)
            if res:
                rows.append({"model": mdir.name, "lang": lang, **res})
    out = Path(args.out) / "extraction_check.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        gain = r["valid_fence_free_pct"] - r["valid_scored_pct"]
        if gain >= 5:
            print(f"  {r['model']:22s} {r['lang']}  {r['valid_scored_pct']:5.1f}% -> "
                  f"{r['valid_fence_free_pct']:5.1f}%   ({r['unfenced_pct']:.0f}% of replies unfenced)")


if __name__ == "__main__":
    main()
