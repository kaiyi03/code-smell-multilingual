#!/usr/bin/env python3
"""
Re-extract every reply with pipeline/extract.py into a parallel output tree.

The generated tree is left exactly as the jobs wrote it. This reads each cell's
untouched replies from results.jsonl and writes, under --out, the same layout:

    <out>/<model>/<lang>/code/<id>.py            Python
    <out>/<plang>/<model>/<lang>/code/<id>.<ext> Java, C++, C

beside a results.jsonl holding what the analyses read from it (the token count)
and which extraction step produced each file. Both analyses then run on --out
unchanged, so the only difference between the two sets of scores is the extractor.

    python -m pipeline.reextract --root ../outputs_arc --out ../outputs_scored
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.extract import extract

PLANGS = ("java", "cpp", "c")
EXT = {"python": "py", "java": "java", "cpp": "cpp", "c": "c"}


def cells(root):
    """Yield (plang, model, lang, cell_dir) for both layouts."""
    for res in sorted(Path(root).rglob("results.jsonl")):
        parts = res.parent.relative_to(root).parts
        if parts[0] in PLANGS and len(parts) == 3:
            yield parts[0], parts[1], parts[2], res.parent
        elif len(parts) == 2:
            yield "python", parts[0], parts[1], res.parent


# A prompt id as the dataset writes them. Two jobs once appended to the same log
# at once, and their interleaved writes left lines that are valid JSON with a run
# of code where the id should be; this keeps those out of the file names.
PROMPT_ID = re.compile(r"^[a-z_]+_(basic|intermediate|advanced)_\d+$")


def replies(cell):
    """The last reply logged for each prompt. A cell written by one job has one each."""
    out, lines = {}, 0
    # errors="replace": a raw reply can carry bytes that are not UTF-8.
    with open(cell / "results.jsonl", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if PROMPT_ID.match(str(r.get("prompt_id", ""))):
                out[r["prompt_id"]] = r
                lines += 1
    return out, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="generation root, as the jobs wrote it")
    ap.add_argument("--out", required=True, help="where the re-extracted tree goes")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)

    totals = Counter()
    for plang, model, lang, cell in cells(root):
        rel = cell.relative_to(root)
        dest = out / rel
        (dest / "code").mkdir(parents=True, exist_ok=True)
        got, lines = replies(cell)
        steps = Counter()
        with open(dest / "results.jsonl", "w", encoding="utf-8") as log:
            for pid, r in sorted(got.items()):
                code, step = extract(r.get("raw_response") or "", plang)
                (dest / "code" / f"{pid}.{EXT[plang]}").write_text(code, encoding="utf-8")
                steps[step] += 1
                log.write(json.dumps({"prompt_id": pid,
                                      "output_tokens": r.get("output_tokens"),
                                      "extraction": step}) + "\n")
        n_code = len(list((cell / "code").glob(f"*.{EXT[plang]}")))
        missing = n_code - len(got)
        note = f"  ({missing} files have no logged reply)" if missing > 0 else ""
        dup = f"  ({lines - len(got)} duplicate log lines)" if lines > len(got) else ""
        print(f"  {str(rel):34s} {len(got):4d}  "
              + "  ".join(f"{k}={v}" for k, v in sorted(steps.items())) + note + dup)
        totals.update(steps)
    print("total: " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))


if __name__ == "__main__":
    main()
