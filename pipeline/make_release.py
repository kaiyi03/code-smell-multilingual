#!/usr/bin/env python3
"""
Package the generations, scores and run metadata as a GitHub Release asset.

The repository holds code and aggregate CSVs and is 2 MB. The actual experimental
output -- ~19,000 generated files and their per-file scores -- is 156 MB and lives
only on ARC scratch and one laptop. ARC's $DATA is not a backed-up archive, so
losing it means regenerating, which is the GPU time this whole phase spent.

A Release asset is the right home: durable, versioned, and downloaded only by
people who want it, so it does not bloat every clone.

    python -m pipeline.make_release --outputs ../outputs_arc --dry-run
    python -m pipeline.make_release --outputs ../outputs_arc --tag data-2026-09

Requires the gh CLI, authenticated.
"""

import argparse
import csv
import json
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGS = ("en", "es", "fr", "zh")
FULL = 426


def survey(outputs: Path):
    """What is actually here, per model and language."""
    rows, totals = [], Counter()
    for mdir in sorted(p for p in outputs.iterdir() if p.is_dir()):
        counts = {}
        for lang in LANGS:
            n = len(list((mdir / lang / "code").glob("*.py"))) if (mdir / lang / "code").is_dir() else 0
            counts[lang] = n
            totals[lang] += n
        rows.append((mdir.name, counts))
    return rows, totals


def write_manifest(outputs: Path, rows, totals, dest: Path):
    """A README inside the archive, so it is readable without this repo."""
    complete = [m for m, c in rows if all(c[l] == FULL for l in LANGS)]
    partial = [(m, c) for m, c in rows if 0 < sum(c.values()) < FULL * len(LANGS)]
    empty = [m for m, c in rows if sum(c.values()) == 0]

    lines = [
        "# Multilingual code-smell study — generations and scores",
        "",
        f"Packaged {date.today().isoformat()} from the ARC cluster run.",
        "Source: https://github.com/kaiyi03/code-smell-multilingual",
        "Results: https://kaiyi03.github.io/code-smell-multilingual/",
        "",
        "## What this is",
        "",
        "Open-weight code models asked, in four human languages, to write Python",
        "containing a named code smell. 426 prompts covering 25 smells, generated",
        "with greedy decoding at 2048 new tokens.",
        "",
        "## Layout",
        "",
        "```",
        "outputs_arc/<model>/<lang>/code/<prompt_id>.py   extracted code, one per prompt",
        "outputs_arc/<model>/<lang>/results.jsonl         raw response, prompt, tokens, timing",
        "analysis/per_file.csv                            every file scored, one row each",
        "analysis/by_*.csv                                aggregates used by the results page",
        "prompts/prompts_core*.json                       the prompt sets, incl. ported ones",
        "```",
        "",
        "`results.jsonl` carries the untouched model response alongside the extracted",
        "code, so an extraction bug can be diagnosed without regenerating.",
        "",
        "## Coverage",
        "",
        f"| Model | {' | '.join(LANGS)} |",
        f"|---|{'---|' * len(LANGS)}",
    ]
    for m, c in rows:
        cells = " | ".join(str(c[l]) if c[l] else "—" for l in LANGS)
        lines.append(f"| {m} | {cells} |")
    lines += [
        "",
        f"{len(complete)} models complete at {FULL} prompts in all four languages.",
        "",
    ]
    if partial:
        lines.append("Partial: " + ", ".join(f"{m}" for m, _ in partial))
    if empty:
        lines += [
            "",
            "Absent, and why:",
            "",
            "- `codellama-7b` — gated Meta repository, access not yet granted.",
            "- `deepseek-coder-v2-lite` — bundled modelling code targets an older",
            "  transformers; needs its own pinned environment.",
            "- `mamba-codestral-7b` — loads, but emits call sites rather than",
            "  definitions. Excluded from every aggregate rather than pooled.",
        ]
    lines += [
        "",
        "## Caveats worth knowing before using this",
        "",
        "- Numbers should be read on files that **parse**. A file that is not valid",
        "  Python has no code quality to measure, and the per-100-line metrics",
        "  explode on it: parsing files average 6.4 violations per 100 lines,",
        "  non-parsing ones average 1,182.",
        "- Language comparisons should use prompts present in **every** language.",
        "- An earlier 75-prompt pilot is superseded. It selected one prompt per",
        "  smell per level by taking the first alphabetically, which is systematic",
        "  rather than random, and it made Chinese look 9 points worse than the",
        "  full set shows.",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return complete, partial, empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="../outputs_arc")
    ap.add_argument("--analysis", default="_analysis_fullsize")
    ap.add_argument("--tag", default=f"data-{date.today().isoformat()}")
    ap.add_argument("--dry-run", action="store_true", help="build the archive, do not upload")
    args = ap.parse_args()

    outputs = Path(args.outputs).resolve()
    if not outputs.is_dir():
        sys.exit(f"no generations at {outputs}")
    analysis = (PROJECT_ROOT / args.analysis).resolve()

    rows, totals = survey(outputs)
    print(f"{'model':26s}" + "".join(f"{l:>7s}" for l in LANGS))
    for m, c in rows:
        print(f"{m:26s}" + "".join(f"{c[l] or '—':>7}" for l in LANGS))
    print(f"{'TOTAL':26s}" + "".join(f"{totals[l]:>7d}" for l in LANGS))

    staging = PROJECT_ROOT / "_release"
    staging.mkdir(exist_ok=True)
    manifest = staging / "MANIFEST.md"
    complete, partial, empty = write_manifest(outputs, rows, totals, manifest)
    print(f"\n{len(complete)} complete, {len(partial)} partial, {len(empty)} absent")

    archive = staging / f"code-smell-generations-{args.tag}.tar.gz"
    print(f"\nbuilding {archive.name} ...", flush=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(manifest, arcname="MANIFEST.md")
        tar.add(outputs, arcname="outputs_arc")
        if analysis.is_dir():
            for f in sorted(analysis.glob("*.csv")):
                tar.add(f, arcname=f"analysis/{f.name}")
        for f in sorted((PROJECT_ROOT / "dataset").glob("prompts_core*.json")):
            tar.add(f, arcname=f"prompts/{f.name}")
    size_mb = archive.stat().st_size / 1e6
    print(f"  {size_mb:.0f} MB")
    if size_mb > 2000:
        sys.exit("archive exceeds the 2GB per-asset limit; split it")

    if args.dry_run:
        print(f"\ndry run -- archive left at {archive}")
        return

    notes = (f"Generations, per-file scores and run metadata from the ARC run.\n\n"
             f"{len(complete)} of {len(rows)} models complete at {FULL} prompts in all "
             f"four languages. See MANIFEST.md inside the archive for coverage and "
             f"caveats.\n")
    print(f"\ncreating release {args.tag} ...", flush=True)
    subprocess.run(["gh", "release", "create", args.tag, str(archive),
                    "--title", f"Generations and scores ({args.tag})",
                    "--notes", notes], check=True)
    print("done")


if __name__ == "__main__":
    main()
