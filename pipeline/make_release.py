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

# The cross-language arm and its source extensions. C is short because eight of the
# 25 smells are defined over classes and C has none, so those prompts are dropped
# rather than asked and scored as failures.
PLANGS = {"java": ".java", "cpp": ".cpp", "c": ".c"}
XFULL = {"java": 426, "cpp": 426, "c": 292}

# Keyed by model id and printed only for models that really have no output, so a
# model that has since been generated cannot leave a stale excuse in the manifest.
ABSENT_NOTES = {
    "deepseek-coder-v2-lite": [
        "- `deepseek-coder-v2-lite` — runs in an environment pinned to transformers",
        "  4.41, the version its bundled modelling code was written against; under",
        "  5.x three symbols it imports no longer exist. It also names flash_attn",
        "  inside a branch that never executes, which the import scan cannot tell",
        "  apart from a real dependency. Pinning is not a confound here: yi-coder",
        "  already runs pinned for an unrelated tokenizer reason, and decoding is",
        "  greedy either way.",
    ],
    "mamba-codestral-7b": [
        "- `mamba-codestral-7b` — 20 files only, and they are not usable. Its replies",
        "  restate the task rather than answering it ('The function should take a",
        "  dictionary as input...'), and only 7 of 20 contain a definition anywhere.",
        "  The cause is not established. It was generated on 2026-08-25, inside the",
        "  window when template-less models were sent plain concatenated text rather",
        "  than the Alpaca form they were trained on -- the same defect that made",
        "  starcoder2-3b score 33.6% unparseable against 6.3% once corrected. Until",
        "  it is regenerated on the fixed path, treat it as untested, not as a model",
        "  that failed.",
    ],
}


def survey(outputs: Path):
    """What is actually here, per model and prompt language (the Python arm)."""
    rows, totals = [], Counter()
    for mdir in sorted(p for p in outputs.iterdir() if p.is_dir()):
        if mdir.name in PLANGS:          # a cross-language tree, surveyed below
            continue
        counts = {}
        for lang in LANGS:
            n = len(list((mdir / lang / "code").glob("*.py"))) if (mdir / lang / "code").is_dir() else 0
            counts[lang] = n
            totals[lang] += n
        rows.append((mdir.name, counts))
    return rows, totals


def survey_xlang(outputs: Path):
    """The cross-language arm, which sits one directory deeper.

    Python was generated as <model>/<lang>/code; Java, C++ and C are
    <plang>/<model>/<lang>/code. A survey that knows only the first layout reports
    the second as absent, and the archive would ship without ever saying so.
    """
    rows, totals = [], Counter()
    for plang, ext in PLANGS.items():
        base = outputs / plang
        if not base.is_dir():
            continue
        for mdir in sorted(p for p in base.iterdir() if p.is_dir()):
            n = sum(len(list((ldir / "code").glob("*" + ext)))
                    for ldir in mdir.iterdir()
                    if ldir.is_dir() and (ldir / "code").is_dir())
            rows.append((plang, mdir.name, n))
            totals[plang] += n
    return rows, totals


def write_manifest(outputs: Path, rows, totals, dest: Path, xrows=(), xtotals=None):
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
        "Open-weight code models asked to write code containing a named code smell.",
        "426 prompts covering 25 smells, greedy decoding at 2048 new tokens.",
        "",
        "Two arms, and they vary different things:",
        "",
        "- **Prompt language.** The same Python task asked in English, Spanish,",
        "  French and Chinese.",
        "- **Target language.** The same English task asked for Python, Java, C++",
        "  and C. English only -- the ported prompt sets were never translated, so",
        "  this is not a four-by-four design.",
        "",
        "## Layout",
        "",
        "```",
        "outputs_arc/<model>/<lang>/code/<prompt_id>.py   extracted code, one per prompt",
        "outputs_arc/<model>/<lang>/results.jsonl         raw response, prompt, tokens, timing",
        "outputs_arc/<plang>/<model>/en/code/…            the same, for java, cpp and c",
        "analysis/per_file.csv                            every file scored, one row each",
        "analysis_xlang/per_file.csv                      the cross-language scores",
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
        "### Cross-language arm (English prompts)",
        "",
        "| Language | Model | Files | Of |",
        "|---|---|---:|---:|",
    ]
    for plang, model, n in xrows:
        want = XFULL.get(plang, FULL)
        lines.append(f"| {plang} | {model} | {n} | {want} |")
    if xtotals:
        lines.append("")
        lines.append("Totals: " + ", ".join(f"{p} {xtotals[p]}" for p in PLANGS
                                            if xtotals.get(p)))
    lines += [
        "",
        "C is asked 292 prompts rather than 426: eight of the 25 smells are defined",
        "over classes, which C does not have. Those prompts are dropped rather than",
        "asked and scored as failures, which would have made C look artificially",
        "clean. Any comparison including C must hold the prompt set fixed.",
        "",
    ]
    if partial:
        lines.append("Partial: " + ", ".join(f"{m}" for m, _ in partial))
    if empty:
        lines += ["", "Absent, and why:", ""]
        for m in empty:
            lines += ABSENT_NOTES.get(
                m, [f"- `{m}` — no generations in this package."])
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
    ap.add_argument("--xanalysis", default="_analysis_xlang",
                    help="cross-language scores; skipped if it does not exist")
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

    xrows, xtotals = survey_xlang(outputs)
    if xrows:
        print(f"\n{'cross-language':26s}{'files':>7s}{'of':>7s}")
        for plang, model, n in xrows:
            want = XFULL.get(plang, FULL)
            flag = "" if n == want else "  <- short"
            print(f"{plang + '/' + model:26s}{n:7d}{want:7d}{flag}")
        print(f"{'TOTAL':26s}{sum(xtotals.values()):7d}")

    staging = PROJECT_ROOT / "_release"
    staging.mkdir(exist_ok=True)
    manifest = staging / "MANIFEST.md"
    complete, partial, empty = write_manifest(outputs, rows, totals, manifest,
                                              xrows, xtotals)
    print(f"\n{len(complete)} complete, {len(partial)} partial, {len(empty)} absent")

    archive = staging / f"code-smell-generations-{args.tag}.tar.gz"
    print(f"\nbuilding {archive.name} ...", flush=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(manifest, arcname="MANIFEST.md")
        tar.add(outputs, arcname="outputs_arc")
        if analysis.is_dir():
            for f in sorted(analysis.glob("*.csv")):
                tar.add(f, arcname=f"analysis/{f.name}")
        xanalysis = (PROJECT_ROOT / args.xanalysis).resolve()
        if xanalysis.is_dir():
            for f in sorted(xanalysis.glob("*.csv")):
                tar.add(f, arcname=f"analysis_xlang/{f.name}")
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
