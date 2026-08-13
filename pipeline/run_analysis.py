#!/usr/bin/env python3
"""
Analysis pipeline - score generated outputs with the project's own smell detector.

Why this exists
---------------
Two analysis paths grew up separately and neither does the whole job:

  * pipeline/run_detection.py calls detector/smell_detector.py -- the eight
    smell-targeted checks this study is actually about -- but only understands the
    flat outputs/<model>/code/ layout, and reports no code-quality metrics.
  * The Colab notebooks compute radon and ruff metrics over the nested
    <model>/<lang>/code/ layout, but embed their own panel of generic hygiene
    checks (bare_except, print_call, placeholder_stub, ...) and never import the
    detector. Of the 25 smells the prompt set targets, that panel corresponds to
    two.

So the targeted smells were never measured on the multilingual outputs at all.
This module runs both instruments over both layouts, and adds the one control the
notebooks were missing.

The syntax-validity control
---------------------------
ruff_per_100loc is the notebooks' headline quality metric, and it collapses its own
denominator: when a model emits something that does not parse, LOC is tiny and the
per-100-line rate explodes. Measured across this corpus, files that parse average
6.4 violations per 100 lines and files that do not average 1,182 -- so the metric
is mostly a proxy for "did generation break". Every quality figure below is
therefore reported twice, over all files and over syntactically valid files only.
The gap between those two columns is the part of any language or model effect that
is really a generation-failure effect.

Usage
-----
    # nested layout: <root>/<model>/<lang>/code/*.py
    python -m pipeline.run_analysis --root /path/to/outputs_multilingual

    # flat layout: <root>/<model>/code/*.py  (treated as lang=en)
    python -m pipeline.run_analysis --root /path/to/code-smell-outputs --lang en

    # both at once, written to one set of CSVs
    python -m pipeline.run_analysis --root A --root B --out _analysis

    # leave a model out of the aggregates (keeps it in per_file.csv)
    python -m pipeline.run_analysis --root A --exclude mamba-codestral-7b
"""

import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.smell_detector import detect_all_smells
from detector.extended_smells import UNDETECTABLE, detect_extended_smells

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    from radon.raw import analyze as raw_analyze
    HAVE_RADON = True
except ImportError:                                    # radon is optional
    HAVE_RADON = False

DATASET_PATH = PROJECT_ROOT / "dataset" / "prompts_core.json"


def _find_ruff():
    """Prefer the ruff belonging to the interpreter running this.

    shutil.which() only sees PATH, so an unactivated virtualenv -- which is what
    `path/to/.venv/bin/python -m pipeline.run_analysis` gives you -- silently finds
    nothing and every file scores zero ruff violations. Look next to the
    interpreter first, then fall back to PATH.
    """
    bindir = Path(sys.executable).parent
    for name in ("ruff.exe", "ruff"):
        cand = bindir / name
        if cand.exists():
            return str(cand)
    return shutil.which("ruff")


RUFF = _find_ruff()

# The detector's smell labels against the dataset's code_smells vocabulary.
# Written out rather than fuzzy-matched: "Data Class" and "Data Clumps" are
# different smells and a substring rule is one edit away from conflating them.
DETECTOR_TO_DATASET = {
    # detector/smell_detector.py
    "Long Method":          "Long Method",
    "Long Parameter List":  "Long Parameter List",
    "Duplicated Code":      "Duplicated Code",
    "Deep Nesting":         "Deep Nesting",
    "God Class":            "God Class / Large Class",
    "Magic Number":         "Magic Numbers/Strings",
    "Global State":         "Global State",
    # check_global_state emits two labels; without this alias every file whose
    # only global signal is the `global` keyword scored as a miss.
    "Global State (global keyword)": "Global State",
    "Data Class":           "Data Class",
    # detector/extended_smells.py -- labels already match the dataset exactly
    "Data Clumps":                       "Data Clumps",
    "Message Chains":                    "Message Chains",
    "Feature Envy":                      "Feature Envy",
    "Middle Man":                        "Middle Man",
    "Lazy Class":                        "Lazy Class",
    "Switch Statements":                 "Switch Statements",
    "Dead Code":                         "Dead Code",
    "Temporary Field":                   "Temporary Field",
    "Inappropriate Intimacy":            "Inappropriate Intimacy",
    "Comments (as smell indicator)":     "Comments (as smell indicator)",
    "Refused Bequest":                   "Refused Bequest",
    "Speculative Generality":            "Speculative Generality",
    "Parallel Inheritance Hierarchies":  "Parallel Inheritance Hierarchies",
}
COVERED_SMELLS = set(DETECTOR_TO_DATASET.values())

CAT_RE = re.compile(r"(.+?)_(basic|intermediate|advanced)_\d+$")


# ----------------------------------------------------------------- collection

def load_prompts():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)}


def discover(root, fixed_lang=None):
    """Yield (model, lang, code_dir) for either output layout."""
    root = Path(root)
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if mdir.name.startswith("_") or mdir.name.startswith("."):
            continue
        if (mdir / "code").is_dir():                   # flat: <model>/code
            yield mdir.name, fixed_lang or "en", mdir / "code"
        for ldir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            if (ldir / "code").is_dir():               # nested: <model>/<lang>/code
                yield mdir.name, ldir.name, ldir / "code"


def ruff_for_dir(code_dir):
    """One ruff invocation per directory -- per-file would dominate runtime."""
    out = defaultdict(list)
    if RUFF is None:
        return out
    try:
        proc = subprocess.run(
            [RUFF, "check", "--select", "E,F,W,B,C90,SIM,PLR,PLW,N,UP,RUF,S",
             "--ignore", "E501", "--output-format", "json", "--no-cache", str(code_dir)],
            capture_output=True,
            # ruff emits UTF-8. Without this, Windows decodes with the ANSI
            # codepage and any accented identifier in the output -- which the
            # translated prompts produce plenty of -- raises UnicodeDecodeError,
            # losing the whole directory's lint results to the except below.
            encoding="utf-8", errors="replace")
        for v in json.loads(proc.stdout or "[]"):
            out[os.path.basename(v.get("filename", ""))].append(v.get("code") or "UNKNOWN")
    except Exception as e:                             # ruff missing or misbehaving
        print(f"  ruff failed on {code_dir}: {e}", file=sys.stderr)
        raise SystemExit(f"aborting: lint results for {code_dir} would be silently "
                         f"zero, which reads as a finding rather than a failure")
    return out


# -------------------------------------------------------------------- scoring

def score_file(path, src, ruff_codes, prompts):
    prompt_id = path.stem
    info = prompts.get(prompt_id, {})
    targets = info.get("code_smells", [])

    m = CAT_RE.match(prompt_id)
    row = {
        "prompt_id": prompt_id,
        "category": m.group(1) if m else prompt_id,
        "level": info.get("complexity") or (m.group(2) if m else "unknown"),
        "domain": info.get("domain", "unknown"),
        "target_smells": ";".join(targets),
    }

    # --- does it parse? every quality number below is conditioned on this
    try:
        ast.parse(src)
        row["syntax_ok"] = 1
    except SyntaxError:
        row["syntax_ok"] = 0

    # --- size and structure
    row["loc"] = 0
    row["cc_max"] = 0
    row["mi"] = ""
    if HAVE_RADON:
        try:
            row["loc"] = raw_analyze(src).loc
        except Exception:
            row["loc"] = src.count("\n") + 1
        if row["syntax_ok"]:
            try:
                blocks = cc_visit(src)
                row["cc_max"] = max((b.complexity for b in blocks), default=0)
            except Exception:
                pass
            try:
                row["mi"] = round(mi_visit(src, multi=True), 2)
            except Exception:
                pass
    else:
        row["loc"] = src.count("\n") + 1

    # --- ruff
    row["ruff_total"] = len(ruff_codes)
    row["undefined_name"] = sum(1 for c in ruff_codes if c == "F821")
    row["unused_var"] = sum(1 for c in ruff_codes if c == "F841")

    # --- the project's own detector plus the extension: what the study is asking
    det = detect_all_smells(src)
    ext = detect_extended_smells(src)
    raw = set(det["smell_types_detected"]) | {s["smell"] for s in ext}
    # Normalise to the dataset's vocabulary here, so everything downstream is a
    # plain equality. check_global_state emits both "Global State" and "Global
    # State (global keyword)"; before this, files carrying only the second scored
    # as misses.
    found = {DETECTOR_TO_DATASET.get(s, s) for s in raw}
    row["n_smells"] = det["total_smells"] + len(ext)
    row["smells_found"] = ";".join(sorted(found))

    # Was the smell the prompt asked for actually produced? Blank when no
    # detector covers the targeted smell -- that is different from a miss, and
    # collapsing the two would silently score those categories as failures.
    covered = [t for t in targets if t in COVERED_SMELLS]
    if covered:
        hits = sum(1 for t in covered if t in found)
        row["target_covered"] = 1
        row["target_hit"] = int(hits == len(covered))
    else:
        row["target_covered"] = 0
        row["target_hit"] = ""
    return row


def collect(roots, fixed_lang, prompts):
    rows = []
    for root in roots:
        for model, lang, code_dir in discover(root, fixed_lang):
            files = sorted(code_dir.glob("*.py"))
            if not files:
                continue
            print(f"  [{model}/{lang}] {len(files)} files", flush=True)
            ruff_map = ruff_for_dir(code_dir)
            for f in files:
                try:
                    src = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                row = score_file(f, src, ruff_map.get(f.name, []), prompts)
                row["model"], row["lang"] = model, lang
                rows.append(row)
    return rows


# ---------------------------------------------------------------- aggregation

def _rate(rows, num, den):
    n = sum(r[num] for r in rows)
    d = sum(r[den] for r in rows)
    return round(100.0 * n / d, 2) if d else ""


def _pct(rows, key):
    return round(100.0 * sum(r[key] for r in rows) / len(rows), 2) if rows else ""


def _mean(rows, key):
    vals = [r[key] for r in rows if r[key] != ""]
    return round(sum(vals) / len(vals), 2) if vals else ""


def summarise(rows, keys):
    """One aggregate row, with every quality figure given twice."""
    valid = [r for r in rows if r["syntax_ok"]]
    out = dict(keys)
    out["n_files"] = len(rows)
    out["syntax_ok_pct"] = _pct(rows, "syntax_ok")
    out["ruff_per_100loc_all"] = _rate(rows, "ruff_total", "loc")
    out["ruff_per_100loc_valid"] = _rate(valid, "ruff_total", "loc")
    out["mi_all"] = _mean(rows, "mi")
    out["mi_valid"] = _mean(valid, "mi")
    out["loc_mean"] = _mean(rows, "loc")

    cov = [r for r in rows if r["target_covered"]]
    cov_valid = [r for r in cov if r["syntax_ok"]]
    out["n_covered"] = len(cov)
    out["induction_all"] = _pct(cov, "target_hit")
    out["induction_valid"] = _pct(cov_valid, "target_hit")
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True,
                    help="output root; repeat for several (flat or nested layout)")
    ap.add_argument("--lang", default=None,
                    help="language label for a flat <model>/code layout (default 'en')")
    ap.add_argument("--out", default="_analysis", help="output directory")
    ap.add_argument("--exclude", action="append", default=[],
                    help="model to leave out of the aggregates; repeatable")
    ap.add_argument("--no-ruff", action="store_true",
                    help="score without ruff (lint columns will all be zero)")
    args = ap.parse_args()

    global RUFF
    if args.no_ruff:
        RUFF = None

    prompts = load_prompts()
    print(f"{len(prompts)} prompts in the dataset")
    if not HAVE_RADON:
        print("  note: radon not installed -- loc/cc/mi will be limited", file=sys.stderr)
    if RUFF is None and not args.no_ruff:
        # Refuse rather than emit a table of zeroes that looks like a finding.
        sys.exit("ruff not found next to this interpreter or on PATH.\n"
                 "  install it into the environment you are running:  pip install ruff\n"
                 "  (pass --no-ruff if you really want to score without it)")
    print(f"ruff: {RUFF or 'DISABLED -- lint columns are meaningless'}")

    print("scoring...")
    rows = collect(args.root, args.lang, prompts)
    if not rows:
        sys.exit("no .py files found under the given roots")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "per_file.csv", rows)

    agg = [r for r in rows if r["model"] not in args.exclude]
    if args.exclude:
        print(f"  excluded from aggregates: {', '.join(args.exclude)} "
              f"({len(rows) - len(agg)} files)")

    langs = sorted({r["lang"] for r in agg})
    models = sorted({r["model"] for r in agg})

    # Matched subset: the (model, prompt_id) pairs generated in EVERY language.
    # Without this, English is compared on all 426 prompts and the other languages
    # on the 75-prompt pilot subset, so a language contrast is partly a contrast
    # between different task mixes -- the categories are not equally represented
    # in the two sets.
    seen = defaultdict(set)
    for r in agg:
        seen[(r["model"], r["prompt_id"])].add(r["lang"])
    matched_keys = {k for k, v in seen.items() if v >= set(langs)}
    matched = [r for r in agg if (r["model"], r["prompt_id"]) in matched_keys]
    print(f"  matched subset: {len(matched)} files "
          f"({len(matched_keys)} model-prompt pairs present in all {len(langs)} languages)")

    write_csv(outdir / "by_lang.csv",
              [summarise([r for r in agg if r["lang"] == l], {"lang": l}) for l in langs])
    write_csv(outdir / "by_lang_matched.csv",
              [summarise([r for r in matched if r["lang"] == l], {"lang": l}) for l in langs])
    write_csv(outdir / "by_model_lang.csv",
              [summarise([r for r in agg if r["model"] == m and r["lang"] == l],
                         {"model": m, "lang": l})
               for m in models for l in langs
               if any(r["model"] == m and r["lang"] == l for r in agg)])

    # Induction rate per targeted smell -- the study's actual question -- with a
    # discrimination check beside it. A detector that fires just as often on files
    # nobody asked for that smell is not detecting anything; it is a base rate
    # wearing the smell's name, and the lift column makes that visible.
    by_smell = []
    valid = [r for r in agg if r["syntax_ok"]]
    for label in sorted(COVERED_SMELLS):
        g = [r for r in agg if label in r["target_smells"].split(";")]
        if not g:
            continue
        row = summarise(g, {"target_smell": label})

        off = [r for r in valid if label not in r["target_smells"].split(";")]
        fires = sum(1 for r in off if label in r["smells_found"].split(";"))
        base = round(100.0 * fires / len(off), 2) if off else ""
        row["base_rate"] = base
        row["lift"] = (round(row["induction_valid"] - base, 2)
                       if row["induction_valid"] != "" and base != "" else "")

        for l in langs:
            gl = [r for r in g if r["lang"] == l and r["syntax_ok"]]
            row[f"induction_{l}"] = _pct(gl, "target_hit")
        by_smell.append(row)
    write_csv(outdir / "by_smell.csv", by_smell)

    write_csv(outdir / "by_category.csv",
              [summarise([r for r in agg if r["category"] == c], {"category": c})
               for c in sorted({r["category"] for r in agg})])
    write_csv(outdir / "by_level.csv",
              [summarise([r for r in agg if r["level"] == lv], {"level": lv})
               for lv in sorted({r["level"] for r in agg})])

    # ------------------------------------------------------------- report
    print("\n" + "=" * 74)
    print("SYNTAX VALIDITY AND QUALITY, BY PROMPT LANGUAGE")
    print("=" * 74)
    for title, data in (("all files scored", agg), ("matched prompts only", matched)):
        print(f"\n  {title}")
        print(f"  {'lang':6s}{'files':>7s}{'valid%':>9s}{'ruff/100loc':>13s}"
              f"{'  (valid only)':>15s}{'induction':>11s}")
        for l in langs:
            s = summarise([r for r in data if r["lang"] == l], {})
            if not s["n_files"]:
                continue
            ind = s["induction_valid"]
            print(f"  {l:6s}{s['n_files']:7d}{s['syntax_ok_pct']:9.1f}"
                  f"{s['ruff_per_100loc_all']:13.2f}{s['ruff_per_100loc_valid']:15.2f}"
                  f"{(f'{ind:.1f}%' if ind != '' else '--'):>11s}")
    print("\n  The gap between the two ruff columns is generation failure, not code")
    print("  quality. The matched block is the one to quote: it holds the prompt set")
    print("  fixed, so a language difference cannot be a difference in task mix.")

    print("\n" + "=" * 74)
    print("INDUCTION RATE -- prompt asks for the smell, detector confirms it")
    print("=" * 74)
    print(f"  {'targeted smell':34s}{'files':>7s}{'asked':>8s}{'not asked':>11s}{'lift':>8s}")
    for r in sorted(by_smell, key=lambda x: -(x["lift"] if x["lift"] != "" else -999)):
        flag = "" if (r["lift"] or 0) >= 20 else "   <- weak"
        print(f"  {r['target_smell']:34s}{r['n_covered']:7d}"
              f"{r['induction_valid']:7.1f}%{r['base_rate']:10.1f}%"
              f"{r['lift']:+8.1f}{flag}")
    print("\n  'not asked' is how often the detector fires on files targeting some")
    print("  other smell. Lift is the difference: a detector with low lift is")
    print("  reporting a base rate, not detecting what it claims to.")

    cov = [r for r in agg if r["target_covered"]]
    print(f"\n  {len(cov)} of {len(agg)} files target a smell the detector covers "
          f"({100 * len(cov) / len(agg):.0f}%).")
    uncovered = sorted({t for r in agg for t in r["target_smells"].split(";")
                        if t and t not in COVERED_SMELLS})
    print(f"  {len(uncovered)} targeted smells still have no detector: "
          f"{', '.join(uncovered)}")
    if uncovered and sorted(uncovered) != sorted(UNDETECTABLE):
        print(f"  (detector/extended_smells.py declares these unreachable by a "
              f"single-file rule: {', '.join(UNDETECTABLE)})")


if __name__ == "__main__":
    main()
