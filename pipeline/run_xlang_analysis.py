#!/usr/bin/env python3
"""
Score the cross-language generations: Python, Java, C++ and C, one instrument.

Why this is separate from run_analysis.py
-----------------------------------------
run_analysis.py answers "does the prompt language change the output?" and leans on
two Python-only tools -- ruff for lint density and radon for maintainability -- plus
the 21-smell Python detector. None of those exist for Java, C++ and C, and the
obvious substitutes (PMD, clang-tidy) would give each language its own definition of
"long method", which is precisely the comparison this phase exists to make. See
detector/cross_language.py.

So this module measures a deliberately smaller thing across all four programming
languages, with one detector and one set of thresholds:

    syntax validity   -- does it parse at all
    induction         -- was the requested smell actually produced
    lift              -- induction minus the rate the same detector fires on files
                         that asked for some other smell, computed per language

The Python arm is re-scored here with the same tree-sitter detector rather than
reusing the numbers in _analysis_fullsize. Comparing Java measured by tree-sitter
against Python measured by the 21-smell detector would confound the language with
the instrument, and Python is the reference the other three are read against.

One asymmetry in the validity measure
-------------------------------------
Validity here is "tree-sitter finds no ERROR node", which is not quite ast.parse.
On 426 Python files the two agree 424 times; the two disagreements are both
IndentationError, which tree-sitter's error recovery absorbs. That leniency is
Python-only -- Java, C++ and C are brace-delimited, so indentation cannot make
them invalid -- and at 0.5% it does not move a headline. It is recorded because
the Python column is the reference the other three are read against, and a
reference measured slightly more leniently than it looks is worth knowing about.

Two denominators, and they are different
----------------------------------------
Validity is measured on every file. Induction is measured only on prompts whose
target smell this detector can decide -- 114 of 426, because six smells are
tree-shaped and the other nineteen are not. A smell that cannot be decided is
recorded as blank, never as a miss: scoring "we cannot tell" as "the model failed"
would manufacture a result out of a gap in the instrument.

    python -m pipeline.run_xlang_analysis --root ../outputs_arc --out _analysis_xlang
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.cross_language import APPLICABLE, LANGUAGES, detect, parser_for
from pipeline.run_analysis import CAT_RE, _mean, _pct, write_csv

# Source extension per programming language. Python keeps the flat <model>/<lang>
# layout it was generated in; the others sit under a <plang>/ level.
EXT = {"python": ".py", "java": ".java", "cpp": ".cpp", "c": ".c"}

PROMPT_FILE = {
    "python": "prompts_core.json",
    "java": "prompts_core_java.json",
    "cpp": "prompts_core_cpp.json",
    "c": "prompts_core_c.json",
}


def load_prompts(plang):
    path = PROJECT_ROOT / "dataset" / PROMPT_FILE[plang]
    if not path.exists():
        sys.exit(f"no {plang} prompt set at {path}")
    with open(path, encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)}


def decidable_for(plang):
    """The smells this detector can decide in this language -- not all 25."""
    return {s for s, langs in APPLICABLE.items() if plang in langs}


def discover(root):
    """Yield (plang, model, nat_lang, code_dir) across both output layouts."""
    root = Path(root)
    for plang in LANGUAGES:
        base = root if plang == "python" else root / plang
        if not base.is_dir():
            continue
        for mdir in sorted(p for p in base.iterdir() if p.is_dir()):
            # The cross-language trees live beside the Python models at the top
            # level, so skip them when walking the Python arm.
            if plang == "python" and mdir.name in LANGUAGES:
                continue
            if mdir.name.startswith((".", "_")):
                continue
            for ldir in sorted(p for p in mdir.iterdir() if p.is_dir()):
                if (ldir / "code").is_dir():
                    yield plang, mdir.name, ldir.name, ldir / "code"


def _count_errors(node):
    n = 0
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type == "ERROR" or cur.is_missing:
            n += 1
        stack.extend(cur.children)
    return n


def score_file(path, src, plang, prompts, decidable):
    prompt_id = path.stem
    info = prompts.get(prompt_id, {})
    targets = info.get("code_smells", [])
    m = CAT_RE.match(prompt_id)

    row = {
        "plang": plang,
        "prompt_id": prompt_id,
        "category": m.group(1) if m else prompt_id,
        "level": info.get("complexity") or (m.group(2) if m else "unknown"),
        "target_smells": ";".join(targets),
    }

    # --- does it parse? tree-sitter is error-tolerant and never raises, so the
    # signal is ERROR/MISSING nodes in the tree rather than an exception. Counting
    # them as well as flagging them separates one stray brace from "not code".
    data = src.encode("utf-8", "replace")
    try:
        tree = parser_for(plang).parse(data)
        root = tree.root_node
        row["syntax_ok"] = 0 if root.has_error else 1
        row["n_error_nodes"] = _count_errors(root) if root.has_error else 0
    except Exception as e:
        print(f"  parse failed on {prompt_id} ({plang}): {type(e).__name__}",
              file=sys.stderr)
        row["syntax_ok"], row["n_error_nodes"] = 0, -1

    # LOC counted the same way in every language: non-blank source lines. radon is
    # Python-only, and a metric that exists for one arm of a four-way comparison is
    # worse than one that exists for all four.
    row["loc"] = sum(1 for ln in src.splitlines() if ln.strip())

    try:
        found_list = detect(data, plang)
    except Exception as e:
        print(f"  detector failed on {prompt_id} ({plang}): {type(e).__name__}",
              file=sys.stderr)
        found_list = []
    found = {s["smell"] for s in found_list}
    row["n_smells"] = len(found_list)
    row["smells_found"] = ";".join(sorted(found))

    # Only prompts whose target this detector can decide in THIS language count
    # towards induction. God Class is undecidable in C, so those prompts are blank
    # for C and scored for the other three -- not counted as C failures.
    covered = [t for t in targets if t in decidable]
    if covered:
        row["target_covered"] = 1
        row["target_hit"] = int(all(t in found for t in covered))
    else:
        row["target_covered"] = 0
        row["target_hit"] = ""
    return row


def collect(root):
    rows, prompt_cache = [], {}
    for plang, model, nat_lang, code_dir in discover(root):
        files = sorted(code_dir.glob("*" + EXT[plang]))
        if not files:
            continue
        if plang not in prompt_cache:
            prompt_cache[plang] = (load_prompts(plang), decidable_for(plang))
        prompts, decidable = prompt_cache[plang]
        print(f"  [{plang}/{model}/{nat_lang}] {len(files)} files", flush=True)
        for f in files:
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            row = score_file(f, src, plang, prompts, decidable)
            row["model"], row["lang"] = model, nat_lang
            rows.append(row)
    return rows


def summarise(rows, keys):
    valid = [r for r in rows if r["syntax_ok"]]
    out = dict(keys)
    out["n_files"] = len(rows)
    out["syntax_ok_pct"] = _pct(rows, "syntax_ok")
    out["loc_mean"] = _mean(rows, "loc")
    out["loc_mean_valid"] = _mean(valid, "loc")
    cov = [r for r in rows if r["target_covered"]]
    cov_valid = [r for r in cov if r["syntax_ok"]]
    out["n_decidable"] = len(cov)
    out["induction_all"] = _pct(cov, "target_hit")
    out["induction_valid"] = _pct(cov_valid, "target_hit")
    return out


def base_rate(rows, smell):
    """How often the detector fires on files that asked for a different smell.

    Computed per language, never pooled. The thresholds are shared across the four
    grammars, but that does not make the base rates equal -- a magic number is far
    more common in C than in Python for reasons that have nothing to do with the
    prompt -- so a raw induction figure is not comparable across languages and the
    lift is.
    """
    off = [r for r in rows
           if smell not in r["target_smells"].split(";") and r["syntax_ok"]]
    if not off:
        return ""
    fires = sum(1 for r in off if smell in r["smells_found"].split(";"))
    return round(100.0 * fires / len(off), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="generation root (holds both layouts)")
    ap.add_argument("--out", default="_analysis_xlang")
    ap.add_argument("--exclude", action="append", default=[],
                    help="model to leave out of the aggregates; repeatable")
    args = ap.parse_args()

    print("scoring with the tree-sitter detector (all four languages)...")
    rows = collect(args.root)
    if not rows:
        sys.exit(f"no generations found under {args.root}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "per_file.csv", rows)

    agg = [r for r in rows if r["model"] not in args.exclude]
    if args.exclude:
        print(f"  excluded from aggregates: {', '.join(args.exclude)}")

    plangs = [p for p in LANGUAGES if any(r["plang"] == p for r in agg)]
    models = sorted({r["model"] for r in agg})

    # The programming-language comparison is English-only: the ported prompt sets
    # were never translated, so a Spanish Java cell does not exist. Mixing the
    # Python arm's four natural languages into a contrast against English-only Java
    # would put the prompt language inside the programming-language effect.
    en = [r for r in agg if r["lang"] == "en"]

    # Matched subset: (model, prompt_id) present in EVERY programming language.
    # C drops the eight class-based smells, so without this the C column is scored
    # on a different 292-prompt subset than the other three.
    seen = defaultdict(set)
    for r in en:
        seen[(r["model"], r["prompt_id"])].add(r["plang"])
    keys = {k for k, v in seen.items() if v >= set(plangs)}
    matched = [r for r in en if (r["model"], r["prompt_id"]) in keys]
    print(f"  matched subset: {len(matched)} files "
          f"({len(keys)} model-prompt pairs present in all {len(plangs)} languages)")

    write_csv(outdir / "by_plang.csv",
              [summarise([r for r in en if r["plang"] == p], {"plang": p})
               for p in plangs])
    write_csv(outdir / "by_plang_matched.csv",
              [summarise([r for r in matched if r["plang"] == p], {"plang": p})
               for p in plangs])
    write_csv(outdir / "by_plang_model.csv",
              [summarise([r for r in en if r["plang"] == p and r["model"] == m],
                         {"plang": p, "model": m})
               for p in plangs for m in models
               if any(r["plang"] == p and r["model"] == m for r in en)])

    by_smell = []
    for smell in sorted(APPLICABLE):
        for p in plangs:
            if p not in APPLICABLE[smell]:
                continue
            g = [r for r in en if r["plang"] == p
                 and smell in r["target_smells"].split(";")]
            if not g:
                continue
            row = summarise(g, {"target_smell": smell, "plang": p})
            row["base_rate"] = base_rate([r for r in en if r["plang"] == p], smell)
            row["lift"] = (round(row["induction_valid"] - row["base_rate"], 2)
                           if row["induction_valid"] != "" and row["base_rate"] != ""
                           else "")
            by_smell.append(row)
    write_csv(outdir / "by_plang_smell.csv", by_smell)

    # ------------------------------------------------------------------ report
    print("\n" + "=" * 78)
    print("SYNTAX VALIDITY AND INDUCTION, BY PROGRAMMING LANGUAGE (English prompts)")
    print("=" * 78)
    for title, data in (("all prompts", en), ("matched across all four", matched)):
        print(f"\n  {title}")
        print(f"  {'plang':9s}{'files':>7s}{'valid%':>9s}{'decidable':>11s}"
              f"{'induction':>11s}{'loc':>8s}")
        for p in plangs:
            s = summarise([r for r in data if r["plang"] == p], {})
            if not s["n_files"]:
                continue
            ind = s["induction_valid"]
            print(f"  {p:9s}{s['n_files']:7d}{s['syntax_ok_pct']:9.1f}"
                  f"{s['n_decidable']:11d}"
                  f"{(f'{ind:.1f}%' if ind != '' else '--'):>11s}"
                  f"{s['loc_mean']:8.1f}")
    print("\n  'decidable' is how many files target a smell this detector can decide")
    print("  in that language; induction is computed over those only. The matched")
    print("  block holds the prompt set fixed, so a language difference cannot be a")
    print("  difference in which prompts that language was asked.")

    print("\n" + "=" * 78)
    print("INDUCTION AGAINST BASE RATE, PER SMELL AND LANGUAGE")
    print("=" * 78)
    print(f"  {'smell':26s}{'plang':>8s}{'files':>7s}{'asked':>8s}"
          f"{'not asked':>11s}{'lift':>8s}")
    # Every one of these can be blank -- induction when no decidable prompt
    # survived parsing, base rate when nothing targeted a different smell -- and a
    # blank is not a zero. Format defensively rather than crash the report after
    # the CSVs are already written.
    def _f(v, width, suffix=""):
        return (f"{v:{width}.1f}{suffix}" if isinstance(v, (int, float))
                else f"{'--':>{width}}{' ' * len(suffix)}")

    for r in by_smell:
        lift = r["lift"]
        flag = "" if isinstance(lift, (int, float)) and lift >= 20 else "  <- weak"
        if not isinstance(lift, (int, float)):
            flag = "  <- not measurable"
        print(f"  {r['target_smell']:26s}{r['plang']:>8s}{r['n_decidable']:7d}"
              f"{_f(r['induction_valid'], 7, '%')}{_f(r['base_rate'], 10, '%')}"
              f"{_f(lift, 8)}{flag}")
    print("\n  Thresholds are identical across the four grammars, but base rates are")
    print("  not -- so the lift column is the comparable one, and a raw induction")
    print("  rate read across languages will mislead.")


if __name__ == "__main__":
    main()
