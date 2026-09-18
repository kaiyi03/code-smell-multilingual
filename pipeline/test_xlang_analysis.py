#!/usr/bin/env python3
"""
The cross-language scorer must find every arm and must not invent results.

Two failure modes here are silent, which is why they get a test rather than a
careful read. The first is discovery: Java, C++ and C sit one directory deeper
than Python, and a scorer that walks only one shape reports nothing for the other
and prints a clean-looking table anyway. The second is the undecidable smell: the
tree-sitter detector decides six of the twenty-five, and if the other nineteen are
scored as misses instead of blanks, every language acquires a manufactured failure
rate proportional to how many prompts it was asked.

    python -m pipeline.test_xlang_analysis
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.cross_language import APPLICABLE
from pipeline.run_xlang_analysis import (base_rate, collect, decidable_for,
                                         discover, load_prompts)

LONG_BODY = {
    "python": "\n".join(f"    total = total + {i}" for i in range(20)),
    "java": "\n".join(f"        total = total + {i};" for i in range(20)),
    "cpp": "\n".join(f"        total = total + {i};" for i in range(20)),
    "c": "\n".join(f"        total = total + {i};" for i in range(20)),
}

# A long method with four parameters: Long Method and Long Parameter List both
# fire, in all four languages, so one file exercises the detector everywhere.
LONG_METHOD = {
    "python": f"def process(a, b, c, d):\n    total = 0\n{LONG_BODY['python']}\n    return total\n",
    "java": ("class Order {\n    int process(int a, int b, int c, int d) {\n"
             "        int total = 0;\n" + LONG_BODY["java"] + "\n        return total;\n    }\n}\n"),
    "cpp": ("int process(int a, int b, int c, int d) {\n    int total = 0;\n"
            + LONG_BODY["cpp"] + "\n    return total;\n}\n"),
    "c": ("int process(int a, int b, int c, int d) {\n    int total = 0;\n"
          + LONG_BODY["c"] + "\n    return total;\n}\n"),
}

BROKEN = {"python": "def f(:\n  pass\n", "java": "class A { void f( {{{ \n",
          "cpp": "void f( {{{ \n", "c": "void f( {{{ \n"}

EXT = {"python": ".py", "java": ".java", "cpp": ".cpp", "c": ".c"}


def _pick(plang, decidable_wanted):
    """A real prompt id from the real prompt set, decidable or not as asked."""
    prompts = load_prompts(plang)
    dec = decidable_for(plang)
    for pid, p in sorted(prompts.items()):
        hit = any(t in dec for t in p["code_smells"])
        if hit == decidable_wanted:
            if not decidable_wanted or "Long Method" in p["code_smells"]:
                return pid
    raise AssertionError(f"no {'decidable' if decidable_wanted else 'undecidable'} "
                         f"prompt found for {plang}")


def build_tree(root):
    """One model, four languages, in the two layouts the real run produces."""
    written = {}
    for plang in ("python", "java", "cpp", "c"):
        base = root if plang == "python" else root / plang
        d = base / "m1" / "en" / "code"
        d.mkdir(parents=True, exist_ok=True)
        good = _pick(plang, True)
        (d / (good + EXT[plang])).write_text(LONG_METHOD[plang], encoding="utf-8")
        bad = _pick(plang, False)
        (d / (bad + EXT[plang])).write_text(LONG_METHOD[plang], encoding="utf-8")
        written[plang] = {"decidable": good, "undecidable": bad}
    # One broken file in Java only, so validity must differ by language and the
    # matched subset must drop it -- it exists in no other language.
    (root / "java" / "m1" / "en" / "code" / "syntax_wreck_basic_001.java").write_text(
        BROKEN["java"], encoding="utf-8")
    return written


def main():
    tmp = Path(tempfile.mkdtemp(prefix="xlang_test_"))
    fails = []
    try:
        ids = build_tree(tmp)

        # --- discovery reaches both layouts, and the plang trees are not mistaken
        # for Python models sitting at the top level
        found = {(p, m, l) for p, m, l, _ in discover(tmp)}
        expect = {(p, "m1", "en") for p in ("python", "java", "cpp", "c")}
        if found != expect:
            fails.append(f"discover() returned {sorted(found)}, expected {sorted(expect)}")

        rows = collect(tmp)
        by = {}
        for r in rows:
            by.setdefault(r["plang"], {})[r["prompt_id"]] = r

        for plang in ("python", "java", "cpp", "c"):
            got = by.get(plang)
            if not got:
                fails.append(f"{plang}: no rows collected")
                continue

            dec = got[ids[plang]["decidable"]]
            if dec["target_covered"] != 1:
                fails.append(f"{plang}: decidable prompt not marked covered")
            if dec["target_hit"] != 1:
                fails.append(f"{plang}: Long Method not detected in a 20-line method "
                             f"(found {dec['smells_found']!r})")
            if dec["syntax_ok"] != 1:
                fails.append(f"{plang}: valid source scored as a syntax failure")

            # The point of the whole exercise: a smell the detector cannot decide
            # must be blank, not a miss. Same file, so the only difference is which
            # smell the prompt asked for.
            und = got[ids[plang]["undecidable"]]
            if und["target_covered"] != 0:
                fails.append(f"{plang}: undecidable smell marked as covered")
            if und["target_hit"] != "":
                fails.append(f"{plang}: undecidable smell scored {und['target_hit']!r}, "
                             f"expected blank -- this manufactures a failure rate")

        wreck = by.get("java", {}).get("syntax_wreck_basic_001")
        if wreck is None:
            fails.append("java: broken file not collected")
        elif wreck["syntax_ok"] != 0:
            fails.append("java: unparseable source scored as valid")
        elif wreck["n_error_nodes"] < 1:
            fails.append("java: broken file reported zero error nodes")

        # --- a base rate with nothing to compare against is blank, not zero. The
        # report used to format this as a float and died after writing the CSVs.
        only_one = [r for r in rows if r["plang"] == "python"]
        for r in only_one:
            r["target_smells"] = "Long Method"
        if base_rate(only_one, "Long Method") != "":
            fails.append("base_rate() returned a number when no off-target files exist")

        print(f"{len(rows)} rows across {len(by)} languages")
        for plang in sorted(by):
            n_ok = sum(r["syntax_ok"] for r in by[plang].values())
            print(f"  {plang:8s} {len(by[plang])} files, {n_ok} parse")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\ndiscovery reaches every arm; undecidable smells stay blank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
