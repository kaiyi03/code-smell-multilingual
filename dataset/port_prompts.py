#!/usr/bin/env python3
"""
Port the 426 English prompts from Python to Java, C++ and C.

Noor scoped the next phase to four programming languages. The prompts cannot
simply be translated, because a prompt is a specification and some
specifications have no equivalent in the target language: eight of the 25 smells
are defined over classes, and C has none. Asking a C model for a God Class
produces nothing measurable, and counting that as "C models resist God Classes"
would be false.

So this does three things, and reports each separately rather than quietly
producing 426 prompts per language:

  1. DROP smells the target language cannot express. C loses the eight
     class-based ones; Java and C++ lose nothing.
  2. REWRITE the surface: the language name, and for Java the identifier
     convention, since snake_case in a Java prompt would invite the model to
     write non-idiomatic code and that is not the variable under study.
  3. FLAG prompts that name a Python-specific API. "wraps str.capitalize()" has
     no mechanical Java equivalent; a human has to choose one. These are written
     out but marked needs_review so they can be fixed rather than trusted.

    python -m dataset.port_prompts                 # writes all three
    python -m dataset.port_prompts --lang java     # just one
    python -m dataset.port_prompts --report        # counts only, writes nothing
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "prompts_core.json"

LANG_NAME = {"java": "Java", "cpp": "C++", "c": "C"}

# The eight smells defined over classes. C has no classes, so these are not
# "zero occurrences in C" -- they are unaskable, and must not appear in its set.
CLASS_BASED = {
    "God Class / Large Class",
    "Data Class",
    "Lazy Class",
    "Middle Man",
    "Refused Bequest",
    "Inappropriate Intimacy",
    "Parallel Inheritance Hierarchies",
    "Temporary Field",
}

# Python APIs and idioms with no mechanical translation. A prompt naming one of
# these needs a human to pick the target-language equivalent.
PY_SPECIFIC = re.compile(
    r"\b(str\.\w+|strftime|strptime|__init__|__str__|__repr__|self\b|dict\b|list\b|"
    r"tuple\b|lambda\b|decorator|@property|dataclass|namedtuple|pandas|numpy|django|"
    r"flask|pytest|__name__|kwargs|args\b|f-string|list comprehension|generator|yield)\b",
    re.IGNORECASE)

SNAKE = re.compile(r"\b([a-z][a-z0-9]*)(_[a-z0-9]+)+\b")


def to_camel(match):
    parts = match.group(0).split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def port(text, lang):
    """Rewrite the prompt surface for the target language."""
    out = re.sub(r"\bPython\b", LANG_NAME[lang], text)

    if lang == "java":
        # Java prompts asking for snake_case would be asking for un-idiomatic
        # Java, which is a second variable this study is not trying to measure.
        out = SNAKE.sub(to_camel, out)
        out = out.replace(" module", " package")
        # Java has no free functions; "write a Java function" invites the model to
        # guess at a wrapper class, which varies the structure being measured.
        out = re.sub(r"\bfunctions\b", "methods", out)
        out = re.sub(r"\bfunction\b", "method", out)
    elif lang == "c":
        # C has structs and free functions, not classes and methods.
        out = re.sub(r"\bclasses\b", "structs", out)
        out = re.sub(r"\bclass\b", "struct", out)
        out = re.sub(r"\bmethods\b", "functions", out)
        out = re.sub(r"\bmethod\b", "function", out)
        out = re.sub(r"\bmodule\b", "translation unit", out)

    # "a Java" / "a C++" read wrong where the original said "a Python"
    out = re.sub(r"\ba (Java|C\+\+|C)\b", r"a \1", out)
    return out


def load_overrides():
    path = HERE / "prompt_overrides.json"
    if not path.exists():
        return {}, set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cleared = set(data.get("_no_change_needed", []))
    by_id = {k: v for k, v in data.items() if not k.startswith("_")}
    return by_id, cleared


def build(lang, prompts, overrides, cleared):
    kept, dropped, flagged, hand = [], [], 0, 0
    for p in prompts:
        smells = p.get("code_smells", [])
        if lang == "c" and any(s in CLASS_BASED for s in smells):
            dropped.append(p["id"])
            continue

        override = overrides.get(p["id"], {}).get(lang)
        if override:
            text, needs_review = override, False
            hand += 1
        else:
            text = port(p["prompt"], lang)
            # A flag survives only if nobody has looked at it. Listing an id under
            # _no_change_needed is a record that it was read and the automatic
            # port was right, which is different from never having been checked.
            needs_review = (bool(PY_SPECIFIC.search(p["prompt"]))
                            and p["id"] not in cleared)
        flagged += needs_review
        kept.append(dict(p,
                         prompt=text,
                         prompt_en_python=p["prompt"],
                         target_language=lang,
                         hand_written=bool(override),
                         needs_review=needs_review))
    return kept, dropped, flagged, hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", action="append", choices=list(LANG_NAME))
    ap.add_argument("--report", action="store_true", help="counts only, write nothing")
    args = ap.parse_args()
    langs = args.lang or list(LANG_NAME)

    with open(SOURCE, encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"source: {len(prompts)} Python prompts\n")

    overrides, cleared = load_overrides()
    print(f"{len(overrides)} prompts have hand-written replacements, "
          f"{len(cleared)} checked and left as ported\n")

    print(f"{'language':10s}{'ported':>9s}{'dropped':>9s}{'hand-written':>14s}"
          f"{'needs review':>14s}")
    for lang in langs:
        kept, dropped, flagged, hand = build(lang, prompts, overrides, cleared)
        print(f"{LANG_NAME[lang]:10s}{len(kept):9d}{len(dropped):9d}{hand:14d}"
              f"{flagged:14d}")
        if not args.report:
            out = HERE / f"prompts_core_{lang}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(kept, f, ensure_ascii=False, indent=2)
            print(f"           -> {out.name}")

    print("\n'dropped' is smells the language cannot express -- only C loses any,")
    print("and it loses the eight defined over classes. 'needs review' names a")
    print("Python API with no mechanical equivalent; those carry needs_review=true")
    print("so they can be corrected rather than silently trusted.")


if __name__ == "__main__":
    main()
