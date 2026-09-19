#!/usr/bin/env python3
"""
The same smell, written in four languages, must be found in all four.

That is the whole claim of detector/cross_language.py: one set of thresholds
applied through tree-sitter rather than four tools with four different ideas of
what "long method" means. This suite checks it by hand-writing each smell in
Python, Java, C++ and C and requiring the detector to find it in every language
where the smell exists at all.

    python -m detector.test_cross_language
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.cross_language import APPLICABLE, LANGUAGES, detect

BODY = {
    "python": "\n".join(f"    x = x + {i}" for i in range(20)),
    "java": "\n".join(f"        x = x + {i};" for i in range(20)),
    "cpp": "\n".join(f"        x = x + {i};" for i in range(20)),
    "c": "\n".join(f"        x = x + {i};" for i in range(20)),
}

CASES = {
    "Long Method": {
        "python": f"def f(x):\n{BODY['python']}\n    return x\n",
        "java": "class A {\n    int f(int x) {\n" + BODY["java"] + "\n        return x;\n    }\n}\n",
        "cpp": "int f(int x) {\n" + BODY["cpp"] + "\n    return x;\n}\n",
        "c": "int f(int x) {\n" + BODY["c"] + "\n    return x;\n}\n",
    },
    "Long Parameter List": {
        "python": "def f(a, b, c, d, e, g):\n    return a\n",
        "java": "class A {\n    int f(int a, int b, int c, int d, int e, int g) { return a; }\n}\n",
        "cpp": "int f(int a, int b, int c, int d, int e, int g) { return a; }\n",
        "c": "int f(int a, int b, int c, int d, int e, int g) { return a; }\n",
    },
    "Deep Nesting": {
        "python": ("def f(a,b,c,d,e):\n    if a:\n        if b:\n            if c:\n"
                   "                if d:\n                    return 1\n    return 0\n"),
        "java": ("class A { int f(int a,int b,int c,int d) {\n  if(a>0){ if(b>0){ if(c>0){"
                 " if(d>0){ return 1; } } } }\n  return 0; } }\n"),
        "cpp": ("int f(int a,int b,int c,int d){\n  if(a){ if(b){ if(c){ if(d){"
                " return 1; } } } }\n  return 0;\n}\n"),
        "c": ("int f(int a,int b,int c,int d){\n  if(a){ if(b){ if(c){ if(d){"
              " return 1; } } } }\n  return 0;\n}\n"),
    },
    "Magic Numbers/Strings": {
        "python": "def f(x):\n    return x * 86400 + 3600\n",
        "java": "class A { int f(int x) { return x * 86400 + 3600; } }\n",
        "cpp": "int f(int x) { return x * 86400 + 3600; }\n",
        "c": "int f(int x) { return x * 86400 + 3600; }\n",
    },
    "Switch Statements": {
        "python": ("def f(k):\n    if k == 1: return 1\n    elif k == 2: return 2\n"
                   "    elif k == 3: return 3\n    elif k == 4: return 4\n    return 0\n"),
        "java": ("class A { int f(int k) { switch(k) { case 1: return 1; case 2: return 2;"
                 " case 3: return 3; case 4: return 4; } return 0; } }\n"),
        "cpp": ("int f(int k){ switch(k){ case 1: return 1; case 2: return 2;"
                " case 3: return 3; case 4: return 4; } return 0; }\n"),
        "c": ("int f(int k){ switch(k){ case 1: return 1; case 2: return 2;"
              " case 3: return 3; case 4: return 4; } return 0; }\n"),
    },
    "God Class / Large Class": {
        "python": "class G:\n" + "\n".join(f"    def m{i}(self): return {i}"
                                           for i in range(14)) + "\n",
        "java": "class G {\n" + "\n".join(f"    int m{i}() {{ return {i}; }}"
                                          for i in range(14)) + "\n}\n",
        "cpp": "class G {\npublic:\n" + "\n".join(f"    int m{i}() {{ return {i}; }}"
                                                  for i in range(14)) + "\n};\n",
        "c": None,          # C has no classes; the detector must not claim otherwise
    },
}


# An if/else-if ladder is the same design problem as a switch, and it is the only
# form the smell can take in Python before 3.10. The four grammars shape it three
# different ways -- Python hangs elif clauses off one if_statement, C and C++ wrap
# the continuation in an else_clause, and Java has no wrapper node at all -- so
# "the same ladder counts the same everywhere" needs its own check. The table above
# does not provide one: its Python fixture is a ladder but the other three are real
# switch statements, so the ladder path went untested in Java, C++ and C, and a
# Java ladder scored 1 branch instead of 4 for as long as that was true.
def _ladder(n_conditions, trailing_else):
    """The same ladder in four languages: n conditions, optionally a plain else."""
    py = "def f(x):\n    if x == 0:\n        a = 0\n"
    py += "".join(f"    elif x == {i}:\n        a = {i}\n"
                  for i in range(1, n_conditions))
    py += "    else:\n        a = 99\n" if trailing_else else ""

    brace = "if (x == 0) { a = 0; }"
    brace += "".join(f" else if (x == {i}) {{ a = {i}; }}"
                     for i in range(1, n_conditions))
    brace += " else { a = 99; }" if trailing_else else ""
    return {"python": py,
            "java": "class A { void f(int x) { " + brace + " } }",
            "cpp": "void f(int x) { " + brace + " }",
            "c": "void f(int x) { " + brace + " }"}


def check_ladders():
    fails = []
    print(f"\n{'if/else ladder':26s}" + "".join(f"{l:>9s}" for l in LANGUAGES))
    print("-" * (26 + 9 * len(LANGUAGES)))

    # (label, conditions, trailing else, must it fire)
    trials = [("4 branches", 3, True, True),      # 3 conditions + else == 4
              ("4 conditions", 4, False, True),
              ("3 branches", 2, True, False),     # below threshold in every language
              ("2 conditions", 2, False, False)]
    for label, n, els, want in trials:
        srcs, cells = _ladder(n, els), []
        for lang in LANGUAGES:
            hits = [s for s in detect(srcs[lang], lang)
                    if s["smell"] == "Switch Statements"]
            got = bool(hits)
            cells.append(("PASS" if got == want else "FAIL")
                         + (f"({hits[0]['branches']})" if hits else ""))
            if got != want:
                fails.append(f"ladder/{label}/{lang}: "
                             f"{'not detected' if want else 'fired below threshold'}")
            # One ladder is one finding. Counting each rung separately would let a
            # single long chain outvote every other file in the aggregate.
            if len(hits) > 1:
                fails.append(f"ladder/{label}/{lang}: reported {len(hits)} times")
        print(f"{label:26s}" + "".join(f"{c:>9s}" for c in cells))

    # `else { if (...) }` is a nested if inside a block, not another rung. Reading
    # it as one would inflate every language that writes braces.
    for lang, src in (("java", "class A { void f(int x){ if(x==0){} else { if(x==1){} } } }"),
                      ("cpp", "void f(int x){ if(x==0){} else { if(x==1){} } }")):
        if [s for s in detect(src, lang) if s["smell"] == "Switch Statements"]:
            fails.append(f"ladder/nested-else/{lang}: block nesting read as a rung")
    print("\nthe same ladder must count the same in all four languages")
    return fails


def main():
    fails = []
    print(f"{'smell':26s}" + "".join(f"{l:>9s}" for l in LANGUAGES))
    print("-" * (26 + 9 * len(LANGUAGES)))
    for smell, per_lang in CASES.items():
        cells = []
        for lang in LANGUAGES:
            src = per_lang.get(lang)
            applicable = lang in APPLICABLE[smell]
            if src is None:
                if applicable:
                    fails.append(f"{smell}/{lang}: no fixture but declared applicable")
                cells.append("n/a")
                continue
            found = {s["smell"] for s in detect(src, lang)}
            if not applicable:
                cells.append("n/a")
                if smell in found:
                    fails.append(f"{smell}/{lang}: fired although declared inapplicable")
                continue
            ok = smell in found
            cells.append("PASS" if ok else "FAIL")
            if not ok:
                fails.append(f"{smell}/{lang}: not detected")
        print(f"{smell:26s}" + "".join(f"{c:>9s}" for c in cells))

    fails += check_ladders()

    print("-" * (26 + 9 * len(LANGUAGES)))
    n = sum(len(v) for v in APPLICABLE.values())
    print(f"{len(CASES)} smells x 4 languages = {n} applicable combinations")
    print("'n/a' is a smell that cannot exist in that language, not a miss.")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nevery smell is found in every language where it can exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
