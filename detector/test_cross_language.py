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
