#!/usr/bin/env python3
"""
The extractor must take the answer, only the answer, and nothing a model wrote
around it. Each case below is a shape of reply found in the real outputs.

    python -m pipeline.test_extract
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.extract import extract

F = "`" * 3
CLASS = "class Order:\n    def total(self):\n        return 1\n"
JAVA = "public class Order {\n    int total() {\n        return 1;\n    }\n}\n"

CASES = [
    # (name, plang, reply, expected code, expected step)
    ("one block", "python",
     f"Here it is:\n{F}python\n{CLASS}{F}\nDone.", CLASS, "first_block"),
    ("usage line after the answer is not the answer", "python",
     f"{F}python\n{CLASS}{F}\nUse it like this:\n{F}python\no = Order()\nprint(o.total())\n{F}",
     CLASS, "first_block"),
    # StarCoder2-7B: the same program again, which joined would read as Duplicated Code
    ("a repeated copy is not part of the answer", "python",
     f"{F}python\n{CLASS}{F}\nAgain:\n{F}python\n{CLASS}{F}\n", CLASS, "first_block"),
    ("a shell block before the code is skipped", "python",
     f"Install first:\n{F}bash\npip install requests\n{F}\nThen:\n{F}python\n{CLASS}{F}",
     CLASS, "first_block"),
    ("cut off by the output limit: runs to the end", "python",
     f"{F}python\nclass Order:\n    def total(self):\n        return (1 +",
     "class Order:\n    def total(self):\n        return (1 +\n", "first_block"),
    # CodeLlama: bare code, as the system prompt asks
    ("no fence, nothing but code", "python", CLASS, CLASS, "whole"),
    # Granite in Spanish: no fence, the language on its own line, prose around
    ("no fence, prose around the code", "python",
     f"Aquí está el código:\n\npython\n{CLASS}\nEste código define una clase.",
     CLASS, "fence_free"),
    ("no fence and no code", "python",
     "I cannot help with that request.", "I cannot help with that request.\n", "unparsed"),
    ("empty reply", "python", "", "", "unparsed"),
    ("Java: first java block, not the longest", "java",
     f"{F}java\n{JAVA}{F}\nOutput:\n{F}\n" + "line\n" * 30 + F, JAVA, "first_block"),
    ("Java: no fence, prose around the code", "java",
     f"Voici le code :\n\n{JAVA}\nCe code définit une classe.", JAVA, "fence_free"),
    ("C++ code tagged c is still C++", "cpp",
     f"{F}c\nint f() {{ return 1; }}\n{F}", "int f() { return 1; }\n", "first_block"),
]


def main():
    failed = 0
    for name, plang, reply, want_code, want_step in CASES:
        code, step = extract(reply, plang)
        ok = code == want_code and step == want_step
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            print(f"       got step={step!r} code={code!r}\n      want step={want_step!r} code={want_code!r}")
    print(f"{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
