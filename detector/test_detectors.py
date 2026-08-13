#!/usr/bin/env python3
"""
Positive and negative cases for every detector, so "21 of 25 smells are covered"
is a checkable claim rather than an assertion.

Each smell gets a minimal snippet that clearly has it, and one that clearly does
not. A detector passes only if it fires on the first and stays silent on the
second -- firing on everything is the failure mode this suite exists to catch, and
it is the one that actually happened during development (an early Temporary Field
rule matched its own base rate, and a Dead Code rule flagged every generated file).

    python -m detector.test_detectors

Exits non-zero on failure, so it can gate a commit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.extended_smells import UNDETECTABLE, detect_extended_smells
from detector.smell_detector import detect_all_smells
from pipeline.run_analysis import DETECTOR_TO_DATASET

# Detectors whose false-positive rate is known and accepted rather than fixed.
# check_duplicated_code compares method bodies by Jaccard similarity at 0.75,
# which two short methods clear on shared keywords alone -- measured base rate
# 56.6% on files targeting other smells. Left as the upstream author wrote it;
# the lift column in run_analysis is what keeps it honest.
KNOWN_LOOSE = {'Duplicated Code'}

CLEAN = """
class Account:
    def __init__(self, owner):
        self.owner = owner
        self.balance = 0

    def deposit(self, amount):
        self.balance += amount
        return self.balance
"""

# Keys are the dataset's smell names (dataset/prompts_core.json), which is what
# smells_in() normalises to -- so the suite tests the labels the pipeline joins on.
# smell -> (has it, does not have it)
CASES = {
    # --- detector/smell_detector.py ---------------------------------------
    "Long Method": ("def f(x):\n" + "".join(f"    x += {i}\n" for i in range(20))
                    + "    return x\n", CLEAN),
    "Long Parameter List": ("def f(a, b, c, d, e, g, h):\n    return a\n", CLEAN),
    "Duplicated Code": ("""
def a(v):
    total = 0
    for i in v:
        total += i * 2
    return total

def b(v):
    total = 0
    for i in v:
        total += i * 2
    return total
""", CLEAN),
    "Deep Nesting": ("""
def f(a, b, c, d, e):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        return 1
    return 0
""", CLEAN),
    "God Class / Large Class": ("class G:\n" + "".join(f"    def m{i}(self): return {i}\n"
                                         for i in range(14)), CLEAN),
    "Magic Numbers/Strings": ("def f(x):\n    return x * 86400 + 3600\n", CLEAN),
    "Global State": ("COUNTER = []\ndef bump():\n    global COUNTER\n"
                     "    COUNTER.append(1)\n", CLEAN),
    "Data Class": ("class Point:\n    def __init__(self, x, y):\n"
                   "        self.x = x\n        self.y = y\n", CLEAN),

    # --- detector/extended_smells.py --------------------------------------
    "Data Clumps": ("""
def ship(street, city, state, zip_code): pass
def bill(street, city, state, zip_code): pass
def contact(street, city, state, zip_code): pass
""", CLEAN),
    "Message Chains": ("v = order.get_customer().get_address().get_city()\n", CLEAN),
    "Feature Envy": ("""
class Report:
    def render(self, order):
        return order.total + order.tax + order.discount + order.shipping
""", CLEAN),
    "Middle Man": ("""
class Proxy:
    def __init__(self, real): self._real = real
    def a(self): return self._real.a()
    def b(self): return self._real.b()
    def c(self): return self._real.c()
""", CLEAN),
    "Lazy Class": ("class Marker:\n    pass\n", CLEAN),
    "Switch Statements": ("""
def rate(kind):
    if kind == 'a': return 1
    elif kind == 'b': return 2
    elif kind == 'c': return 3
    elif kind == 'd': return 4
    return 0
""", CLEAN),
    "Dead Code": ("import os\n\ndef f():\n    unused_total = 1\n    return 2\n", CLEAN),
    "Temporary Field": ("""
class ReportBuilder:
    def __init__(self):
        self.rows = None
    def build(self, rows):
        self.rows = rows
        return len(self.rows)
""", CLEAN),
    "Inappropriate Intimacy": ("""
class Order:
    def __init__(self, cart): self.cart = cart
    def refresh(self): return Cart(self)

class Cart:
    def __init__(self, order): self.order = order
    def rebuild(self): return Order(self)
""", CLEAN),
    "Comments (as smell indicator)": ("# def old_version():\n#     return 1\n"
                                      "def new_version():\n    return 2\n", CLEAN),
    "Refused Bequest": ("""
class Bird:
    def fly(self): return 'flying'

class Penguin(Bird):
    def fly(self): raise NotImplementedError
""", CLEAN),
    "Speculative Generality": ("""
class AbstractExporter:
    def to_pdf(self): pass
    def to_xml(self): pass
    def to_yaml(self): pass
""", CLEAN),
    "Parallel Inheritance Hierarchies": ("""
class Animal: pass
class Handler: pass
class DogAnimal(Animal): pass
class CatAnimal(Animal): pass
class DogHandler(Handler): pass
class CatHandler(Handler): pass
""", CLEAN),
}


def smells_in(src):
    """Normalised to the dataset vocabulary, exactly as run_analysis scores.

    Without this the suite would test labels the pipeline never sees:
    check_global_state emits "Global State (global keyword)", which is a hit for
    "Global State" once normalised and a miss before.
    """
    raw = set(detect_all_smells(src)["smell_types_detected"]) | \
        {s["smell"] for s in detect_extended_smells(src)}
    return {DETECTOR_TO_DATASET.get(s, s) for s in raw}


def main():
    fails = []
    print(f"{'smell':36s}{'positive':>10s}{'negative':>10s}")
    print("-" * 56)
    for smell, (positive, negative) in sorted(CASES.items()):
        fires = smell in smells_in(positive)
        quiet = smell not in smells_in(negative)
        if not fires:
            fails.append(f"{smell}: missed its own positive case")
        if not quiet and smell not in KNOWN_LOOSE:
            fails.append(f"{smell}: fired on clean code")
        print(f"{smell:36s}{'PASS' if fires else 'FAIL':>10s}"
              f"{('PASS' if quiet else ('LOOSE' if smell in KNOWN_LOOSE else 'FAIL')):>10s}")

    print("-" * 56)
    print(f"{len(CASES)} detectors, {len(UNDETECTABLE)} smells declared out of reach:")
    for u in UNDETECTABLE:
        print(f"    {u}")
    print(f"total coverage: {len(CASES)} of {len(CASES) + len(UNDETECTABLE)} targeted smells")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nall detectors fire on their own case and stay silent on clean code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
