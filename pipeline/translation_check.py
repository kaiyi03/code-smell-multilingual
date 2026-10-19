#!/usr/bin/env python3
"""
Does losing identifiers in translation explain the drop in smell production?

NLLB is not told which words are code, so it translates some of them:
place_order becomes lugar_orden, and printOrderDetails becomes
imprimirOrderDetails. The prompts name the methods and fields a smell is built
from, so a prompt that loses them may simply be a worse prompt, and part of a
"prompt language" effect would then be a translation-quality effect. It matters
most for comparing programming languages: Java's camelCase names survive
translation less often than Python's snake_case ones.

Two measurements, per prompt language and programming language:

  retention   the share of the English prompt's identifiers that appear verbatim
              in the translation
  drop        smell production in the prompt language minus English, on the same
              models and prompts, computed separately for prompts that kept every
              identifier and prompts that lost at least one; the interval on the
              difference resamples prompts within each group

If losing identifiers drives the effect, the drop is larger where they were lost.

    python -m pipeline.translation_check --analysis _analysis_fullsize --xlang _analysis_xlang
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS = PROJECT_ROOT / "dataset" / "translations"
RESAMPLES = 1000
SEED = 7
EXCLUDED = {"mamba-codestral-7b"}

# snake_case, lowerCamelCase and UpperCamelCase names: what a prompt asks for by name.
IDENT = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b"
                   r"|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"
                   r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def retention(lang, plang):
    """{prompt_id: (identifiers in English, how many survive)} for one translation."""
    name = f"prompts_{lang}.json" if plang == "python" else f"prompts_{lang}_{plang}.json"
    path = TRANSLATIONS / name
    if not path.exists():
        return {}
    out = {}
    for p in json.loads(path.read_text(encoding="utf-8")):
        ids = set(IDENT.findall(p["prompt_en"]))
        out[p["id"]] = (len(ids), sum(1 for i in ids if i in p["prompt"]))
    return out


def counts(rows, prompts):
    """Per prompt: [valid-and-decidable, hits] for English and for the other language."""
    c = defaultdict(lambda: np.zeros(4))
    for r in rows:
        if r["prompt_id"] not in prompts:
            continue
        if r["syntax_ok"] == "1" and r["target_covered"] == "1":
            off = 0 if r["lang"] == "en" else 2
            c[r["prompt_id"]][off:off + 2] += (1, r["target_hit"] == "1")
    return np.array([c[p] for p in sorted(c)]) if c else np.zeros((0, 4))


def drop(m, w=None):
    w = np.ones(len(m)) if w is None else w
    t = w @ m
    if not t[0] or not t[2]:
        return np.nan
    return 100 * (t[3] / t[2] - t[1] / t[0])


def compare(rows, lang, plang, ret):
    rows = [r for r in rows if r["lang"] in ("en", lang)]
    have = defaultdict(set)
    for r in rows:
        have[(r["model"], r["prompt_id"])].add(r["lang"])
    rows = [r for r in rows if len(have[(r["model"], r["prompt_id"])]) == 2]
    named = {p for p, (n, _) in ret.items() if n}
    kept = {p for p in named if ret[p][1] == ret[p][0]}
    lost = named - kept
    mk, ml = counts(rows, kept), counts(rows, lost)
    if not len(mk) or not len(ml):
        return None
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(RESAMPLES):
        wk = rng.multinomial(len(mk), np.full(len(mk), 1 / len(mk))).astype(float)
        wl = rng.multinomial(len(ml), np.full(len(ml), 1 / len(ml))).astype(float)
        diffs.append(drop(ml, wl) - drop(mk, wk))
    d = np.array(diffs)
    d = d[~np.isnan(d)]
    tot = sum(n for n, _ in ret.values())
    return {"plang": plang, "lang": lang,
            "identifiers_retained_pct": round(100.0 * sum(k for _, k in ret.values()) / tot, 1)
            if tot else "",
            "prompts_kept_all": len(kept), "prompts_lost_some": len(lost),
            "drop_kept_all": round(float(drop(mk)), 2),
            "drop_lost_some": round(float(drop(ml)), 2),
            "difference": round(float(drop(ml) - drop(mk)), 2),
            "ci_lo": round(float(np.percentile(d, 2.5)), 2),
            "ci_hi": round(float(np.percentile(d, 97.5)), 2)}


def run(per_file, plangs, label):
    rows = [r for r in read(per_file) if r["model"] not in EXCLUDED]
    if "plang" in rows[0]:
        five = {r["model"] for r in rows if r["plang"] != "python"}
        rows = [r for r in rows if r["model"] in five]
    out = []
    for plang in plangs:
        prow = [r for r in rows if r.get("plang", "python") == plang]
        for lang in ("es", "fr", "zh"):
            ret = retention(lang, plang)
            res = compare(prow, lang, plang, ret) if ret and prow else None
            if res:
                out.append({"scores": label, **res})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="_analysis_fullsize")
    ap.add_argument("--xlang", default="_analysis_xlang")
    args = ap.parse_args()
    out = run(Path(args.analysis) / "per_file.csv", ["python"], "prompt language (13 models)")
    out += run(Path(args.xlang) / "per_file.csv", ["python", "java", "cpp", "c"],
               "both languages (5 models)")
    path = Path(args.xlang) / "translation_check.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} rows)")
    print(f"  {'scores':28s} {'plang':6s} {'lang':4s} {'kept%':>6s} {'n kept/lost':>12s} "
          f"{'drop kept':>10s} {'drop lost':>10s} {'difference':>22s}")
    for r in out:
        print(f"  {r['scores']:28s} {r['plang']:6s} {r['lang']:4s} {r['identifiers_retained_pct']:6.1f} "
              f"{r['prompts_kept_all']:5d}/{r['prompts_lost_some']:<5d} {r['drop_kept_all']:+10.1f} "
              f"{r['drop_lost_some']:+10.1f} {r['difference']:+8.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]")


if __name__ == "__main__":
    main()
