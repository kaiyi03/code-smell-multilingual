#!/usr/bin/env python3
"""
The contrasts the paper reports, each with a 95% interval from resampling prompts.

intervals.py gives an interval for each rate on its own. A claim such as "Chinese
prompts lower smell production" is about a difference, and two intervals cannot
settle it either way: the same prompts sit in both rates, so their errors move
together. Here each difference is recomputed inside every resample, which keeps
that pairing, and the interval is read off the differences.

Three families, one per experiment:

  prompt language        each language against English: Python, 13 models,
                         the Python detector
  programming language   Java, C++ and C against Python: English, 5 models,
                         the cross-language detector
  both at once           each prompt language against English inside each
                         programming language (5 models); and the interaction,
                         how far that language effect differs from Python's

Every contrast is computed on the (model, prompt) pairs present in all the cells
it compares, so a difference can never be a difference in which prompts were asked.
Beside each estimate is how many of the models show a change in the same direction,
which is the plainest check that one model is not carrying the result.

    python -m pipeline.effects --analysis _analysis_fullsize --xlang _analysis_xlang
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RESAMPLES = 1000
SEED = 7
EXCLUDED = {"mamba-codestral-7b"}
LANGS = ["en", "es", "fr", "zh"]
PLANGS = ["python", "java", "cpp", "c"]


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Cells:
    """Per-prompt counts for each cell, so a resample is a weighted sum."""

    def __init__(self, rows, cells, cell_of):
        # Keep only (model, prompt) pairs present in every cell compared.
        have = defaultdict(set)
        for r in rows:
            c = cell_of(r)
            if c in cells:
                have[(r["model"], r["prompt_id"])].add(c)
        keep = {k for k, v in have.items() if v >= set(cells)}
        self.prompts = sorted({p for _, p in keep})
        self.models = sorted({m for m, _ in keep})
        pi = {p: i for i, p in enumerate(self.prompts)}
        ci = {c: j for j, c in enumerate(cells)}
        mi = {m: k for k, m in enumerate(self.models)}
        # counts[prompt, cell, model, (files, valid, decidable-and-valid, hits)]
        self.counts = np.zeros((len(self.prompts), len(cells), len(self.models), 4))
        for r in rows:
            c = cell_of(r)
            if c not in ci or (r["model"], r["prompt_id"]) not in keep:
                continue
            ok = r["syntax_ok"] == "1"
            dec = ok and r["target_covered"] == "1"
            hit = dec and r["target_hit"] == "1"
            self.counts[pi[r["prompt_id"]], ci[c], mi[r["model"]]] += (1, ok, dec, hit)
        self.cells = cells

    def rates(self, weights=None, model=None):
        """(validity, induction) per cell, in percent, for prompt weights w."""
        c = self.counts if model is None else self.counts[:, :, [model], :]
        w = np.ones(len(self.prompts)) if weights is None else weights
        tot = np.einsum("p,pcmk->ck", w, c)
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100 * tot[:, 1] / tot[:, 0], 100 * tot[:, 3] / tot[:, 2]


def contrast(cells, combine, label_rows):
    """Point estimate, interval and per-model agreement for combine(rates) -> value."""
    n = len(cells.prompts)
    if not n:
        return []          # nothing answered in every cell compared
    rng = np.random.default_rng(SEED)
    point = [combine(r) for r in cells.rates()]
    draws = [[], []]
    for _ in range(RESAMPLES):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        for k, r in enumerate(cells.rates(w)):
            draws[k].append(combine(r))
    out = []
    for k, metric in enumerate(("validity", "induction")):
        d = np.array(draws[k])
        d = d[~np.isnan(d)]
        per_model = [combine(cells.rates(model=i)[k]) for i in range(len(cells.models))]
        sign = np.sign(point[k])
        same = sum(1 for v in per_model if not np.isnan(v) and np.sign(v) == sign and v != 0)
        out.append(dict(label_rows, metric=metric,
                        estimate=round(float(point[k]), 2),
                        ci_lo=round(float(np.percentile(d, 2.5)), 2) if len(d) else "",
                        ci_hi=round(float(np.percentile(d, 97.5)), 2) if len(d) else "",
                        n_prompts=n, n_models=len(cells.models),
                        models_same_direction=same))
    return out


def prompt_language(rows):
    rows = [r for r in rows if r["model"] not in EXCLUDED]
    out = []
    for lang in LANGS[1:]:
        cells = Cells(rows, ["en", lang], lambda r: r["lang"])
        out += contrast(cells, lambda r: r[1] - r[0],
                        {"experiment": "prompt language", "within": "python",
                         "contrast": f"{lang} - en"})
    return out


def programming_language(rows):
    rows = [r for r in rows if r["model"] not in EXCLUDED and r["lang"] == "en"]
    five = {r["model"] for r in rows if r["plang"] != "python"}
    rows = [r for r in rows if r["model"] in five]
    out = []
    for pl in PLANGS[1:]:
        cells = Cells(rows, ["python", pl], lambda r: r["plang"])
        out += contrast(cells, lambda r: r[1] - r[0],
                        {"experiment": "programming language", "within": "en",
                         "contrast": f"{pl} - python"})
    return out


def both(rows):
    rows = [r for r in rows if r["model"] not in EXCLUDED]
    five = {r["model"] for r in rows if r["plang"] != "python" and r["lang"] != "en"}
    if not five:
        return []
    rows = [r for r in rows if r["model"] in five]
    out = []
    for pl in PLANGS:
        for lang in LANGS[1:]:
            cells = Cells(rows, [(pl, "en"), (pl, lang)], lambda r: (r["plang"], r["lang"]))
            if not cells.prompts:
                continue
            out += contrast(cells, lambda r: r[1] - r[0],
                            {"experiment": "both", "within": pl,
                             "contrast": f"{lang} - en"})
    # The interaction: does a prompt language move this programming language more
    # than it moves Python? Same prompts in all four cells.
    for pl in PLANGS[1:]:
        for lang in LANGS[1:]:
            cl = [("python", "en"), ("python", lang), (pl, "en"), (pl, lang)]
            cells = Cells(rows, cl, lambda r: (r["plang"], r["lang"]))
            if not cells.prompts:
                continue
            out += contrast(cells, lambda r: (r[3] - r[2]) - (r[1] - r[0]),
                            {"experiment": "both", "within": f"{pl} vs python",
                             "contrast": f"({lang} - en) difference"})
    return out


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def show(rows):
    for r in rows:
        print(f"  {r['experiment']:21s} {r['within']:15s} {r['contrast']:22s} {r['metric']:9s} "
              f"{r['estimate']:+7.1f}  [{r['ci_lo']:+6.1f}, {r['ci_hi']:+6.1f}]  "
              f"{r['models_same_direction']}/{r['n_models']} models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="_analysis_fullsize")
    ap.add_argument("--xlang", default="_analysis_xlang")
    args = ap.parse_args()

    e1 = prompt_language(read(Path(args.analysis) / "per_file.csv"))
    write(Path(args.analysis) / "effects.csv", e1)
    x = read(Path(args.xlang) / "per_file.csv")
    e23 = programming_language(x) + both(x)
    write(Path(args.xlang) / "effects.csv", e23)
    show(e1 + e23)


if __name__ == "__main__":
    main()
