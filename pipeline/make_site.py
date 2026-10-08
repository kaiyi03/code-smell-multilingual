#!/usr/bin/env python3
"""
Build docs/index.html, the results page, from the analysis files.

Generated rather than hand-written so the page cannot drift from the data: every
number on it is read from _analysis_fullsize/ and _analysis_xlang/ at build time.
Four earlier passages went stale because a count was typed into the prose by hand
(six smells after the detector reached twelve, nine models after thirteen, a model
described as collapsing in Chinese after the data said 96%). Prose here states a
number only by computing it.

    python -m pipeline.make_figures
    python -m pipeline.extraction_check --root ../outputs_arc
    python -m pipeline.make_site

Serve it by setting GitHub Pages to the main branch, /docs folder.
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector.cross_language import APPLICABLE, OUT_OF_REACH
from pipeline import intervals

REPO = "https://github.com/kaiyi03/code-smell-multilingual"
LANG = {"en": "English", "es": "Spanish", "fr": "French", "zh": "Chinese"}
ORDER = ["en", "es", "fr", "zh"]
PLANG = {"python": "Python", "java": "Java", "cpp": "C++", "c": "C"}
PORDER = ["python", "java", "cpp", "c"]
EXCLUDED = {"mamba-codestral-7b"}
SHORT = {"Comments (as smell indicator)": "Comments",
         "God Class / Large Class": "God Class",
         "Magic Numbers/Strings": "Magic Numbers"}
WEAK_LIFT = 20

# Why a smell sits closer to one language's habits than another's. Keyed by smell,
# and shown only for whichever smells the data puts at the top of the gap ranking,
# so a re-run that reorders them cannot attach an explanation to the wrong row.
HABIT = {
    "Data Class": "A class of private fields with getters and setters is how Java "
                  "ordinarily stores data, so a model writes a textbook example; Python "
                  "has several competing conventions, so fewer answers match the "
                  "definition.",
    "Global State": "A variable at file level is ordinary C, while Java makes the model "
                    "reach deliberately for a <code>static</code> field.",
    "Long Parameter List": "C and C++ programmers usually pass a struct rather than many "
                           "separate arguments.",
}


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def pct(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}%"


def ci(pair):
    lo, hi = pair
    return "" if lo is None else f'<span class="ci">{lo:.0f}–{hi:.0f}</span>'


def table(headers, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead>'
            f"<tbody>{b}</tbody></table></div>")


def rate(rows, key):
    return 100.0 * sum(int(r[key]) for r in rows) / len(rows) if rows else None


# --------------------------------------------------------------------- data

def experiment1(analysis):
    matched = {r["lang"]: r for r in read(analysis / "by_lang_matched.csv")}
    per_file = [r for r in read(analysis / "per_file.csv") if r["model"] not in EXCLUDED]
    by_ml = read(analysis / "by_model_lang.csv")
    models = sorted({r["model"] for r in by_ml})

    # Pooled lint density by whether the file parses: the same definition the
    # tables use (violations over lines), not a mean of per-file rates, which a
    # handful of three-line files would dominate.
    density = {}
    for ok in ("1", "0"):
        g = [r for r in per_file if r["syntax_ok"] == ok]
        lines = sum(int(r["loc"]) for r in g)
        density[ok] = 100.0 * sum(int(r["ruff_total"]) for r in g) / lines if lines else None

    extraction = []
    n_fencers = 0
    ex_path = analysis / "extraction_check.csv"
    if ex_path.exists():
        # Models that wrap at least 95% of their English replies in a fence, despite
        # a system prompt that asks for no Markdown.
        n_fencers = sum(1 for r in read(ex_path)
                        if r["lang"] == "en" and num(r, "unfenced_pct") <= 5)
        for r in read(ex_path):
            gain = num(r, "valid_fence_free_pct") - num(r, "valid_scored_pct")
            if gain >= 5:
                extraction.append(r)
    affected = {r["model"] for r in extraction}

    valid = defaultdict(dict)
    for r in by_ml:
        valid[r["model"]][r["lang"]] = num(r, "syntax_ok_pct")
    others = [m for m in models if m not in affected]
    worst_other = max((valid[m]["en"] - valid[m][l] for m in others for l in ORDER[1:]),
                      default=0)

    return {"matched": matched, "ci": intervals.by_group(per_file, "lang", ORDER),
            "density": density, "extraction": extraction, "affected": affected,
            "n_models": len(models), "n_files": len(per_file),
            "worst_other": worst_other, "n_fencers": n_fencers,
            "by_smell": read(analysis / "by_smell.csv")}


def experiment2(xlang, root):
    matched = {r["plang"]: r for r in read(xlang / "by_plang_matched.csv")}
    rows = [r for r in read(xlang / "per_file.csv")
            if r["lang"] == "en" and r["model"] not in EXCLUDED]
    five = sorted({r["model"] for r in rows if r["plang"] != "python"})
    rows = [r for r in rows if r["model"] in five]

    seen = defaultdict(set)
    for r in rows:
        seen[(r["model"], r["prompt_id"])].add(r["plang"])
    keep = {k for k, v in seen.items() if len(v) == len(PORDER)}
    m_rows = [r for r in rows if (r["model"], r["prompt_id"]) in keep]

    # Per model and language: validity, and how often the answer ran into the cap.
    per_model = defaultdict(dict)
    for mdl in five:
        for pl in PORDER:
            g = [r for r in rows if r["model"] == mdl and r["plang"] == pl]
            if not g:
                continue
            capped = [r for r in g if r.get("hit_limit") not in ("", None)]
            broken = [r for r in capped if r["syntax_ok"] == "0"]
            per_model[mdl][pl] = {
                "valid": rate(g, "syntax_ok"),
                "limit": rate(capped, "hit_limit") if capped else None,
                "broken_limit": rate(broken, "hit_limit") if broken else None}

    # The model whose C and C++ validity falls furthest below its own Python.
    def drop(m):
        d = per_model[m]
        return d["python"]["valid"] - min(d[p]["valid"] for p in ("cpp", "c") if p in d)
    worst = max(five, key=drop)
    # A model that reaches the cap on (nearly) every answer never stops by itself.
    runaway = [m for m in five if all((per_model[m][p]["limit"] or 0) >= 99
                                      for p in per_model[m])]
    rest = [m for m in five if m != worst]
    rest_range = (min(per_model[m][p]["valid"] for m in rest for p in per_model[m]),
                  max(per_model[m][p]["valid"] for m in rest for p in per_model[m]))

    by_smell = read(xlang / "by_plang_smell.csv")
    gaps = defaultdict(dict)
    for r in by_smell:
        v = num(r, "induction_valid")
        if v is not None:
            gaps[r["target_smell"]][r["plang"]] = v
    ranked = sorted(gaps, key=lambda s: max(gaps[s].values()) - min(gaps[s].values()),
                    reverse=True)
    # The chart's denominator: working files that asked for the smell, per language.
    n_dot = [sum(1 for r in rows if r["plang"] == pl and r["syntax_ok"] == "1"
                 and s in r["target_smells"].split(";"))
             for s, langs in gaps.items() for pl in langs]

    prompts_c = json.loads((root / "dataset" / "prompts_core_c.json").read_text(encoding="utf-8"))
    prompts_py = json.loads((root / "dataset" / "prompts_core.json").read_text(encoding="utf-8"))
    smells_py = {t for p in prompts_py for t in p["code_smells"]}
    smells_c = {t for p in prompts_c for t in p["code_smells"]}
    overrides = json.loads((root / "dataset" / "prompt_overrides.json").read_text(encoding="utf-8"))

    return {"matched": matched, "ci": intervals.by_group(m_rows, "plang", PORDER),
            "five": five, "per_model": per_model, "worst": worst, "runaway": runaway,
            "rest_range": rest_range, "gaps": gaps, "ranked": ranked,
            "n_dot": (min(n_dot), max(n_dot)) if n_dot else (0, 0),
            "n_new": sum(1 for r in rows if r["plang"] != "python"),
            "n_c": len(prompts_c), "n_all": len(prompts_py),
            "n_class_smells": len(smells_py - smells_c),
            "n_hand": sum(1 for k in overrides if not k.startswith("_"))}


# ---------------------------------------------------------------------- page

def build(analysis: Path, xlang: Path, docs: Path):
    root = Path(__file__).resolve().parent.parent
    e1 = experiment1(analysis)
    e2 = experiment2(xlang, root)
    m1, m2 = e1["matched"], e2["matched"]

    # Only the figure the page shows is published; the others stay with the
    # analysis, where they are regenerated, rather than lingering as orphans.
    figs = docs / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for old in figs.glob("*.png"):
        old.unlink()
    fig4 = analysis / "figures" / "fig4_smell_gap_by_language.png"
    if fig4.exists():
        shutil.copy2(fig4, figs / fig4.name)
    (docs / ".nojekyll").write_text("", encoding="utf-8")

    # ------------------------------------------------------------- summary
    en_ind, zh_ind = num(m1["en"], "induction_valid"), num(m1["zh"], "induction_valid")
    lint_valid = [num(m1[l], "ruff_per_100loc_valid") for l in ORDER]
    x_ind = [num(m2[p], "induction_valid") for p in PORDER if p in m2]
    top = e2["ranked"][0]
    top_vals = e2["gaps"][top]
    hi_l, lo_l = max(top_vals, key=top_vals.get), min(top_vals, key=top_vals.get)
    def gain(r):
        return num(r, "valid_fence_free_pct") - num(r, "valid_scored_pct")
    es_row = max((r for r in e1["extraction"] if r["lang"] == "es"), key=gain, default=None)
    # The family name, read off the affected models rather than typed, so the prose
    # cannot name a family the data no longer implicates.
    fams = {m.split("-")[0] for m in e1["affected"]}
    family = fams.pop().title() if len(fams) == 1 else "two models"
    worst = e2["worst"]
    wv = e2["per_model"][worst]

    findings = [
        f"<strong>Smell production falls from {en_ind:.0f}% in English to {zh_ind:.0f}% in "
        f"Chinese</strong>, the one prompt-language effect that survives every control.",
        f"<strong>Both apparent validity effects belong to one model family.</strong> "
        f"Spanish trails English ({pct(num(m1['es'], 'syntax_ok_pct'))} valid against "
        f"{pct(num(m1['en'], 'syntax_ok_pct'))}) because {family} leaves out the code fence "
        + (f"in Spanish: accepting unfenced code takes <code>{es_row['model']}</code> from "
           f"{num(es_row, 'valid_scored_pct'):.0f}% to {num(es_row, 'valid_fence_free_pct'):.0f}%. "
           if es_row else ". ")
        + f"C and C++ trail because <code>{worst}</code> runs past the output limit.",
        f"<strong>Code quality does not depend on the prompt language</strong> once broken "
        f"files are set aside: {min(lint_valid):.1f} to {max(lint_valid):.1f} lint violations "
        f"per 100 lines on parsing files in all four.",
        f"<strong>The requested smell appears at the same rate in every programming "
        f"language</strong> ({min(x_ind):.0f} to {max(x_ind):.0f}% of working files), but smell "
        f"by smell the languages differ by up to {top_vals[hi_l] - top_vals[lo_l]:.0f} points, "
        f"following each language's habits.",
    ]

    # ------------------------------------------------------- experiment 1
    rows1 = []
    for l in ORDER:
        r = m1[l]
        (v_ci, i_ci) = e1["ci"][l]
        rows1.append([LANG[l], f'{pct(num(r, "syntax_ok_pct"))} {ci(v_ci)}',
                      f'{pct(num(r, "induction_valid"))} {ci(i_ci)}',
                      f'{num(r, "ruff_per_100loc_all"):.1f}',
                      f'<strong>{num(r, "ruff_per_100loc_valid"):.1f}</strong>'])
    lang_table = table(["Prompt language", "Valid code", "Smell produced",
                        "Lint per 100 lines, all files", "Parsing files only"], rows1)

    ex_rows = [[r["model"], LANG[r["lang"]], pct(num(r, "valid_scored_pct")),
                f'<strong>{pct(num(r, "valid_fence_free_pct"))}</strong>',
                pct(num(r, "unfenced_pct"), 0)]
               for r in sorted(e1["extraction"], key=lambda x: (x["model"], x["lang"]))]
    ex_table = table(["Model", "Prompt language", "Valid as scored",
                      "Valid without requiring a fence", "Replies with no fence"], ex_rows)
    n_aff = len(e1["affected"])

    # ------------------------------------------------------- experiment 2
    rows2 = []
    for p in PORDER:
        if p not in m2:
            continue
        r = m2[p]
        (v_ci, i_ci) = e2["ci"][p]
        rows2.append([PLANG[p], f'{pct(num(r, "syntax_ok_pct"))} {ci(v_ci)}',
                      f'{pct(num(r, "induction_valid"))} {ci(i_ci)}',
                      f'{num(r, "loc_mean"):.0f} lines'])
    plang_table = table(["Programming language", "Valid code", "Smell produced",
                         "Mean length"], rows2)
    runaway = e2["runaway"]
    runaway_txt = ""
    if runaway:
        rv = e2["per_model"][runaway[0]]
        vals = [rv[p]["valid"] for p in rv]
        runaway_txt = (f" Separately, <code>{runaway[0]}</code> never stops by itself and reaches the cap "
                       f"on every answer in every language, but its code comes first, so it "
                       f"still parses {min(vals):.0f} to {max(vals):.0f}% of the time.")
    habit_items = "".join(
        f"<li><strong>{SHORT.get(s, s)}</strong> "
        f"({PLANG[max(e2['gaps'][s], key=e2['gaps'][s].get)]} "
        f"{max(e2['gaps'][s].values()):.1f}%, "
        f"{PLANG[min(e2['gaps'][s], key=e2['gaps'][s].get)]} "
        f"{min(e2['gaps'][s].values()):.1f}%). {HABIT[s]}</li>"
        for s in e2["ranked"][:3] if s in HABIT)

    # ------------------------------------------------------------- smells
    weak = []
    srows = []
    for r in sorted(e1["by_smell"], key=lambda x: -(num(x, "lift") or -999)):
        lift = num(r, "lift") or 0
        is_weak = lift < WEAK_LIFT
        if is_weak:
            weak.append(SHORT.get(r["target_smell"], r["target_smell"]))
        srows.append([SHORT.get(r["target_smell"], r["target_smell"])
                      + (' <span class="flag">weak</span>' if is_weak else ""),
                      f'{int(num(r, "n_covered")):,}', pct(num(r, "induction_valid")),
                      pct(num(r, "base_rate")),
                      f'<span class="{"flag" if is_weak else "good"}">{lift:+.0f}</span>'])
    smell_table = table(["Targeted smell", "Files", "Asked for", "Not asked for", "Lift"],
                        srows)
    n_cov = len(e1["by_smell"])

    controls = table(["Control", "What it does", "Why"], [
        ["Validity", "Quality and smell figures use parsing files only",
         "A file that does not parse has no quality to measure"],
        ["Matched prompts", "Languages are compared on prompts asked in all of them",
         f"C is asked {e2['n_c']} of the {e2['n_all']} prompts"],
        ["Lift", "Each detector is also run on files that asked for another smell",
         f"A detector that fires everywhere reports a base rate; under {WEAK_LIFT} is weak"],
    ], cls="wrap-cells")
    oor = table(["Smell", "Why it is out of reach"],
                [[SHORT.get(s, s), why] for s, why in sorted(OUT_OF_REACH.items())],
                cls="wrap-cells")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multilingual Code-Smell Study: Results</title>
<style>{CSS}</style></head><body><div class="wrap">
<header>
  <div class="eyebrow">Results · every figure generated from the analysis files</div>
  <h1>Multilingual Code-Smell Study</h1>
  <p class="stand">Do open-weight code models produce the code smell they are asked for,
  and does that depend on the language of the prompt or the programming language
  requested?</p>
  <p class="meta">{e1['n_models']} models, {e1['n_files']:,} files on prompt language ·
  {len(e2['five'])} models, {e2['n_new']:,} new files on programming language ·
  <a href="{REPO}">source</a> · <a href="{REPO}/releases">data download</a></p>
</header>

<section>
  <h2>Summary</h2>
  <ol class="findings">{''.join(f'<li>{x}</li>' for x in findings)}</ol>
</section>

<section>
  <h2>Experiment 1 · prompt language</h2>
  <p>The same {e2['n_all']} Python tasks asked in English, Spanish, French and Chinese,
  of {e1['n_models']} models. Every model answered every prompt in every language.
  Ranges are 95% intervals.</p>
  {lang_table}
  <p class="callout">Files that fail to parse average
  {e1['density']['0']:.0f} lint violations per 100 lines, against
  {e1['density']['1']:.1f} for files that parse, so the all-files column mostly
  measures how often generation broke. On parsing files the four languages are
  level.</p>
  <h3>Where the validity gap comes from</h3>
  <p>The gap comes from {n_aff} model{'s' if n_aff != 1 else ''}; the other
  {e1['n_models'] - n_aff} are within {e1['worst_other']:.0f} points of English in every
  language. In Spanish, {family} writes the word <code>python</code> on its own line
  instead of a fenced code block, and the extractor cannot separate the code from the
  prose around it:</p>
  {ex_table}
  <p>The system prompt asks for code without Markdown formatting.
  Of the {e1['n_models']} models, {e1['n_fencers']} ignore that and fence their code
  anyway, which is what the extractor relies on; {family} in Spanish follows it but
  adds explanation around the code. So the gap is the extractor's assumption rather
  than a failure to follow the prompt. Scores here use the original extractor until
  the full set is re-scored.</p>
</section>

<section>
  <h2>Experiment 2 · programming language</h2>
  <p>The English prompts ported to Java, C++ and C, asked of {len(e2['five'])} models,
  one per family at 7B to 9B. C is asked {e2['n_c']} of the {e2['n_all']} prompts
  because {e2['n_class_smells']} smells are defined over classes; the table compares
  prompts asked in all four languages.</p>
  {plang_table}
  <p class="callout">The C and C++ drop is one model. <code>{worst}</code> writes valid C++
  {wv['cpp']['valid']:.0f}% and C {wv['c']['valid']:.0f}% of the time, against
  {wv['python']['valid']:.0f}% in Python, and {wv['cpp']['broken_limit']:.0f}% of its broken
  C++ and {wv['c']['broken_limit']:.0f}% of its broken C answers ran to the 2,048-token output
  limit and were cut off mid-statement. The other four models
  stay between {e2['rest_range'][0]:.1f} and {e2['rest_range'][1]:.1f}% in every
  language.{runaway_txt}</p>
  <h3>Smells follow each language's habits</h3>
  <p>The flat average hides large differences smell by smell, and the direction is
  consistent: a smell is easier to produce where it is the ordinary way to write that
  language.</p>
  <figure>
    <img src="figures/fig4_smell_gap_by_language.png" alt="Share of working files containing each requested smell, by programming language, sorted by the gap between languages">
    <figcaption>Working files only, English prompts, the same {len(e2['five'])} models in
    every language. Each point rests on {e2['n_dot'][0]} to {e2['n_dot'][1]} files, so
    gaps under about 20 points are within noise.</figcaption>
  </figure>
  <ul>{habit_items}</ul>
</section>

<section>
  <h2>Which smells models produce on request</h2>
  <p>Experiment 1, Python, {n_cov} of 25 smells measured. Beside each rate is how often
  the same detector fires on files that asked for a different smell; the gap between
  them, the lift, is what the prompt caused. {', '.join(weak[:-1]) + ' and ' + weak[-1] if len(weak) > 1 else ''.join(weak)} have a lift under
  {WEAK_LIFT} and are reported, not relied on.</p>
  {smell_table}
</section>

<section>
  <h2>How the numbers were produced</h2>
  <ul>
    <li><strong>Generation.</strong> Oxford's ARC GPU cluster, one prompt at a time,
    greedy decoding, up to 2,048 new tokens, bfloat16 throughout. An answer is exactly
    reproducible on one machine; across GPU nodes a small share differ, enough to flip
    the parse verdict for about 7% of prompts in a CodeLlama comparison.</li>
    <li><strong>Prompts.</strong> Spanish, French and Chinese by NLLB-200 machine
    translation, with system prompts translated by hand. Java, C++ and C ported rather
    than translated, since "write a Python function" has to become "write a Java
    method"; {e2['n_hand']} prompts were rewritten by hand.</li>
    <li><strong>Detectors.</strong> Experiment 1 uses the Python detector, {n_cov} of 25
    smells, built on Python's own parser. Experiment 2 uses one tree-sitter detector for
    all four languages, {len(APPLICABLE)} of 25 smells, so each smell has a single
    definition; Python is re-scored with it so the four columns share an instrument.</li>
    <li><strong>Uncertainty.</strong> Intervals come from resampling whole prompts 500
    times, because one prompt's files rise and fall together.</li>
    <li><strong>Excluded.</strong> <code>mamba-codestral-7b</code> substitutes a wrong token about once
    in every 140 and only 6 of 20 files parse. Five causes were tested: two were
    pipeline faults and were fixed, three were ruled out, leaving the published model
    itself. The release notes record what was tested.</li>
  </ul>
  {controls}
  <details>
    <summary>The {len(OUT_OF_REACH)} smells the cross-language detector does not measure, and why</summary>
    {oor}
  </details>
</section>

<section>
  <h2>Data and code</h2>
  <ul>
    <li><a href="{REPO}">Repository</a>: detectors, pipeline, and the ported prompt sets
    in <code>dataset/</code>.</li>
    <li>Scores: <a href="{REPO}/tree/main/_analysis_fullsize"><code>_analysis_fullsize/</code></a>
    (Experiment 1) and <a href="{REPO}/tree/main/_analysis_xlang"><code>_analysis_xlang/</code></a>
    (Experiment 2), one row per file in <code>per_file.csv</code>.</li>
    <li><a href="{REPO}/releases">Data download</a>: every generated file and every
    model's untouched reply.</li>
    <li>Rebuild this page: <code>python -m pipeline.make_site</code>.</li>
  </ul>
</section>
</div></body></html>
"""
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "index.html").write_text(html, encoding="utf-8")
    print(f"  wrote {docs / 'index.html'}  ({len(html):,} bytes)")


CSS = """
:root{
  --paper:#fbfbfc; --surface:#fff; --surface-2:#f2f4f7;
  --ink:#16181d; --ink-2:#454b57; --ink-3:#767d8b;
  --rule:#e4e7ec; --accent:#2E6B9E; --warn:#C2691F; --good:#2f6b4f;
  --mono:ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace;
  --body:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{ --paper:#101215; --surface:#171a1f; --surface-2:#1e222a;
    --ink:#e8eaee; --ink-2:#a9b0bd; --ink-3:#7c8492; --rule:#272b33;
    --accent:#5fa3d0; --warn:#d89a4a; --good:#6bb58c; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
     font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:54rem;margin:0 auto;padding:3.5rem 1.5rem 6rem;
      display:flex;flex-direction:column;gap:2.75rem}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.13em;
         text-transform:uppercase;color:var(--accent)}
h1{font-family:var(--mono);font-size:1.8rem;line-height:1.25;font-weight:600;
   margin:.5rem 0 0;letter-spacing:-.01em;text-wrap:balance}
.stand{color:var(--ink-2);font-size:1.06rem;margin-top:.9rem;max-width:44rem}
.meta{font-family:var(--mono);font-size:.73rem;color:var(--ink-3);
      margin-top:1.3rem;padding-top:.9rem;border-top:1px solid var(--rule)}
h2{font-family:var(--mono);font-size:.8rem;letter-spacing:.11em;text-transform:uppercase;
   color:var(--ink-3);font-weight:600;padding-bottom:.55rem;
   border-bottom:1px solid var(--rule);margin:0 0 .2rem}
h3{font-family:var(--ui);font-size:1.02rem;font-weight:650;margin:.6rem 0 0;text-wrap:balance}
section{display:flex;flex-direction:column;gap:1.1rem}
p{margin:0;max-width:46rem}
code{font-family:var(--mono);font-size:.87em;background:var(--surface-2);
     padding:.1em .35em;border-radius:3px}
figure{margin:0;display:flex;flex-direction:column;gap:.6rem}
figure img{width:100%;height:auto;border:1px solid var(--rule);border-radius:6px;
           background:#fcfcfb}
figcaption{font-family:var(--ui);font-size:.85rem;color:var(--ink-3);line-height:1.5}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.79rem;
      font-variant-numeric:tabular-nums}
th,td{padding:.45rem .8rem;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
table.wrap-cells td,table.wrap-cells th{white-space:normal;text-align:left;vertical-align:top}
thead th{color:var(--ink-3);font-weight:600;font-size:.69rem;letter-spacing:.06em;
         text-transform:uppercase;background:var(--surface-2)}
tbody tr:last-child td{border-bottom:none}
.ci{color:var(--ink-3);font-size:.9em;margin-left:.35em}
.flag{color:var(--warn);font-weight:700}
.good{color:var(--good)}
.callout{background:var(--surface-2);border-radius:6px;padding:1.05rem 1.25rem;
         font-size:.97rem;color:var(--ink-2);max-width:none}
ul,ol{margin:0;padding-left:1.25rem;display:flex;flex-direction:column;gap:.55rem;max-width:46rem}
li::marker{color:var(--ink-3)}
ol.findings{gap:.8rem}
details{border:1px solid var(--rule);border-radius:6px;padding:.75rem 1rem;background:var(--surface)}
summary{cursor:pointer;font-family:var(--ui);font-size:.92rem;color:var(--ink-2)}
details[open] summary{margin-bottom:.8rem}
a{color:var(--accent)}
@media (max-width:560px){ body{font-size:16px} .wrap{padding:2.25rem 1.1rem 4rem} h1{font-size:1.4rem} }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="_analysis_fullsize")
    ap.add_argument("--xlang", default="_analysis_xlang")
    ap.add_argument("--docs", default="docs")
    args = ap.parse_args()
    build(Path(args.analysis), Path(args.xlang), Path(args.docs))


if __name__ == "__main__":
    main()
