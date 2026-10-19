#!/usr/bin/env python3
"""
Build docs/index.html, the results page, from the analysis files.

Generated rather than hand-written so the page cannot drift from the data: every
number on it is read from _analysis_fullsize/ and _analysis_xlang/ at build time.
Four earlier passages went stale because a count was typed into the prose by hand
(six smells after the detector reached twelve, nine models after thirteen, a model
described as collapsing in Chinese after the data said 96%). A fifth called one
language effect "the one that survives" after comparing intervals around each
rate, which cannot settle a difference. Prose here states a number only by
computing it, and a difference only from pipeline/effects.py.

    python -m pipeline.make_figures
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
MINUS = "−"

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


def signed(v, nd=1):
    """+5.5 or −5.5, with a true minus sign."""
    s = f"{v:+.{nd}f}"
    return s.replace("-", MINUS)


def table(headers, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead>'
            f"<tbody>{b}</tbody></table></div>")


def rate(rows, key):
    return 100.0 * sum(int(r[key]) for r in rows) / len(rows) if rows else None


def plain_list(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def mixed(*paths):
    """{(experiment, term): (odds ratio, lo, hi)} from pipeline/mixed_model.py."""
    out = {}
    for p in paths:
        if p.exists():
            for r in read(p):
                if r["odds_ratio"] not in ("", None):
                    out[(r["experiment"], r["term"])] = (float(r["odds_ratio"]),
                                                         float(r["or_lo"]), float(r["or_hi"]))
    return out


class Effects:
    """The contrasts from pipeline/effects.py, keyed as they are written."""

    def __init__(self, *paths):
        self.rows = {}
        for p in paths:
            if p.exists():
                for r in read(p):
                    self.rows[(r["experiment"], r["within"], r["contrast"], r["metric"])] = r

    def get(self, experiment, within, contrast, metric="induction"):
        r = self.rows.get((experiment, within, contrast, metric))
        if r is None:
            return None
        return {"est": num(r, "estimate"), "lo": num(r, "ci_lo"), "hi": num(r, "ci_hi"),
                "same": int(r["models_same_direction"]), "n": int(r["n_models"])}


def clear(d):
    """Does the interval exclude zero?"""
    return d is not None and (d["lo"] > 0 or d["hi"] < 0)


def cell(d, nd=1):
    """An estimate with its interval, for a table."""
    if d is None:
        return "—"
    return (f'{signed(d["est"], nd)} <span class="ci">{signed(d["lo"], nd)} to '
            f'{signed(d["hi"], nd)}</span>')


# --------------------------------------------------------------------- data

def experiment1(analysis):
    matched = {r["lang"]: r for r in read(analysis / "by_lang_matched.csv")}
    per_file = [r for r in read(analysis / "per_file.csv") if r["model"] not in EXCLUDED]
    models = sorted({r["model"] for r in per_file})

    # Pooled lint density by whether the file parses: violations over lines, not a
    # mean of per-file rates, which a handful of three-line files would dominate.
    density = {}
    for ok in ("1", "0"):
        g = [r for r in per_file if r["syntax_ok"] == ok]
        lines = sum(int(r["loc"]) for r in g)
        density[ok] = 100.0 * sum(int(r["ruff_total"]) for r in g) / lines if lines else None

    return {"matched": matched, "density": density, "models": models,
            "n_models": len(models), "n_files": len(per_file),
            "by_smell": read(analysis / "by_smell.csv"),
            "extraction": read(analysis / "extraction_check.csv")
            if (analysis / "extraction_check.csv").exists() else []}


def experiment2(xlang, root):
    per_file = [r for r in read(xlang / "per_file.csv") if r["model"] not in EXCLUDED]
    five = sorted({r["model"] for r in per_file if r["plang"] != "python"})
    rows = [r for r in per_file if r["lang"] == "en" and r["model"] in five]
    matched = {r["plang"]: r for r in read(xlang / "by_plang_matched.csv")}

    # Per model and language: validity, the output cap, and headerless includes.
    per_model = defaultdict(dict)
    for mdl in five:
        for pl in PORDER:
            g = [r for r in rows if r["model"] == mdl and r["plang"] == pl]
            if not g:
                continue
            capped = [r for r in g if r.get("hit_limit") not in ("", None)]
            per_model[mdl][pl] = {
                "valid": rate(g, "syntax_ok"),
                "limit": rate(capped, "hit_limit") if capped else None,
                "bare": 100.0 * sum(int(r.get("bare_includes") or 0) > 0 for r in g) / len(g),
                "looping": 100.0 * sum(int(r.get("bare_includes") or 0) >= 50
                                       and r.get("hit_limit") == "1" for r in g) / len(g)}

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
    others_bare = max(per_model[m][p]["bare"] for m in rest for p in ("cpp", "c")
                      if p in per_model[m])

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

    grid = read(xlang / "by_plang_lang.csv") if (xlang / "by_plang_lang.csv").exists() else []
    return {"matched": matched, "five": five, "per_model": per_model, "worst": worst,
            "runaway": runaway, "rest_range": rest_range, "others_bare": others_bare,
            "gaps": gaps, "ranked": ranked,
            "n_dot": (min(n_dot), max(n_dot)) if n_dot else (0, 0),
            "n_new": sum(1 for r in per_file if r["plang"] != "python"),
            "n_new_langs": len({r["lang"] for r in per_file if r["plang"] != "python"}),
            "n_c": len(prompts_c), "n_all": len(prompts_py),
            "n_class_smells": len(smells_py - smells_c),
            "n_hand": sum(1 for k in overrides if not k.startswith("_")),
            "grid": {(r["plang"], r["lang"]): r for r in grid},
            "translation": read(xlang / "translation_check.csv")
            if (xlang / "translation_check.csv").exists() else [],
            "extraction": read(xlang / "extraction_check.csv")
            if (xlang / "extraction_check.csv").exists() else []}


# ---------------------------------------------------------------------- page

def build(analysis: Path, xlang: Path, docs: Path):
    root = Path(__file__).resolve().parent.parent
    e1 = experiment1(analysis)
    e2 = experiment2(xlang, root)
    eff = Effects(analysis / "effects.csv", xlang / "effects.csv")
    mm = mixed(analysis / "mixed_model.csv", xlang / "mixed_model.csv")
    m1, m2 = e1["matched"], e2["matched"]
    has_grid = any(pl != "python" and l != "en" for pl, l in e2["grid"])

    # Only the figures the page shows are published; the others stay with the
    # analysis, where they are regenerated, rather than lingering as orphans.
    figs = docs / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for old in figs.glob("*.png"):
        old.unlink()
    shown = ["fig4_smell_gap_by_language.png"] + (["fig5_language_effect_by_plang.png"]
                                                  if has_grid else [])
    for name in shown:
        f = analysis / "figures" / name
        if f.exists():
            shutil.copy2(f, figs / name)
    (docs / ".nojekyll").write_text("", encoding="utf-8")

    # --------------------------------------------- experiment 1: prompt language
    d1 = {l: eff.get("prompt language", "python", f"{l} - en") for l in ORDER[1:]}
    d1s = {l: eff.get("prompt language", "python, strong detectors", f"{l} - en")
           for l in ORDER[1:]}
    strong_txt = ""
    if all(d1s.values()):
        strong_txt = ("<p>Restricted to the smells whose detectors discriminate well (a "
                      f"lift of at least {WEAK_LIFT}, below), the changes are "
                      + plain_list(f"{signed(d1s[l]['est'])} in {LANG[l]}" for l in ORDER[1:])
                      + ", so the weak detectors are not what carries the effect.</p>")
    v1 = {l: eff.get("prompt language", "python", f"{l} - en", "validity") for l in ORDER[1:]}
    all_down = all(d and d["est"] < 0 and clear(d) for d in d1.values())
    every_model = all(d and d["same"] == d["n"] for d in d1.values())
    vmax = max(abs(v["est"]) for v in v1.values() if v)
    lint_valid = [num(m1[l], "ruff_per_100loc_valid") for l in ORDER]

    rows1 = []
    for l in ORDER:
        r = m1[l]
        d = d1.get(l)
        rows1.append([LANG[l], pct(num(r, "syntax_ok_pct")), pct(num(r, "induction_valid")),
                      "—" if l == "en" else cell(d),
                      "—" if l == "en" or d is None else f'{d["same"]} of {d["n"]}'])
    lang_table = table(["Prompt language", "Valid code", "Smell produced",
                        "Change from English",
                        "Models that drop" if all(d and d["est"] < 0 for d in d1.values())
                        else "Models moving the same way"], rows1)

    # How much of the drop travels with translation damage.
    tr = {r["lang"]: r for r in e2["translation"] if r["scores"].startswith("prompt language")}
    tr_rows, tr_clear = [], []
    for l in ORDER[1:]:
        r = tr.get(l)
        if not r:
            continue
        d = {"est": num(r, "difference"), "lo": num(r, "ci_lo"), "hi": num(r, "ci_hi")}
        if clear(d):
            tr_clear.append((l, r))
        tr_rows.append([LANG[l], f'{num(r, "identifiers_retained_pct"):.0f}%',
                        signed(num(r, "drop_kept_all")), signed(num(r, "drop_lost_some")),
                        cell(d)])
    tr_table = table(["Prompt language", "Names kept by the translator",
                      "Change where every name was kept", "Change where a name was lost",
                      "Difference"], tr_rows) if tr_rows else ""
    if tr_clear:
        tr_txt = " ".join(
            f"In {LANG[l]}, prompts where the translator changed a name lose "
            f"{abs(num(r, 'drop_lost_some')):.0f} points against "
            f"{abs(num(r, 'drop_kept_all')):.0f} where every name survived, so part of "
            f"the {LANG[l]} drop travels with damage to the prompt rather than with the "
            f"language alone."
            for l, r in tr_clear)
        rest = [LANG[l] for l in tr if l not in dict(tr_clear)]
        if rest:
            tr_txt += f" In {plain_list(rest)} it makes no clear difference."
        tr_txt += (" The drop is there even on prompts that kept every name, so it is "
                   "not only translation damage.")
    else:
        tr_txt = ("Losing a name in translation makes no clear difference to the drop "
                  "in any language.")

    # -------------------------------------- experiment 2: programming language
    d2 = {p: eff.get("programming language", "en", f"{p} - python") for p in PORDER[1:]}
    rows2 = []
    for p in PORDER:
        if p not in m2:
            continue
        r = m2[p]
        rows2.append([PLANG[p], pct(num(r, "syntax_ok_pct")), pct(num(r, "induction_valid")),
                      "—" if p == "python" else cell(d2.get(p)),
                      f'{num(r, "loc_mean"):.0f} lines'])
    plang_table = table(["Programming language", "Valid code", "Smell produced",
                         "Change from Python", "Mean length"], rows2)
    x_clear = [p for p, d in d2.items() if clear(d)]
    worst = e2["worst"]
    wv = e2["per_model"][worst]
    runaway = e2["runaway"]
    runaway_txt = ""
    if runaway:
        rv = e2["per_model"][runaway[0]]
        vals = [rv[p]["valid"] for p in rv]
        runaway_txt = (f" Separately, <code>{runaway[0]}</code> never stops by itself and "
                       f"reaches the output limit on every answer in every language, but its "
                       f"answer comes first, so {min(vals):.0f} to {max(vals):.0f}% of its "
                       f"files still parse.")
    top = e2["ranked"][0]
    top_vals = e2["gaps"][top]
    hi_l, lo_l = max(top_vals, key=top_vals.get), min(top_vals, key=top_vals.get)
    habit_items = "".join(
        f"<li><strong>{SHORT.get(s, s)}</strong> "
        f"({PLANG[max(e2['gaps'][s], key=e2['gaps'][s].get)]} "
        f"{max(e2['gaps'][s].values()):.1f}%, "
        f"{PLANG[min(e2['gaps'][s], key=e2['gaps'][s].get)]} "
        f"{min(e2['gaps'][s].values()):.1f}%). {HABIT[s]}</li>"
        for s in e2["ranked"][:3] if s in HABIT)

    # ------------------------------------------ experiment 3: both at once
    d3 = {(p, l): eff.get("both", p, f"{l} - en") for p in PORDER for l in ORDER[1:]}
    inter = {(p, l): eff.get("both", f"{p} vs python", f"({l} - en) difference")
             for p in PORDER[1:] for l in ORDER[1:]}
    inter = {k: v for k, v in inter.items() if v}
    grid_rows = []
    for p in PORDER:
        if not any((p, l) in e2["grid"] for l in ORDER):
            continue
        grid_rows.append([PLANG[p]] + [
            pct(num(e2["grid"][(p, l)], "induction_valid")) if (p, l) in e2["grid"] else "—"
            for l in ORDER])
    grid_table = table(["Smell produced"] + [LANG[l] for l in ORDER], grid_rows)
    found3 = [d for d in d3.values() if d]
    n_down = sum(1 for d in found3 if d["est"] < 0)
    n_down_clear = sum(1 for d in found3 if d["est"] < 0 and clear(d))
    inter_clear = [(k, v) for k, v in inter.items() if clear(v)]
    if inter and not inter_clear:
        lo = min(v["est"] for v in inter.values())
        hi = max(v["est"] for v in inter.values())
        inter_txt = (f"In no programming language is the effect distinguishable from "
                     f"Python's: the {len(inter)} differences run from {signed(lo)} to "
                     f"{signed(hi)} points, and every interval includes zero.")
    elif inter_clear:
        inter_txt = ("The effect differs from Python's in "
                     f"{len(inter_clear)} of {len(inter)} combinations: "
                     + plain_list(f"{PLANG[p]} in {LANG[l]} ({signed(v['est'])} points)"
                                  for (p, l), v in sorted(inter_clear))
                     + "; the rest are within noise of Python.")
    else:
        inter_txt = ""
    # Validity in the grid: report any language that moves it beyond noise.
    v3 = {(p, l): eff.get("both", p, f"{l} - en", "validity") for p in PORDER for l in ORDER[1:]}
    v3_clear = [(k, v) for k, v in v3.items() if v and clear(v) and abs(v["est"]) >= 3]

    # ---------------------------------------------------------------- summary
    findings = []
    if all_down:
        findings.append(
            f"<strong>Asking in another language lowers smell production.</strong> Against "
            f"English, the requested smell appears "
            + plain_list(f"{abs(d1[l]['est']):.1f} points less often in {LANG[l]}" for l in ORDER[1:])
            + (f", in every one of the {e1['n_models']} models" if every_model else "")
            + f". Whether the code is valid barely changes (at most {vmax:.1f} points).")
    else:
        findings.append(
            "<strong>Prompt language and smell production.</strong> Against English: "
            + plain_list(f"{LANG[l]} {signed(d1[l]['est'])} points" for l in ORDER[1:] if d1[l])
            + ".")
    if tr_clear:
        l, r = tr_clear[0]
        findings.append(
            f"<strong>In {LANG[l]}, the drop is larger where translation damaged the "
            f"prompt.</strong> Where machine translation changed a name the prompt asked "
            f"for, smell production falls "
            f"{abs(num(r, 'drop_lost_some')):.0f} points; where every name survived, "
            f"{abs(num(r, 'drop_kept_all')):.0f}.")
    d2v = [d for d in d2.values() if d]
    lean = (len({d["est"] > 0 for d in d2v}) == 1
            and all(d["same"] == d["n"] for d in d2v))
    findings.append(
        f"<strong>The programming language matters smell by smell, not on "
        f"average.</strong> Overall, Java, C++ and C are within "
        f"{max(abs(d['est']) for d in d2v):.0f} points of Python"
        + ((" (" + plain_list(f"{PLANG[p]} {signed(d2[p]['est'])}" for p in PORDER[1:] if d2.get(p))
            + (", every model leaning the same way" if lean else "") + ")") if d2v else "")
        + f"; smell by smell they differ by up to {top_vals[hi_l] - top_vals[lo_l]:.0f} "
        f"points, following each language's habits.")
    if has_grid and found3:
        findings.append(
            f"<strong>The prompt-language effect is the same in every programming "
            f"language.</strong> " if inter and not inter_clear else
            f"<strong>The prompt-language effect depends on the programming "
            f"language.</strong> ")
        findings[-1] += (f"Smell production falls in {n_down} of the {len(found3)} "
                         f"combinations of prompt and programming language"
                         + (f" ({n_down_clear} beyond noise)" if n_down_clear != n_down else "")
                         + ". " + inter_txt)
    findings.append(
        f"<strong>C and C++ produce less valid code because of one model.</strong> "
        f"<code>{worst}</code> writes <code>#include</code> lines that name no header in "
        f"{wv['cpp']['bare']:.0f}% of its C++ and {wv['c']['bare']:.0f}% of its C answers; "
        f"no other model does this in more than {e2['others_bare']:.0f}%.")

    # ---------------------------------------------------------------- smells
    weak, srows = [], []
    for r in sorted(e1["by_smell"], key=lambda x: -(num(x, "lift") or -999)):
        lift = num(r, "lift") or 0
        is_weak = lift < WEAK_LIFT
        if is_weak:
            weak.append(SHORT.get(r["target_smell"], r["target_smell"]))
        srows.append([SHORT.get(r["target_smell"], r["target_smell"])
                      + (' <span class="flag">weak</span>' if is_weak else ""),
                      f'{int(num(r, "n_covered")):,}', pct(num(r, "induction_valid")),
                      pct(num(r, "base_rate")),
                      f'<span class="{"flag" if is_weak else "good"}">{signed(lift, 0)}</span>'])
    smell_table = table(["Targeted smell", "Files", "Asked for", "Not asked for", "Lift"],
                        srows)
    n_cov = len(e1["by_smell"])

    # ------------------------------------------------------------ extraction
    # Python cells from the Python detector (Experiment 1), the rest from tree-sitter.
    ex = ([r for r in e1["extraction"] if r["model"] not in EXCLUDED]
          + [r for r in e2["extraction"] if r["model"] not in EXCLUDED and r["plang"] != "python"])
    cells_ex = [r for r in ex if r["model"] != "all"]
    totals_ex = [r for r in ex if r["model"] == "all"]
    fencers = sum(1 for r in cells_ex if r["plang"] == "python" and r["lang"] == "en"
                  and num(r, "first_block_pct") >= 95)
    moved = [r for r in cells_ex if num(r, "valid_original") is not None
             and abs(num(r, "valid") - num(r, "valid_original")) >= 5]
    ind_shift = max((abs(num(r, "induction") - num(r, "induction_original")) for r in totals_ex
                     if num(r, "induction") is not None and num(r, "induction_original") is not None),
                    default=0)
    if len(moved) <= 4:
        moved_txt = "only for " + plain_list(
            f"<code>{r['model']}</code> in {LANG[r['lang']]}"
            + ("" if r["plang"] == "python" else f" {PLANG[r['plang']]}")
            + f" ({num(r, 'valid_original'):.0f} to {num(r, 'valid'):.0f}%)"
            for r in sorted(moved, key=lambda r: (r["model"], r["plang"], r["lang"])))
    else:
        moved_txt = (f"in {len(moved)} model-and-language cells, belonging to "
                     + plain_list(f"<code>{m}</code>" for m in sorted({r['model'] for r in moved}))
                     + " (<code>extraction_check.csv</code> lists them)")

    # The mixed model as a check: same direction everywhere? where more confident?
    mm_txt = ""
    if mm:
        e1_or = [(l, mm.get(("prompt language", f"C(lang, Treatment('en'))[T.{l}]")))
                 for l in ORDER[1:]]
        e2_or = [(p, mm.get(("programming language", f"C(plang, Treatment('python'))[T.{p}]")))
                 for p in PORDER[1:]]
        agree = all(o and d1[l] and (o[0] < 1) == (d1[l]["est"] < 0) for l, o in e1_or) and \
            all(o and d2[p] and (o[0] < 1) == (d2[p]["est"] < 0) for p, o in e2_or)
        above = [PLANG[p] for p, o in e2_or if o and o[1] > 1 and not clear(d2[p])]
        mm_txt = (
            "As a check, a logistic mixed-effects model with crossed random intercepts "
            "for prompt and model (<code>mixed_model.csv</code>) "
            + ("agrees on the direction of every effect" if agree
               else "disagrees on the direction of at least one effect")
            + "; its odds ratios against English are "
            + plain_list(f"{o[0]:.2f} in {LANG[l]}" for l, o in e1_or if o) + "."
            + (f" It is more confident than the resampling about the programming "
               f"language, putting {plain_list(above)} above Python." if above else ""))

    controls = table(["Control", "What it does", "Why"], [
        ["Validity", "Smell and quality figures use parsing files only",
         "A file that does not parse has no smell to find"],
        ["Matched prompts", "Every comparison uses prompts asked in all the cells it compares",
         f"C is asked {e2['n_c']} of the {e2['n_all']} prompts"],
        ["Paired differences", "Each difference is recomputed inside every resample of prompts",
         "The same prompts sit in both rates, so their errors move together"],
        ["Lift", "Each detector is also run on files that asked for another smell",
         f"A detector that fires everywhere reports a base rate; under {WEAK_LIFT} is weak"],
    ], cls="wrap-cells")
    oor = table(["Smell", "Why it is out of reach"],
                [[SHORT.get(s, s), why] for s, why in sorted(OUT_OF_REACH.items())],
                cls="wrap-cells")

    exp3 = ""
    if has_grid:
        exp3 = f"""
<section>
  <h2>Experiment 3 · both languages at once</h2>
  <p>The Java, C++ and C prompts translated into Spanish, French and Chinese the same
  way as the Python ones, and asked of the same {len(e2['five'])} models, so every
  prompt language meets every programming language.</p>
  <figure>
    <img src="figures/fig5_language_effect_by_plang.png" alt="Change in smell production against English for each prompt language, within each programming language, with 95% intervals">
    <figcaption>Change in the share of working files containing the requested smell,
    against English, on the same prompts and models. Lines are 95% intervals.</figcaption>
  </figure>
  <p>{inter_txt}</p>
  {grid_table}
  {('<p>Valid code moves beyond noise in ' + plain_list(
      f"{PLANG[p]} in {LANG[l]} ({signed(v['est'])} points)" for (p, l), v in sorted(v3_clear))
    + '.</p>') if v3_clear else
   '<p>Whether the code is valid does not move by more than 3 points in any combination.</p>'}
</section>"""

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
  <p class="meta">{e1['n_models']} models, {e1['n_files']:,} Python files in four prompt
  languages · {len(e2['five'])} models, {e2['n_new']:,} Java, C++ and C files in
  {e2['n_new_langs']} prompt language{'s' if e2['n_new_langs'] != 1 else ''} ·
  <a href="{REPO}">source</a> · <a href="{REPO}/releases">data download</a></p>
</header>

<section>
  <h2>Summary</h2>
  <ol class="findings">{''.join(f'<li>{x}</li>' for x in findings)}</ol>
</section>

<section>
  <h2>Experiment 1 · prompt language</h2>
  <p>The same {e2['n_all']} Python tasks asked in English, Spanish, French and Chinese,
  of {e1['n_models']} models. Every model answered every prompt in every language. The
  change from English is measured on the same prompts and models, with a 95% interval;
  the last column counts the models whose own rate falls.</p>
  {lang_table}
  {strong_txt}
  <p>Code quality does not depend on the prompt language: on files that parse,
  {min(lint_valid):.1f} to {max(lint_valid):.1f} lint violations per 100 lines in all
  four. Files that fail to parse average {e1['density']['0']:.0f} per 100 lines, which is
  why quality is only measured on parsing files.</p>
  <h3>How much of the drop is translation</h3>
  <p>The prompts were machine-translated, and the translator is not told which words
  are code, so it sometimes translates a name the prompt asks for
  (<code>place_order</code> becomes <code>lugar_orden</code>).</p>
  {tr_table}
  <p>{tr_txt}</p>
</section>

<section>
  <h2>Experiment 2 · programming language</h2>
  <p>The English prompts ported to Java, C++ and C, asked of {len(e2['five'])} models,
  one per family at 7B to 9B. C is asked {e2['n_c']} of the {e2['n_all']} prompts
  because {e2['n_class_smells']} smells are defined over classes; each comparison uses
  the prompts asked in both languages.</p>
  {plang_table}
  <p class="callout">The C and C++ validity gap is one model. <code>{worst}</code> writes
  valid C++ {wv['cpp']['valid']:.0f}% and C {wv['c']['valid']:.0f}% of the time, against
  {wv['python']['valid']:.0f}% in Python. Its C and C++ answers open with
  <code>#include</code> lines that name no header, which no compiler accepts, and in
  {wv['cpp']['looping']:.0f}% of C++ and {wv['c']['looping']:.0f}% of C answers it repeats
  that line until the output limit. The other {len(e2['five']) - 1} models stay between
  {e2['rest_range'][0]:.1f} and {e2['rest_range'][1]:.1f}% in every
  language.{runaway_txt}</p>
  <h3>Smells follow each language's habits</h3>
  <p>The average hides large differences smell by smell, and the direction is
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
{exp3}
<section>
  <h2>Which smells models produce on request</h2>
  <p>Experiment 1, Python, {n_cov} of 25 smells measured. Beside each rate is how often
  the same detector fires on files that asked for a different smell; the gap between
  them, the lift, is what the prompt caused. {plain_list(weak)} {'have' if len(weak) != 1 else 'has'}
  a lift under {WEAK_LIFT} and {'are' if len(weak) != 1 else 'is'} reported, not relied on.</p>
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
    method"; {e2['n_hand']} prompts were rewritten by hand. The ported prompts were then
    translated exactly as the Python ones were.</li>
    <li><strong>Extracting the code.</strong> The answer is the first fenced code block
    in the requested language. A reply with no fence is taken whole if it parses;
    otherwise the code is recovered from between the prose. The system prompt asks for
    no Markdown, yet {fencers} of the {e1['n_models']} models fence their English code
    anyway. Against the first version of the extractor, which joined every block and
    required a fence, smell production moves by at most {ind_shift:.1f} points in any
    language; validity moves by 5 points or more {moved_txt}.</li>
    <li><strong>Detectors.</strong> Experiment 1 uses the Python detector, {n_cov} of 25
    smells, built on Python's own parser. Experiments 2 and 3 use one tree-sitter
    detector for all four languages, {len(APPLICABLE)} of 25 smells, so each smell has a
    single definition; Python is re-scored with it so the columns share an
    instrument.</li>
    <li><strong>Uncertainty.</strong> Each difference is recomputed in 1,000 resamples
    of whole prompts, because one prompt's files rise and fall together. {mm_txt}</li>
    <li><strong>Excluded.</strong> <code>mamba-codestral-7b</code> substitutes a wrong
    token about once in every 140 and only 6 of 20 files parse. Five causes were tested:
    two were pipeline faults and were fixed, three were ruled out, leaving the published
    model itself. The release notes record what was tested.</li>
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
    <li><a href="{REPO}">Repository</a>: detectors, pipeline, the ported prompt sets in
    <code>dataset/</code> and every translation in <code>dataset/translations/</code>.</li>
    <li>Scores: <a href="{REPO}/tree/main/_analysis_fullsize"><code>_analysis_fullsize/</code></a>
    (Experiment 1) and <a href="{REPO}/tree/main/_analysis_xlang"><code>_analysis_xlang/</code></a>
    (Experiments 2 and 3), one row per file in <code>per_file.csv</code>, every reported
    difference in <code>effects.csv</code>.</li>
    <li><a href="{REPO}/releases">Data download</a>: every generated file and every
    model's untouched reply.</li>
    <li>Rebuild everything: <code>bash pipeline/rebuild_results.sh</code>.</li>
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
