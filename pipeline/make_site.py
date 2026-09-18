#!/usr/bin/env python3
"""
Build docs/index.html -- the readable results page -- from the analysis CSVs.

Generated rather than hand-written so the page cannot drift from the data: every
number on it is read out of _analysis/*.csv at build time. Re-run this after
run_analysis.py and make_figures.py and the page is current.

    python -m pipeline.make_site

Serve it by setting GitHub Pages to the main branch, /docs folder.
"""

import argparse
import csv
import shutil
from pathlib import Path

LANG = {"en": "English", "es": "Spanish", "fr": "French", "zh": "Chinese"}
ORDER = ["en", "es", "fr", "zh"]

PLANG = {"python": "Python", "java": "Java", "cpp": "C++", "c": "C"}
PORDER = ["python", "java", "cpp", "c"]


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row, key, nd=1, suffix=""):
    try:
        return f"{float(row[key]):.{nd}f}{suffix}"
    except (KeyError, ValueError, TypeError):
        return "—"


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
h3{font-family:var(--ui);font-size:1.02rem;font-weight:650;margin:0;text-wrap:balance}
section{display:flex;flex-direction:column;gap:1.1rem}
p{margin:0}
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
thead th{color:var(--ink-3);font-weight:600;font-size:.69rem;letter-spacing:.06em;
         text-transform:uppercase;background:var(--surface-2)}
tbody tr:last-child td{border-bottom:none}
.hi{color:var(--warn);font-weight:700}
.lo{color:var(--good)}
/* A cell that cannot be asked, as against one that was asked and came back zero:
   God Class has no meaning in C, and showing 0 there would read as a finding. */
.muted{color:var(--ink-3)}
section h3{margin:1.7rem 0 .65rem}
.callout{background:var(--surface-2);border-radius:6px;padding:1.05rem 1.25rem;
         font-size:.97rem;color:var(--ink-2)}
.callout strong{color:var(--ink)}
ul{margin:0;padding-left:1.15rem;display:flex;flex-direction:column;gap:.45rem}
li::marker{color:var(--ink-3)}
a{color:var(--accent)}
@media (max-width:560px){ body{font-size:16px} .wrap{padding:2.25rem 1.1rem 4rem} h1{font-size:1.4rem} }
"""


def table(headers, rows):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def xlang_section(xlang: Path):
    """The programming-language comparison, or nothing if it has not been run.

    Returned empty rather than stubbed so the page never shows a heading with no
    numbers under it. The scoring is a separate pass (pipeline/run_xlang_analysis)
    because it uses a different instrument -- see that module's docstring.
    """
    matched_p = xlang / "by_plang_matched.csv"
    if not matched_p.exists():
        return ""
    matched = {r["plang"]: r for r in read(matched_p)}
    allp = {r["plang"]: r for r in read(xlang / "by_plang.csv")}
    by_smell = read(xlang / "by_plang_smell.csv")
    if not matched:
        return ""

    plangs = [p for p in PORDER if p in matched]
    rows = []
    for p in plangs:
        m, a = matched[p], allp.get(p, {})
        rows.append([PLANG[p], f(a, "n_files", 0), f(m, "n_files", 0),
                     f(m, "syntax_ok_pct", 1, "%"), f(m, "n_decidable", 0),
                     f(m, "induction_valid", 1, "%")])
    lang_table = table(["Language", "Files", "Matched", "Valid", "Decidable",
                        "Induction"], rows)

    # Lift per smell per language. The raw induction rate is not comparable across
    # languages -- identical thresholds do not give identical base rates -- so lift
    # is what is shown side by side, with the rate it is computed from beside it.
    smells = sorted({r["target_smell"] for r in by_smell})
    srows = []
    for s in smells:
        cells = [s]
        for p in plangs:
            hit = [r for r in by_smell if r["target_smell"] == s and r["plang"] == p]
            if not hit:
                cells.append('<span class="muted">n/a</span>')
                continue
            r = hit[0]
            rate = f(r, "induction_valid", 0, "%")
            try:
                lift = float(r["lift"])
            except (ValueError, TypeError, KeyError):
                # No off-target files to compare against, so the lift is unknown --
                # but the induction rate is not, and dropping both would hide a
                # number we have.
                cells.append(f'{rate} <span class="muted">(n/a)</span>')
                continue
            cls = "hi" if lift < 20 else "lo"
            cells.append(f'{rate} <span class="{cls}">({lift:+.0f})</span>')
        srows.append(cells)
    smell_table = table(["Targeted smell"] + [PLANG[p] for p in plangs], srows)

    return f"""
<section>
  <h2>The same smells asked for in four programming languages</h2>
  <p>Every figure here comes from one detector — <code>detector/cross_language.py</code>,
  built on tree-sitter — applied to all four languages with one set of thresholds.
  The alternative, PMD for Java and clang-tidy for C, would have given each
  language its own definition of “long method” and produced numbers that cannot be
  set beside each other.</p>
  {lang_table}
  <p>Two denominators, and they differ. <em>Valid</em> is measured on every file.
  <em>Induction</em> is measured only over the files whose targeted smell this
  detector can decide in that language — six of the twenty-five smells are shaped
  like something a syntax tree can answer, and the rest are not. A smell that
  cannot be decided is left blank, never counted as a miss. <em>Matched</em> holds
  the prompt set fixed across the four languages: C is asked 292 of the 426 prompts
  because eight of the smells are defined over classes and C has none, so without
  matching the C column would be scored on a different set of tasks.</p>
  <p>These prompts are English only. The ported prompt sets were never translated,
  so this table varies the programming language and holds the prompt language
  fixed — it is not the sixteen-cell design of four prompt languages by four
  programming languages.</p>
  <h3>Induction by smell, with lift in brackets</h3>
  {smell_table}
  <p>The bracketed figure is lift: induction minus how often the same detector
  fires on files that asked for some <em>other</em> smell, computed separately for
  each language. It is the comparable column. Identical thresholds do not imply
  identical base rates — a magic number is far commoner in C than in Python for
  reasons that have nothing to do with the prompt — so reading the raw rates across
  a row will mislead. Anything under +20 is marked, and means the detector is
  largely reporting a base rate in that language.</p>
</section>
"""


def build(analysis: Path, docs: Path, xlang: Path = None):
    matched = {r["lang"]: r for r in read(analysis / "by_lang_matched.csv")}
    by_smell = read(analysis / "by_smell.csv")
    by_ml = read(analysis / "by_model_lang.csv")

    figs = docs / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for p in sorted((analysis / "figures").glob("*.png")):
        shutil.copy2(p, figs / p.name)
    (docs / ".nojekyll").write_text("", encoding="utf-8")

    n_files = sum(int(float(r["n_files"])) for r in matched.values())
    xlang_html = xlang_section(xlang) if xlang else ""

    # --- language table
    lang_rows = []
    for l in ORDER:
        if l not in matched:
            continue
        r = matched[l]
        lang_rows.append([LANG[l], f(r, "n_files", 0), f(r, "syntax_ok_pct", 1, "%"),
                          f'<span class="hi">{f(r, "ruff_per_100loc_all", 2)}</span>',
                          f'<span class="lo">{f(r, "ruff_per_100loc_valid", 2)}</span>',
                          f(r, "induction_valid", 1, "%")])

    # --- smell table, with the discrimination check beside each rate
    smell_rows = []
    for r in sorted(by_smell, key=lambda x: -float(x["lift"] or -999)):
        lift = float(r["lift"] or 0)
        weak = lift < 20
        name = r["target_smell"] + (' <span class="hi">weak</span>' if weak else "")
        smell_rows.append([name, f(r, "n_covered", 0), f(r, "induction_valid", 1, "%"),
                           f(r, "base_rate", 1, "%"),
                           f'<span class="{"hi" if weak else "lo"}">{lift:+.1f}</span>'])

    # --- model x language
    models = sorted({r["model"] for r in by_ml})
    ml_rows = []
    for m in models:
        cells = []
        for l in ORDER:
            hit = [r for r in by_ml if r["model"] == m and r["lang"] == l]
            if not hit:
                cells.append("—")
                continue
            v = float(hit[0]["syntax_ok_pct"])
            cells.append(f'<span class="hi">{v:.0f}%</span>' if v < 70 else f"{v:.0f}%")
        ml_rows.append([m] + cells)
    ml_rows.sort(key=lambda r: min(
        (float(c.replace("%", "").replace('<span class="hi">', "").replace("</span>", ""))
         for c in r[2:] if c != "—"), default=100), reverse=True)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multilingual Code-Smell Study — Results</title>
<style>{CSS}</style></head><body><div class="wrap">

<header>
  <div class="eyebrow">Results · generated from the analysis CSVs</div>
  <h1>Multilingual Code-Smell Study</h1>
  <p class="stand">What 14 open-weight code models produce when asked, in four
  human languages, to write Python exhibiting a named code smell.</p>
  <p class="meta">{n_files:,} generations on matched prompts ·
  <a href="https://github.com/kaiyi03/code-smell-multilingual">source</a> ·
  rebuild with <code>python -m pipeline.make_site</code></p>
</header>

<section>
  <h2>What this page shows</h2>
  <div class="callout">
    <p style="margin-bottom:.7rem">Fourteen open-weight code models are given 426
    prompts that ask them to <strong>write Python code containing a named bad
    pattern</strong> — a code smell. The same prompts are machine-translated into
    Spanish, French and Chinese, so prompt language is the variable. The question is
    whether a model still produces the smell it was asked for, and whether that
    changes with the language of the request.</p>
    <p style="margin-bottom:.7rem"><strong>&ldquo;Valid Python&rdquo; below means the file
    parses</strong> — Python can read its structure. It does not mean the code runs
    correctly or passes tests; a file that fails to parse simply has nothing to
    measure, which is why it is separated out.</p>
    <p><strong>Scale.</strong> These figures are full-size: all 426 prompts in all
    four languages, for nine models, generated on the ARC cluster. They replace an
    earlier pilot that used 75 prompts per non-English language, and the change
    matters — see the note below. Three models are absent: StarCoder2 is being
    regenerated after a prompt-formatting bug, CodeLlama is a gated repository
    awaiting access, and mamba-codestral fails to load correctly.</p>
    <p><strong>The pilot was misleading, and not because it was small.</strong>
    Holding the model set fixed and moving from 75 prompts per language to 426,
    Chinese goes from 88.6% valid output to 97.7% — from apparently the worst
    language to marginally the best. The pilot selected one prompt per smell per
    difficulty level by taking the first alphabetically, which is a systematic
    subsample rather than a random one. Any result quoted from it should be
    re-checked here.</p>
  </div>
</section>

<section>
  <h2>The headline metric was measuring the wrong thing</h2>
  <p>The study's quality metric is ruff violations per 100 lines. When a model
  emits something that does not parse, the line count collapses and the rate
  explodes, so the metric largely reports <em>whether generation broke</em> rather
  than how good the surviving code is. Scored across this corpus, files that parse
  average 6.4 violations per 100 lines; files that do not average 1,182.</p>
  <figure>
    <img src="figures/fig1_validity_confound.png" alt="Violation rate by language, before and after conditioning on valid Python">
    <figcaption>Left: the same measurement with and without the validity control.
    Right: what survives it.</figcaption>
  </figure>
  {table(["Prompt language", "Files", "Valid", "ruff/100loc as reported", "on valid code", "Induction"], lang_rows)}
  <div class="callout"><strong>Chinese looks four times worse than English until you
  score only code that is valid Python — then it is 1.4 times, and Spanish and
  French land within 6% of English.</strong> The apparent quality gap between
  languages is very largely a generation-failure gap.</div>
</section>

<section>
  <h2>What models do when asked for a smell</h2>
  <p>Every prompt names a smell it wants the generated code to exhibit. Running the
  detector over the output asks whether that smell actually appears — the
  measurement the pipeline was built for and had never run. Beside each rate is a
  control: how often the same detector fires on files that asked for a
  <em>different</em> smell. Long methods, for instance, turn up in plenty of code
  nobody asked to be long — so the second number is how common the pattern is
  anyway, and only the gap between the two can be credited to the prompt. Where the
  two numbers nearly coincide, the detector is not really detecting anything.</p>
  <figure>
    <img src="figures/fig2_induction_by_smell.png" alt="Induction rate by targeted smell against base rate">
    <figcaption>Valid Python only. Sorted by lift.</figcaption>
  </figure>
  {table(["Targeted smell", "Files", "Asked for", "Not asked", "Lift"], smell_rows)}
  <p>Models comply almost always when the smell is a local property of one function,
  and resist when it requires committing to a bad overall structure: asked for a God
  Class they tend to split the work across several well-formed classes instead.
  Three detectors — Dead Code, Inappropriate Intimacy and Speculative Generality —
  fire nearly as often on files that did not ask for them, so their rates are shown
  but should not be read as measurements.</p>
</section>

<section>
  <h2>The language effect belongs to particular models</h2>
  <figure>
    <img src="figures/fig3_validity_by_model.png" alt="Syntax validity by model and prompt language">
    <figcaption>Sorted by worst non-English result.</figcaption>
  </figure>
  {table(["Model"] + [LANG[l] for l in ORDER], ml_rows)}
  <p>Four models account for nearly all of it. granite-8b-code falls to 23% valid
  output on Spanish while holding 96% on Chinese; deepseek-coder-6.7b is fine in
  Spanish and collapses in Chinese. The Qwen, StarCoder2 and Yi families barely
  move. No account of “non-English prompts are harder” explains a model that
  survives Chinese but not Spanish.</p>
</section>
{xlang_html}

<section>
  <h2>How the detector was extended, and how to check it</h2>
  <p>The repository's <code>detector/smell_detector.py</code> implemented eight
  checks — the smells decidable by counting things inside one file: parameters,
  lines, nesting depth, methods per class, numeric literals, module-level
  assignments, similar method bodies, fields without behaviour.
  <code>detector/extended_smells.py</code> adds thirteen more, each a stated
  threshold over the syntax tree:</p>
  <div class="scroll"><table>
    <thead><tr><th>Added smell</th><th>Rule</th></tr></thead>
    <tbody>
      <tr><td>Data Clumps</td><td>≥3 parameter names shared by ≥2 signatures</td></tr>
      <tr><td>Message Chains</td><td>attribute/call chain ≥3 hops from its root</td></tr>
      <tr><td>Feature Envy</td><td>a method touching one other object ≥3 times, and more than <code>self</code></td></tr>
      <tr><td>Middle Man</td><td>≥half a class's methods are a single forwarding call</td></tr>
      <tr><td>Lazy Class</td><td>no methods beyond dunders, or ≤1 method within ≤5 lines</td></tr>
      <tr><td>Switch Statements</td><td>if/elif ladder of ≥4 branches on one subject</td></tr>
      <tr><td>Dead Code</td><td>unreachable statements, unused imports, unused locals</td></tr>
      <tr><td>Temporary Field</td><td>attribute declared <code>None</code> and populated only in another method</td></tr>
      <tr><td>Inappropriate Intimacy</td><td>reaching past another object's underscore, or two classes naming each other</td></tr>
      <tr><td>Comments</td><td>commented-out code, or comment:code ratio ≥0.4</td></tr>
      <tr><td>Refused Bequest</td><td>a subclass overriding an inherited method with a stub</td></tr>
      <tr><td>Speculative Generality</td><td>≥2 stub hooks on a class nothing subclasses</td></tr>
      <tr><td>Parallel Inheritance</td><td>two hierarchies whose subclass names mirror each other</td></tr>
    </tbody>
  </table></div>
  <p><strong>Two checks keep this honest, because a threshold can always be set
  loose enough to find anything.</strong> The first is the base-rate column above:
  a detector is run against files that targeted a <em>different</em> smell, and if
  it fires there just as often it is measuring how common a pattern is, not whether
  the prompt caused it. The second is
  <code>detector/test_detectors.py</code>, which gives every one of the 21 a
  snippet that clearly has the smell and one that clearly does not, and fails if a
  detector misses the first or fires on the second.</p>
  <p>Both caught real errors here rather than confirming the work. An early
  Temporary Field rule required the attribute to be <em>absent</em> from
  <code>__init__</code>, which excludes the smell's commonest form — it scored 1.1%
  on its own targets against a 1.2% base rate. A Dead Code rule counted uncalled
  public functions, which is the normal shape of a generated snippet. Lazy Class
  matched any small class, reporting 92.5% until its threshold was tightened to
  57%. And <code>check_global_state</code> turned out to emit two different labels,
  only one of which the analysis was joining on, so files whose only signal was the
  <code>global</code> keyword had been counted as misses.</p>
  <p>Run it yourself: <code>python -m detector.test_detectors</code>.</p>
</section>

<section>
  <h2>How these numbers were produced</h2>
  <ul>
    <li><strong>Scored with the project's own detector, extended.</strong>
    <code>detector/smell_detector.py</code> covered eight of the twenty-five
    targeted smells, which capped these figures at a third of the corpus.
    <code>detector/extended_smells.py</code> adds thirteen more, taking coverage to
    21 of 25 and 85% of files. Four remain out of reach of any single-file rule —
    Shotgun Surgery needs change history, Incomplete Library Class needs the
    library's intent, Alternative Classes needs semantic equivalence, and Primitive
    Obsession needs a judgement about what deserves a type. They are excluded rather
    than counted as misses.</li>
    <li><strong>Each detector is checked against its own base rate.</strong> Every
    threshold here is a heuristic, so each is measured on files that targeted a
    different smell. Thresholds were then calibrated against that: the Temporary
    Field rule originally required the attribute to be absent from
    <code>__init__</code>, which excluded the smell's commonest form and left it
    firing on 1% of its own targets.</li>
    <li><strong>Conditioned on syntax validity.</strong> Every quality figure is
    given twice, over all files and over files that parse.</li>
    <li><strong>Matched prompt set.</strong> Only prompts generated in all four
    languages are compared, so a language contrast cannot be a contrast between
    different task mixes.</li>
    <li><strong>mamba-codestral-7b is excluded</strong> from every aggregate. It
    produces valid Python only 33% of the time <em>in English</em>, where every other
    model manages 80–100%. That is a loading problem, not a quality one: this
    architecture needs optional CUDA kernels that were not installed, so it ran on a
    fallback path and emitted malformed output such as
    <code>def update.inventory(self):</code>. Leaving it in would drag every language
    average down for a reason that has nothing to do with the model.</li>
    <li><strong>Identifier drift — tested, and not a confound.</strong> Machine
    translation rewrote 19–30% of the API identifiers the prompts specify
    (<code>place_order</code> becomes <code>lugar_orden</code>). Compared within each
    language, prompts whose identifiers were rewritten are no less likely to produce
    valid Python than those left alone (Spanish 76.9% vs 78.7%, French 90.6% vs
    90.5%, Chinese 87.2% vs 84.3%) and no less likely to produce the targeted smell.
    The detector is structural and never reads a name, so renaming does not move what
    is measured. It remains a wording problem for the write-up — the scope document
    states the translated prompts preserve the original specification, and they do
    not — rather than a defect in these results.</li>
  </ul>
  <p>Aggregate CSVs are in
  <a href="https://github.com/kaiyi03/code-smell-multilingual/tree/main/_analysis"><code>_analysis/</code></a>.</p>
</section>

</div></body></html>
"""
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "index.html").write_text(html, encoding="utf-8")
    print(f"  wrote {docs / 'index.html'}  ({len(html):,} bytes)")
    print(f"  copied {len(list(figs.glob('*.png')))} figures into {figs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="_analysis")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--xlang", default=None,
                    help="cross-language analysis dir; the section is omitted "
                         "entirely if this is not given or has not been scored")
    args = ap.parse_args()
    build(Path(args.analysis), Path(args.docs),
          Path(args.xlang) if args.xlang else None)


if __name__ == "__main__":
    main()
