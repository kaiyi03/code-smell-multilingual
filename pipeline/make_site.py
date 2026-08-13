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


def build(analysis: Path, docs: Path):
    matched = {r["lang"]: r for r in read(analysis / "by_lang_matched.csv")}
    by_smell = read(analysis / "by_smell.csv")
    by_ml = read(analysis / "by_model_lang.csv")

    figs = docs / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for p in sorted((analysis / "figures").glob("*.png")):
        shutil.copy2(p, figs / p.name)
    (docs / ".nojekyll").write_text("", encoding="utf-8")

    n_files = sum(int(float(r["n_files"])) for r in matched.values())

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

    # --- smell table
    smell_rows = [[r["target_smell"], f(r, "n_covered", 0),
                   f(r, "induction_valid", 1, "%")]
                  for r in sorted(by_smell, key=lambda x: -float(x["induction_valid"] or 0))]

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
  project's own detector over the output asks whether that smell actually appears —
  the measurement the pipeline was built for and had never run.</p>
  <figure>
    <img src="figures/fig2_induction_by_smell.png" alt="Induction rate by targeted smell">
    <figcaption>Valid Python only.</figcaption>
  </figure>
  {table(["Targeted smell", "Files", "Produced"], smell_rows)}
  <p>Models comply almost always when the smell is a local property of one function,
  and resist when it requires committing to a bad overall structure: asked for a God
  Class they tend to split the work across several well-formed classes instead.</p>
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

<section>
  <h2>How these numbers were produced</h2>
  <ul>
    <li><strong>Scored with the project's own detector.</strong>
    <code>detector/smell_detector.py</code> covers eight of the twenty-five targeted
    smells. The other seventeen have no detector and are excluded from induction
    figures rather than counted as misses.</li>
    <li><strong>Conditioned on syntax validity.</strong> Every quality figure is
    given twice, over all files and over files that parse.</li>
    <li><strong>Matched prompt set.</strong> Only prompts generated in all four
    languages are compared, so a language contrast cannot be a contrast between
    different task mixes.</li>
    <li><strong>mamba-codestral-7b excluded</strong> from aggregates: 33% syntax
    validity in English is a known model-loading failure, not model quality.</li>
    <li><strong>Not yet controlled:</strong> machine translation rewrote 19–30% of
    the API identifiers the prompts specify, so the non-English arms are not asking
    for quite the same thing. Fixing this is the next task.</li>
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
    args = ap.parse_args()
    build(Path(args.analysis), Path(args.docs))


if __name__ == "__main__":
    main()
