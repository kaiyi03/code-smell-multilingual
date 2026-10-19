#!/usr/bin/env python3
"""
The language effects as a logistic mixed-effects model, beside the resampling.

pipeline/effects.py answers each question with a difference in percentage points
and an interval from resampling prompts, with the models held fixed. This asks the
same questions the way a reviewer will expect: whether a working file contains the
requested smell, modelled with the language as a fixed effect and random
intercepts for the prompt and for the model, crossed. Treating the model as random
asks whether the effect generalises beyond these particular models, which holding
them fixed does not.

    induction ~ prompt language                       Experiment 1, Python, 13 models
    induction ~ programming language                  Experiment 2, English, 5 models
    induction ~ prompt language x programming lang.   Experiment 3, 5 models

Fitted by variational Bayes (statsmodels BinomialBayesMixedGLM), which is fast on
twenty thousand files but tends to understate posterior spread; read the intervals
as a check on direction and rough size, with effects.py as the primary estimate.
Effects are odds ratios against English, or against Python; for Experiment 3 the
interaction terms say how much more (or less) a prompt language moves Java, C++ or
C than it moves Python.

    python -m pipeline.mixed_model --analysis _analysis_fullsize --xlang _analysis_xlang
"""

import argparse
import csv
import math
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

EXCLUDED = {"mamba-codestral-7b"}
SEED = 7


def frame(path, xlang):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[(~df["model"].isin(EXCLUDED)) & (df["syntax_ok"] == "1")
            & (df["target_covered"] == "1")].copy()
    df["hit"] = (df["target_hit"] == "1").astype(int)
    if xlang:
        five = set(df.loc[df["plang"] != "python", "model"])
        df = df[df["model"].isin(five)]
    return df


def fit(df, formula, label):
    """Fit, then return one row per fixed effect other than the intercept."""
    model = BinomialBayesMixedGLM.from_formula(
        formula, {"prompt": "0 + C(prompt_id)", "model": "0 + C(model)"}, df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit_vb()
    names = model.exog_names
    out = []
    for i, name in enumerate(names):
        if name == "Intercept":
            continue
        mean, sd = res.fe_mean[i], res.fe_sd[i]
        out.append({"experiment": label, "term": name, "n_files": len(df),
                    "log_odds": round(mean, 4), "sd": round(sd, 4),
                    "odds_ratio": round(math.exp(mean), 3),
                    "or_lo": round(math.exp(mean - 1.96 * sd), 3),
                    "or_hi": round(math.exp(mean + 1.96 * sd), 3)})
    vc = dict(zip(model.vcp_names, res.vcp_mean))
    for k, v in vc.items():
        out.append({"experiment": label, "term": f"sd of random intercept: {k}",
                    "n_files": len(df), "log_odds": "", "sd": round(math.exp(v), 4),
                    "odds_ratio": "", "or_lo": "", "or_hi": ""})
    return out


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="_analysis_fullsize")
    ap.add_argument("--xlang", default="_analysis_xlang")
    args = ap.parse_args()

    e1 = frame(Path(args.analysis) / "per_file.csv", xlang=False)
    out1 = fit(e1, "hit ~ C(lang, Treatment('en'))", "prompt language")
    write(Path(args.analysis) / "mixed_model.csv", out1)

    x = frame(Path(args.xlang) / "per_file.csv", xlang=True)
    out2 = fit(x[x["lang"] == "en"], "hit ~ C(plang, Treatment('python'))",
               "programming language")
    if (x.loc[x["plang"] != "python", "lang"] != "en").any():
        out2 += fit(x, "hit ~ C(lang, Treatment('en')) * C(plang, Treatment('python'))",
                    "both")
    write(Path(args.xlang) / "mixed_model.csv", out2)

    for r in out1 + out2:
        if r["odds_ratio"] != "":
            print(f"  {r['experiment']:21s} {r['term'][:60]:60s} OR {r['odds_ratio']:5.2f} "
                  f"[{r['or_lo']:.2f}, {r['or_hi']:.2f}]")
        else:
            print(f"  {r['experiment']:21s} {r['term']:60s} {r['sd']}")


if __name__ == "__main__":
    main()
