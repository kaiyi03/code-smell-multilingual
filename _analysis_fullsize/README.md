# Analysis outputs — which directory is which

Three analysis directories exist because two of them are superseded and kept only
so the change between them can be checked rather than taken on trust.

| Directory | What it is | Use it? |
|---|---|---|
| `_analysis_fullsize/` | **The current results.** All 426 prompts in four languages, generated on ARC. What the results page shows. | **Yes** |
| `_analysis/` | The original Colab pilot: 426 English prompts, 75 per non-English language. Superseded. | No |
| `_analysis_pilot9/` | The pilot restricted to the same models as the full run. Exists only to separate sample size from model mix. | No |

## Why the pilot is superseded rather than merely smaller

The pilot kept one prompt per smell per difficulty by taking the **first
alphabetically**, which is a systematic subsample, not a random one. Holding the
model set fixed and moving to the full 426 prompts, Chinese goes from 88.6% valid
output to 97.7% — from apparently the worst language to marginally the best. Any
figure quoted from `_analysis/` should be re-checked against
`_analysis_fullsize/`.

## Files

| File | Contents |
|---|---|
| `per_file.csv` | One row per generated file: validity, size, complexity, lint counts, every smell detected, and whether the targeted smell appeared |
| `by_lang.csv` | Aggregated by prompt language |
| `by_lang_matched.csv` | Same, restricted to prompts present in **every** language — the table to quote |
| `by_model_lang.csv` | Each model in each language |
| `by_smell.csv` | Induction rate per targeted smell, with its base rate and lift |
| `by_category.csv`, `by_level.csv` | By smell category and by difficulty |
| `figures/` | The three figures on the results page |

Regenerate with `python -m pipeline.run_analysis --root <outputs> --out _analysis_fullsize`.

Raw generations are not here — they are a release asset, since they are 150 MB:
https://github.com/kaiyi03/code-smell-multilingual/releases
