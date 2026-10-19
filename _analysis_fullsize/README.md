# Analysis outputs — which directory is which

Three analysis directories exist because two of them are superseded and kept only
so the change between them can be checked rather than taken on trust.

| Directory | What it is | Use it? |
|---|---|---|
| `_analysis_fullsize/` | **The current results.** Experiment 1: all 426 prompts in four languages, 13 models, generated on ARC. What the results page shows. | **Yes** |
| `_analysis_xlang/` | **The current results.** Experiments 2 and 3: Python, Java, C++ and C, scored with one tree-sitter detector, in all four prompt languages. | **Yes** |
| `_analysis/` | The original Colab pilot: 426 English prompts, 75 per non-English language. Superseded. | No |
| `_analysis_pilot9/` | The pilot restricted to the same models as the full run. Exists only to separate sample size from model mix. | No |

## Why the pilot is superseded rather than merely smaller

The pilot kept one prompt per smell per difficulty by taking the **first
alphabetically**, which is a systematic subsample, not a random one. Holding the
model set fixed and moving to the full 426 prompts, Chinese goes from 88.6% valid
output to 97.7% — from apparently the worst language to marginally the best. Any
figure quoted from `_analysis/` should be re-checked against
`_analysis_fullsize/`.

## Which code is scored

Scores are computed on the code `pipeline/extract.py` takes out of each untouched
reply: the first fenced block in the requested language, or, for a reply with no
fence, the code between the prose. The extraction run at generation time joined
every fenced block and needed a fence; `extraction_check.csv` compares the two cell
by cell.

## Files

| File | Contents |
|---|---|
| `per_file.csv` | One row per generated file: validity, size, complexity, lint counts, every smell detected, and whether the targeted smell appeared |
| `effects.csv` | Every difference the page reports (each language against English), with a 95% interval from paired prompt resamples and how many models move the same way |
| `extraction_check.csv` | Validity and induction per model and language under the generation-time extraction and the current one |
| `by_lang.csv` | Aggregated by prompt language |
| `by_lang_matched.csv` | Same, restricted to prompts present in **every** language — the table to quote |
| `by_model_lang.csv` | Each model in each language |
| `by_smell.csv` | Induction rate per targeted smell, with its base rate and lift |
| `by_category.csv`, `by_level.csv` | By smell category and by difficulty |
| `figures/` | Figures 1 to 5; the results page shows 4 and 5 |

Regenerate everything with `bash pipeline/rebuild_results.sh <outputs>`.

Raw generations are not here — they are a release asset, since they are 150 MB:
https://github.com/kaiyi03/code-smell-multilingual/releases
