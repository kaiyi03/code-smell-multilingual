# Cross-language results — experiments 2 and 3

Python, Java, C++ and C, scored with one tree-sitter detector and one set of
thresholds (`detector/cross_language.py`), so each smell has a single definition in
every language. Python is scored again here, rather than read from
`_analysis_fullsize/`, so the four languages share one instrument.

- **Experiment 2** compares the programming languages on English prompts.
- **Experiment 3** crosses them with the prompt language: Spanish, French and
  Chinese versions of the Java, C++ and C prompts, for the same five models.

## Files

| File | Contents |
|---|---|
| `per_file.csv` | One row per generated file: programming and prompt language, model, validity, error nodes, size, headerless `#include` lines, every smell detected, whether the requested smell appeared, tokens used and whether the output limit was reached |
| `by_plang.csv` | Experiment 2 by programming language, English prompts |
| `by_plang_matched.csv` | The same on prompts asked in all four languages, which is the table to quote (C is asked 292 of the 426) |
| `by_plang_model.csv` | Each model in each programming language |
| `by_plang_smell.csv` | Each smell in each language: induction, base rate and lift |
| `by_plang_lang.csv` | Experiment 3: every prompt language in every programming language, same five models |
| `by_plang_lang_model.csv` | The same, per model |
| `effects.csv` | Every difference the page reports (programming language against Python; prompt language against English inside each programming language; and the interaction), with a 95% interval from paired prompt resamples and how many models move the same way |
| `mixed_model.csv` | The same questions as a logistic mixed-effects model with random intercepts for prompt and model: odds ratios and their intervals |
| `translation_check.csv` | How many named identifiers survive machine translation, and whether prompts that lost one show a larger drop from English |
| `extraction_check.csv` | Validity and induction per cell under the generation-time extraction and the current one |

Induction is computed on files that parse and whose requested smell this detector
can decide in that language; an undecidable smell is blank, never a miss.

Regenerate everything with `bash pipeline/rebuild_results.sh <outputs>`.
