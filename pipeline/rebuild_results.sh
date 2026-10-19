#!/usr/bin/env bash
#
# Rebuild every score, table, figure and the results page from the generations.
#
#   PY=/path/to/python bash pipeline/rebuild_results.sh ../outputs_arc
#
# The generations are read, never written. The code that gets scored is
# re-extracted from the untouched replies into $SCORED (pipeline/extract.py);
# the files the generation jobs extracted are scored once more into $ORIG, only
# so extraction_check can show what the newer extractor changed.
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
# The scripts print non-ASCII (language names, minus signs); Windows consoles need this.
export PYTHONIOENCODING=utf-8
RAW=${1:-../outputs_arc}
SCORED=${SCORED:-../outputs_scored}
ORIG=${ORIG:-../_scored_original}
EXCL="--exclude mamba-codestral-7b"

echo "== tests"
$PY -m pipeline.test_extract
$PY -m pipeline.test_xlang_analysis

echo "== re-extracting every reply"
rm -rf "$SCORED"
$PY -m pipeline.reextract --root "$RAW" --out "$SCORED"

echo "== scoring, current and generation-time extraction side by side"
mkdir -p "$ORIG"
$PY -m pipeline.run_analysis --root "$SCORED" --out _analysis_fullsize $EXCL > "$ORIG/fullsize.log" 2>&1 &
$PY -m pipeline.run_xlang_analysis --root "$SCORED" --out _analysis_xlang $EXCL > "$ORIG/xlang.log" 2>&1 &
$PY -m pipeline.run_analysis --root "$RAW" --out "$ORIG/fullsize" $EXCL > "$ORIG/fullsize_orig.log" 2>&1 &
$PY -m pipeline.run_xlang_analysis --root "$RAW" --out "$ORIG/xlang" $EXCL > "$ORIG/xlang_orig.log" 2>&1 &
wait
tail -n 25 "$ORIG/fullsize.log"
tail -n 40 "$ORIG/xlang.log"

echo "== checks and contrasts"
$PY -m pipeline.extraction_check --original "$ORIG/fullsize" --current _analysis_fullsize --scored "$SCORED"
$PY -m pipeline.extraction_check --original "$ORIG/xlang" --current _analysis_xlang --scored "$SCORED"
$PY -m pipeline.effects
$PY -m pipeline.mixed_model
$PY -m pipeline.translation_check

echo "== figures and page"
$PY -m pipeline.make_figures --analysis _analysis_fullsize --xlang _analysis_xlang
$PY -m pipeline.make_site
