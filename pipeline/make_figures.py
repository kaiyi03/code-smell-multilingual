#!/usr/bin/env python3
"""
Figures for the analysis produced by pipeline/run_analysis.py.

Three figures, each carrying one claim:

  fig1  the syntax-validity confound -- what the headline quality metric says
        before and after conditioning on the code being valid Python
  fig2  induction rate per targeted smell -- does the smell the prompt asked
        for actually appear
  fig3  syntax validity by model and prompt language -- whether the language
        effect is general or belongs to particular models

Usage:
    python -m pipeline.make_figures --analysis _analysis
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Two categorical series, validated for CVD separation (protan dE 19.6,
# normal-vision dE 26.1) against a light surface.
C_RAW = "#C2691F"      # as reported
C_ADJ = "#2E6B9E"      # conditioned on valid output
INK = "#1a1d21"
MUTED = "#6b7280"
GRID = "#dfe3e8"
SURFACE = "#fcfcfb"

SEQ = LinearSegmentedColormap.from_list("blues", ["#f2f6fa", "#9dbdd8", "#2E6B9E", "#173e5e"])

LANG_NAME = {"en": "English", "es": "Spanish", "fr": "French", "zh": "Chinese"}
ORDER = ["en", "es", "fr", "zh"]


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig1(rows, out):
    """The confound: the metric before and after the validity control."""
    rows = {r["lang"]: r for r in rows}
    langs = [l for l in ORDER if l in rows]
    raw = [num(rows[l], "ruff_per_100loc_all") for l in langs]
    adj = [num(rows[l], "ruff_per_100loc_valid") for l in langs]
    ind = [num(rows[l], "induction_valid") for l in langs]
    val = [num(rows[l], "syntax_ok_pct") for l in langs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(langs))
    w = 0.38

    b1 = ax1.bar(x - w / 2, raw, w, color=C_RAW, label="as reported (all files)")
    b2 = ax1.bar(x + w / 2, adj, w, color=C_ADJ, label="on valid Python only")
    for bars in (b1, b2):
        for bar in bars:
            ax1.annotate(f"{bar.get_height():.1f}",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=8.5, color=INK)
    style(ax1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{LANG_NAME[l]}\n{val[i]:.0f}% valid"
                         for i, l in enumerate(langs)], fontsize=9)
    ax1.set_ylabel("ruff violations per 100 lines", fontsize=9.5, color=INK)
    ax1.set_title("The metric is mostly reporting generation failure",
                  fontsize=11, color=INK, loc="left", pad=10)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")

    bars = ax2.bar(x, ind, 0.55, color=C_ADJ)
    for bar in bars:
        ax2.annotate(f"{bar.get_height():.0f}%",
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8.5, color=INK)
    style(ax2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LANG_NAME[l] for l in langs], fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("targeted smell produced (%)", fontsize=9.5, color=INK)
    ax2.set_title("What does survive the control", fontsize=11, color=INK,
                  loc="left", pad=10)

    fig.text(0.008, 0.022, "Matched prompts only — the same tasks in every language",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig2(rows, out):
    """Induction rate per targeted smell."""
    rows = sorted(rows, key=lambda r: num(r, "induction_valid"))
    labels = [r["target_smell"] for r in rows]
    vals = [num(r, "induction_valid") for r in rows]
    ns = [int(num(r, "n_covered", 0)) for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.4), facecolor=SURFACE)
    y = np.arange(len(labels))
    ax.barh(y, vals, 0.62, color=C_ADJ)
    for i, (v, n) in enumerate(zip(vals, ns)):
        # Keep a decimal when rounding would print 100% for something short of it
        # -- "100%" reads as "always", and 99.7% is not always.
        label = f"{v:.1f}%" if v < 100 and round(v) == 100 else f"{v:.0f}%"
        ax.annotate(label, (v, i), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=9, color=INK)
        ax.annotate(f"n={n}", (0, i), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=8, color="#ffffff")
    style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(False)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, 108)
    ax.set_xlabel("generations containing the smell the prompt asked for (%)",
                  fontsize=9.5, color=INK)
    ax.set_title("Models comply readily with some smell requests and resist others",
                 fontsize=11.5, color=INK, loc="left", pad=10)
    fig.text(0.008, 0.02, "Valid Python only. The 17 targeted smells with no "
             "matching detector are not shown.", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig3(rows, out):
    """Validity by model and language -- is the effect general or per-model?"""
    models = sorted({r["model"] for r in rows})
    langs = [l for l in ORDER if any(r["lang"] == l for r in rows)]
    grid = np.full((len(models), len(langs)), np.nan)
    for r in rows:
        if r["model"] in models and r["lang"] in langs:
            grid[models.index(r["model"]), langs.index(r["lang"])] = \
                num(r, "syntax_ok_pct")

    # Sort by worst non-English cell: the models that break go to the bottom.
    order = np.argsort([-np.nanmin(grid[i, 1:]) for i in range(len(models))])
    grid, models = grid[order], [models[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor=SURFACE)
    im = ax.imshow(grid, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    for i in range(len(models)):
        for j in range(len(langs)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.annotate(f"{v:.0f}", (j, i), ha="center", va="center", fontsize=9,
                        color="#ffffff" if v > 62 else INK,
                        fontweight="bold" if v < 70 else "normal")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([LANG_NAME[l] for l in langs], fontsize=9.5, color=INK)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Syntax validity (%) — the language effect belongs to four models",
                 fontsize=11, color=INK, loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    fig.text(0.008, 0.02, "Sorted by worst non-English cell. mamba-codestral-7b "
             "excluded — 33% in English is a known load bug.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="_analysis")
    args = ap.parse_args()
    a = Path(args.analysis)
    figdir = a / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    fig1(read(a / "by_lang_matched.csv"), figdir / "fig1_validity_confound.png")
    fig2(read(a / "by_smell.csv"), figdir / "fig2_induction_by_smell.png")
    fig3(read(a / "by_model_lang.csv"), figdir / "fig3_validity_by_model.png")


if __name__ == "__main__":
    main()
