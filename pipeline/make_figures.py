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
  fig4  the cross-language arm, smell by smell -- how far apart the four
        programming languages are on each smell, which the flat average hides

Usage:
    python -m pipeline.make_figures --analysis _analysis_fullsize --xlang _analysis_xlang
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
    """Induction rate per smell, against the rate when nobody asked for it.

    Plotted as a pair of points joined by a line rather than as bars, because the
    quantity that matters is the distance between them: a detector whose two
    points nearly coincide is reporting how common the pattern is, not whether the
    prompt caused it.
    """
    rows = sorted(rows, key=lambda r: num(r, "lift"))
    labels = [r["target_smell"] for r in rows]
    asked = [num(r, "induction_valid") for r in rows]
    base = [num(r, "base_rate") for r in rows]
    lift = [num(r, "lift") for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 6.4), facecolor=SURFACE)
    y = np.arange(len(labels))
    for i, (a, b) in enumerate(zip(asked, base)):
        ax.plot([b, a], [i, i], color=GRID, lw=2.4, zorder=1,
                solid_capstyle="round")
    ax.scatter(base, y, s=46, color=C_RAW, zorder=3, label="not asked for (base rate)")
    ax.scatter(asked, y, s=46, color=C_ADJ, zorder=3, label="asked for")
    for i, (a, lf) in enumerate(zip(asked, lift)):
        # Keep a decimal where rounding would print 100% for something short of it
        # -- "100%" reads as "always", and 99.7% is not always.
        shown = f"{a:.1f}%" if a < 100 and round(a) == 100 else f"{a:.0f}%"
        ax.annotate(f"{shown}   +{lf:.0f}", (a, i), textcoords="offset points",
                    xytext=(9, 0), va="center", fontsize=8.5, color=INK)
    style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(False)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  ⚠" if lf < 20 else l for l, lf in zip(labels, lift)],
                       fontsize=9.5, color=INK)
    ax.set_xlim(-2, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("generations containing the smell (%)", fontsize=9.5, color=INK)
    ax.set_title("Which smells a model will produce on request — and which detectors\n"
                 "can tell the difference", fontsize=11.5, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.008, 0.02, "Valid Python only. Numbers are the asked-for rate and the "
             "lift over base rate. ⚠ marks a lift under 20 points — reported, not relied on.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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


# Okabe-Ito: four hues that stay distinct under the common colour-vision
# deficiencies, which four arbitrary hues usually do not.
PLANG_COLOUR = {"python": "#0072B2", "java": "#009E73", "cpp": "#E69F00", "c": "#D55E00"}
PLANG_NAME = {"python": "Python", "java": "Java", "cpp": "C++", "c": "C"}
SHORT = {"Comments (as smell indicator)": "Comments",
         "God Class / Large Class": "God Class",
         "Magic Numbers/Strings": "Magic Numbers"}


def fig4(rows, out):
    """Each smell's rate in each programming language, sorted by the gap.

    The aggregate across smells is flat to within two points, and it is flat
    because these differences cancel: drawn as an average it would show nothing.
    One row per smell, one point per language, a bar from the lowest to the
    highest, so the reader sees the spread rather than reconstructing it.
    """
    by = {}
    for r in rows:
        v = num(r, "induction_valid")
        if v == v:                                   # skip blanks (NaN != NaN)
            by.setdefault(r["target_smell"], {})[r["plang"]] = v
    smells = sorted(by, key=lambda s: max(by[s].values()) - min(by[s].values()))

    fig, ax = plt.subplots(figsize=(9.2, 6.2), facecolor=SURFACE)
    for i, sm in enumerate(smells):
        vals = by[sm]
        lo, hi = min(vals.values()), max(vals.values())
        ax.plot([lo, hi], [i, i], color=GRID, lw=4, zorder=1, solid_capstyle="round")
        for pl, v in vals.items():
            ax.scatter(v, i, s=58, color=PLANG_COLOUR[pl], zorder=3,
                       edgecolor=SURFACE, linewidth=0.8)
        ax.annotate(f"{hi - lo:.0f} pts", (104, i), va="center", fontsize=9,
                    color=INK, annotation_clip=False)
    # Name the two ends of the widest gap, which is the finding the eye starts on.
    top = smells[-1]
    vals = by[top]
    for pl in (max(vals, key=vals.get), min(vals, key=vals.get)):
        # One decimal: 42.5 rounds to 42 half-to-even and to 43 half-up, and a
        # label that disagrees with the prose by a point invites the wrong question.
        ax.annotate(f"{PLANG_NAME[pl]} {vals[pl]:.1f}%", (vals[pl], len(smells) - 1),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=INK, fontweight="bold")
    style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(False)
    ax.set_yticks(range(len(smells)))
    ax.set_yticklabels([SHORT.get(s, s) for s in smells], fontsize=9.5, color=INK)
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("working files containing the requested smell", fontsize=9.5,
                  color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=PLANG_COLOUR[k],
                          markersize=7, label=PLANG_NAME[k]) for k in PLANG_COLOUR]
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=4,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="_analysis_fullsize")
    ap.add_argument("--xlang", default="_analysis_xlang")
    args = ap.parse_args()
    a = Path(args.analysis)
    figdir = a / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    fig1(read(a / "by_lang_matched.csv"), figdir / "fig1_validity_confound.png")
    fig2(read(a / "by_smell.csv"), figdir / "fig2_induction_by_smell.png")
    fig3(read(a / "by_model_lang.csv"), figdir / "fig3_validity_by_model.png")
    x = Path(args.xlang)
    if (x / "by_plang_smell.csv").exists():
        fig4(read(x / "by_plang_smell.csv"), figdir / "fig4_smell_gap_by_language.png")


if __name__ == "__main__":
    main()
