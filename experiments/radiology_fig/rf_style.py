"""Shared figure system for the radiology manuscript (paper/radiology), same font stack and size
scale as experiments/nature_fig/nf_style.py.

Widths: the elsarticle preprint text block is 390 pt = 137.2 mm, close to Elsevier's 1.5-column
width (140 mm), so a full-width figure is drawn at 137 mm and prints at its true size.
Model colours: one categorical set validated with the dataviz skill's palette validator
(--pairs all, light surface): worst CVD dE 9.8, worst normal-vision dE 19.6. Teal and amber sit
below 3:1 contrast on white, so every model also has its own marker and a legend or label.
"""
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

RF = pathlib.Path(__file__).resolve().parent
REPO = RF.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "nature_fig"))
from nf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, PANEL_LABEL_PT,  # noqa: E402,F401
                      add_panel_label, add_panel_title, minus)

# RF_TEXT_W_MM / RF_OUT re-render a figure at another print width into another folder (the CVPR
# supplement uses \textwidth = 174.6 mm), RF_FONT=times in Times New Roman; defaults are the radiology
# manuscript's
TEXT_W_MM = float(os.environ.get("RF_TEXT_W_MM", "137.2"))
COL_W_MM = 90.0
OUT = pathlib.Path(os.environ["RF_OUT"]) if os.environ.get("RF_OUT") else REPO / "paper" / "radiology" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "results"


def setup():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
    mpl.rcParams.update({
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": BASE_PT, "axes.labelsize": BASE_PT, "axes.titlesize": BASE_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": LEGEND_PT,
        "axes.linewidth": 0.8, "axes.spines.right": False, "axes.spines.top": False,
        "axes.grid": False, "legend.frameon": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "axes.edgecolor": PAL["ink"], "text.color": PAL["ink"],
        "axes.labelcolor": PAL["ink"], "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
    })
    if os.environ.get("RF_FONT") == "times":       # CVPR supplement re-renders: Times New Roman, as cf_style
        from matplotlib import font_manager
        for f in sorted((REPO / "experiments" / "cvpr_fig" / "fonts").glob("TimesNewRoman-*.ttf")):
            font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
        mpl.rcParams.update({"mathtext.fontset": "stix"})   # Times-style math; Greek letters italic as in the text


MODELS = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001", "qwen3.8-flash"]
MODEL = {
    "gpt-5-mini":                {"label": "GPT-5 mini",       "color": "#454F91", "marker": "o"},
    "gemini-3.7-flash":          {"label": "Gemini 3.7 Flash", "color": "#3AAFA0", "marker": "s"},
    "claude-haiku-4-5-20251001": {"label": "Claude Haiku 4.5", "color": "#C98A1E", "marker": "D"},
    "qwen3.8-flash":             {"label": "Qwen 3.8 Flash",   "color": "#B83B5E", "marker": "^"},
}
SETS = ["vqarad", "slake", "mxqa_mm_rad"]
SET_LABEL = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM",
             "slake_target": "SLAKE-target", "slake_pairs": "SLAKE-pairs"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    return OUT / f"{name}.pdf"


def errbar(ax, x, ci, color, marker, size=4.2, lw=0.9, zorder=3, filled=True, label=None):
    """Point estimate with its 95% CI as a thin vertical bar."""
    est, lo, hi = ci
    ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    ax.plot([x], [est], marker=marker, ms=size, color=color, mfc=color if filled else "white",
            mec=color, mew=0.9, lw=0, zorder=zorder + 1, label=label)
