"""A6_bubble — candidate for Fig. 6 (a), full width (174.6 mm): bubble matrices.

Claim: reports from the Gemini 3.7 Flash perception module raise every reasoning module above its own
image reading, and Claude Haiku 4.5 reports lower them.
Each facet: reasoning module (rows) by perception module (columns); bubble area and colour, mean agent
accuracy reading that perception module own reports minus the same model reading the image (pp);
numbers, the estimate; outlined, 95% CI excludes zero.
Data: results/imaging_v2_B.json.
Nature-figure skill, Python backend.
"""
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── MANDATORY (nature-figure api.md): sans-serif stack + editable SVG/PDF text ──
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": False,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans",
    "mathtext.cal": "Liberation Sans",
    "mathtext.tt": "Liberation Sans",
    "axes.labelcolor": "#1F1F1F",
    "xtick.color": "#1F1F1F",
    "ytick.color": "#1F1F1F",
    "text.color": "#1F1F1F",
    "axes.edgecolor": "#1F1F1F",
})

ALT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ALT.parent))
sys.path.insert(0, str(pathlib.Path.home() / ".claude" / "skills" / "nature-figure" / "scripts"))
from cf_style import (MM, PAL, API3, MODEL, SETS, SET_LABEL, BASE_PT, TICK_PT, ANNOT_PT, LEGEND_PT, load,  # noqa: E402
                      add_panel_label, add_panel_title)
from audit_panel_alignment import require_matplotlib_panel_alignment                                     # noqa: E402

import numpy as np                                                                                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm                                        # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "A6_bubble"), str(QA / "A6_bubble")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 60

B = load("B")
ORDER = ["gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4-5-20251001"]
SHORT = {"gemini-3.7-flash": "Gemini\n3.7 Flash", "gpt-5-mini": "GPT-5\nmini", "claude-haiku-4-5-20251001": "Claude\nHaiku 4.5"}
CMAP = LinearSegmentedColormap.from_list("div", ["#B2453A", "#E6B3AC", "#F2F2F2", "#A9C4DE", "#2F5D8C"])
NORM = TwoSlopeNorm(vmin=-26, vcenter=0, vmax=26)
SMAX = 26.0                                  # pp at the largest bubble

fig, axes = plt.subplots(1, 3, figsize=(width_mm * MM, height_mm * MM),
                         gridspec_kw=dict(left=0.155, right=0.985, bottom=0.04, top=0.66, wspace=0.14))
for ax, s, letter in zip(axes, SETS, "abc"):
    for i, r in enumerate(ORDER):
        for j, p in enumerate(ORDER):
            est, lo, hi = B[s][r][f"{p}|own"]["reader_acc_minus_image"]
            area = (abs(est) / SMAX) * 420 + 6
            ax.scatter([j], [-i], s=area, c=[CMAP(NORM(est))], edgecolors=PAL["ink"] if (lo > 0 or hi < 0) else "none",
                       linewidths=0.7, zorder=3)
            ax.text(j, -i - 0.36, (f"{est:+.1f}" if abs(est) >= 0.05 else "0.0").replace("-", "−"),
                    ha="center", va="top", fontsize=ANNOT_PT, color=PAL["ink"])
            if r == p:                     # same model as perception and reasoning module
                ax.add_patch(plt.Rectangle((j - 0.47, -i - 0.66), 0.94, 1.0, facecolor="#F1F1F1", edgecolor="none",
                                           zorder=0))
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-2.75, 0.6)
    ax.set_xticks([0, 1, 2], [SHORT[p] for p in ORDER], fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([0, -1, -2], [MODEL[r]["label"] for r in ORDER] if letter == "a" else ["", "", ""], fontsize=TICK_PT)
    ax.tick_params(axis="y", length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    add_panel_label(ax, letter, x_offset_pt=-2, y_offset_pt=29)
    add_panel_title(ax, SET_LABEL[s], x_offset_pt=10, y_offset_pt=29)
fig.text(0.155, 0.985, "Columns: perception module that wrote the reports.   Rows: reasoning module that read them.",
         ha="left", va="top", fontsize=LEGEND_PT, color=PAL["neutral_dark"])

require_matplotlib_panel_alignment(fig, json_out=QBASE + ".alignment.json", overlay_svg=QBASE + ".alignment.svg",
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
