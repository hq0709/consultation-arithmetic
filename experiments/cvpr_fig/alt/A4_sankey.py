"""A4_sankey — candidate for Fig. 4 (c), full width (174.6 mm): alluvial plots.

Claim: on MedXpertQA-MM most agent answers have the same correctness with the original and with the
blank image; on VQA-RAD and SLAKE the image changes the correctness of a third to a half of them.
Each panel: every initial answer of the Independent MAS of 9 (250 items), right or wrong with the
original image (left) and with the blank image (right). Dark ribbons: correctness changed.
Data: results/imaging_v2_A.json, key prior_decomposition.
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
from cf_style import (MM, PAL, MODELS, MODEL, SETS, SET_LABEL, BASE_PT, LEGEND_PT, ANNOT_PT, load)  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment                              # noqa: E402
from alluvial import alluvial                                                                         # noqa: E402

from matplotlib.patches import Patch                                                                  # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "A4_sankey"), str(QA / "A4_sankey")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 96

A = load("A")
COL = {("R", "R"): PAL["pos_soft"], ("R", "W"): PAL["pos_dark"], ("W", "R"): PAL["neg_dark"], ("W", "W"): PAL["neg_soft"]}

fig, axes = plt.subplots(3, 4, figsize=(width_mm * MM, height_mm * MM),
                         gridspec_kw=dict(left=0.13, right=0.985, bottom=0.12, top=0.9, wspace=0.36, hspace=0.42))
for i, s in enumerate(SETS):
    for j, m in enumerate(MODELS):
        ax = axes[i, j]
        p = A[f"{m}|{s}"]["prior_decomposition"]
        flows = {("R", "R"): p["prior_right_stays_right"], ("R", "W"): p["image_corrects_prior"],
                 ("W", "R"): p["image_misleads_prior"], ("W", "W"): p["prior_wrong_stays_wrong"]}
        lp, rp = alluvial(ax, ["R", "W"], ["R", "W"], flows, COL, gap=0.06, w=0.07)
        changed = flows[("R", "W")] + flows[("W", "R")]
        ax.text(0.5, -0.1, f"changed {changed:.0f}%", transform=ax.transAxes, ha="center", va="top",
                fontsize=ANNOT_PT, color=PAL["ink"])
        if i == 0 and j == 0:
            for nd, (a, b) in lp.items():
                ax.text(-0.06, (a + b) / 2, {"R": "Right", "W": "Wrong"}[nd], ha="right", va="center", fontsize=ANNOT_PT)
        if i == 0:
            ax.set_title(MODEL[m]["label"], fontsize=BASE_PT, pad=10)
            ax.text(0.035, 1.03, "Original", transform=ax.transAxes, ha="center", va="bottom", fontsize=ANNOT_PT,
                    color=PAL["neutral_dark"])
            ax.text(0.965, 1.03, "Blank", transform=ax.transAxes, ha="center", va="bottom", fontsize=ANNOT_PT,
                    color=PAL["neutral_dark"])
        if j == 0:
            ax.text(-0.42, 0.5, SET_LABEL[s], transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=BASE_PT, fontweight="bold")
        ax.axis("off")
fig.legend(handles=[Patch(color=COL[("R", "R")], label="Right with both images"),
                    Patch(color=COL[("R", "W")], label="Right only with original"),
                    Patch(color=COL[("W", "R")], label="Wrong only with original"),
                    Patch(color=COL[("W", "W")], label="Wrong with both images")],
           loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.0), handlelength=1.0, handleheight=0.75,
           columnspacing=1.3, handletextpad=0.4, fontsize=LEGEND_PT, borderaxespad=0.2)

require_matplotlib_panel_alignment(fig, json_out=QBASE + ".alignment.json", overlay_svg=QBASE + ".alignment.svg",
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
