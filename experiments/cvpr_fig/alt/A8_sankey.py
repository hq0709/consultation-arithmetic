"""A8_sankey — candidate for Fig. 8, full width (174.6 mm): alluvial plots of debate.

Claim: debate keeps correct majority answers and turns most correct minority answers wrong.
Each panel: every agent answer of the Decentralized MAS of 9 (4 models x 250 items x 9 agents),
from its initial state (correct and in the initial majority, correct and in the minority, wrong) to
its final state. Counts summed over the four models; the annotation gives the per-model range.
Data: results/imaging_v2_A.json, key discussion.conformity.
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
from cf_style import (MM, PAL, MODELS, SETS, SET_LABEL, BASE_PT, LEGEND_PT, ANNOT_PT, load, add_panel_label)  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment                                        # noqa: E402
from alluvial import alluvial                                                                                   # noqa: E402

from matplotlib.patches import Patch                                                                            # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "A8_sankey"), str(QA / "A8_sankey")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 64

A = load("A")
LEFT, RIGHT = ["Maj", "Min", "Wr"], ["R", "W"]
LL = {"Maj": "Correct, initial majority", "Min": "Correct, initial minority", "Wr": "Wrong initially"}
RL = {"R": "Correct finally", "W": "Wrong finally"}
COL = {("Maj", "R"): PAL["pos_soft"], ("Min", "R"): PAL["pos_soft"], ("Maj", "W"): PAL["neg_dark"],
       ("Min", "W"): PAL["neg_dark"], ("Wr", "R"): PAL["pos_dark"], ("Wr", "W"): PAL["neg_soft"]}

fig, axes = plt.subplots(1, 3, figsize=(width_mm * MM, height_mm * MM),
                         gridspec_kw=dict(left=0.165, right=0.885, bottom=0.2, top=0.88, wspace=0.18))
for ax, s, letter in zip(axes, SETS, "abc"):
    T, rates = {}, []
    for m in MODELS:
        c = A[f"{m}|{s}"]["discussion"]["conformity"]
        rmaj, rmin = c["n_right_majority"], c["n_right_minority"]
        cmaj, cmin = c["right_to_wrong_majority"][0] / 100, c["right_to_wrong_minority"][0] / 100
        nwi, nwp = c["n_wrong_image_dependent"], c["n_wrong_prior_consistent"]
        w2r = c["wrong_to_right_image_dependent"][0] / 100 * nwi + c["wrong_to_right_prior_consistent"][0] / 100 * nwp
        f = {("Maj", "R"): rmaj * (1 - cmaj), ("Maj", "W"): rmaj * cmaj, ("Min", "R"): rmin * (1 - cmin),
             ("Min", "W"): rmin * cmin, ("Wr", "R"): w2r, ("Wr", "W"): nwi + nwp - w2r}
        for k, v in f.items():
            T[k] = T.get(k, 0.0) + v
        rates.append(c["right_to_wrong_minority"][0])
    assert round(sum(T.values())) == 9000
    lp, rp = alluvial(ax, LEFT, RIGHT, T, COL, gap=0.05, w=0.07)
    if letter == "a":                   # initial states once, at the left of the first panel
        for nd, (a, b) in lp.items():
            ax.text(-0.05, (a + b) / 2, LL[nd], ha="right", va="center", fontsize=ANNOT_PT)
    if letter == "c":                   # final states once, at the right of the last panel
        for nd, (a, b) in rp.items():
            ax.text(1.05, (a + b) / 2, RL[nd], ha="left", va="center", fontsize=ANNOT_PT)
    ax.set_title(SET_LABEL[s], fontsize=BASE_PT, pad=6)
    ax.text(0.5, -0.05, f"Minority changed to wrong:\n{min(rates):.1f}% to {max(rates):.1f}% (by model)",
            transform=ax.transAxes, ha="center", va="top", fontsize=ANNOT_PT, color=PAL["ink"])
    ax.axis("off")
fig.legend(handles=[Patch(color=PAL["pos_soft"], label="Stayed correct"), Patch(color=PAL["neg_dark"], label="Changed to wrong"),
                    Patch(color=PAL["pos_dark"], label="Changed to correct"), Patch(color=PAL["neg_soft"], label="Stayed wrong")],
           loc="lower center", ncol=4, bbox_to_anchor=(0.52, -0.01), handlelength=1.0, handleheight=0.75,
           columnspacing=1.4, handletextpad=0.4, fontsize=LEGEND_PT, borderaxespad=0.2)

require_matplotlib_panel_alignment(fig, json_out=QBASE + ".alignment.json", overlay_svg=QBASE + ".alignment.svg",
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
