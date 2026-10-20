"""T_units — candidate teaser, one column (83.34 mm): unit chart.

Claim: a system of 9 agents from one VLM carries the information of 1.17 to 1.99 independent agents.
Each row: 9 circles, one per agent; the filled area equals N_eff = 9 / (1 + 8 phi) (Independent MAS,
initial answers, original images).
Data: results/imaging_v2_A.json.
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
from cf_style import (MM, PAL, MODELS, MODEL, SETS, SET_LABEL, BASE_PT, TICK_PT, ANNOT_PT, load,  # noqa: E402
                      neff)
from audit_panel_alignment import require_matplotlib_panel_alignment                          # noqa: E402

from matplotlib.patches import Circle, Wedge                                                 # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "T_units"), str(QA / "T_units")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 72

A = load("A")
GAP = 1.4
ypos, y_head, y = {}, {}, 0.0
for s in SETS:
    y_head[s] = y
    y -= 1.0
    for m in MODELS:
        ypos[(s, m)] = y
        y -= 1.0
    y -= GAP - 1.0
R = 0.36                                    # circle radius in row units

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
ax = fig.add_axes([0.30, 0.02, 0.56, 0.93])
for (s, m), yy in ypos.items():
    ne = float(neff(A[f"{m}|{s}"]["real"]["phi"][0]))
    col = MODEL[m]["color"]
    for i in range(9):
        x = float(i)
        ax.add_patch(Circle((x, yy), R, facecolor="white", edgecolor=PAL["neutral_light"], lw=0.7, zorder=2))
        share = min(max(ne - i, 0.0), 1.0)
        if share >= 1:
            ax.add_patch(Circle((x, yy), R, facecolor=col, edgecolor=col, lw=0.7, zorder=3))
        elif share > 0:                     # wedge from 12 o'clock, clockwise: area share = fraction
            ax.add_patch(Wedge((x, yy), R, 90 - 360 * share, 90, facecolor=col, edgecolor="none", zorder=3))
    ax.text(9.0, yy, f"{ne:.2f}", ha="left", va="center", fontsize=TICK_PT, color=PAL["ink"])
    ax.text(-0.9, yy, MODEL[m]["label"], ha="right", va="center", fontsize=TICK_PT, color=PAL["ink"])
for s in SETS:
    ax.text(-0.9, y_head[s], SET_LABEL[s], ha="right", va="center", fontsize=BASE_PT, fontweight="bold")
top = max(y_head.values())
ax.text(4.0, top + 0.95, "9 agents; filled area, effective number of independent agents", ha="center",
        va="bottom", fontsize=ANNOT_PT, color=PAL["neutral_dark"])
ax.text(9.0, top, r"$N_{\mathrm{eff}}$", ha="left", va="center", fontsize=BASE_PT)
ax.set_xlim(-0.6, 9.6)
ax.set_ylim(min(ypos.values()) - 0.6, top + 1.6)
ax.set_aspect("equal", adjustable="box")
ax.axis("off")

require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
