"""A5_scatter — candidate for Fig. 5, one column (83.34 mm): scatter with 95% CI crosshairs.

Claim: models whose answers follow the queried region also answer paired images correctly more
often among the proprietary models; open-weight models spread along both axes.
x: occlusion sensitivity on SLAKE-target (pp); y: pair accuracy on SLAKE-pairs (%).
Data: results/imaging_v2_A.json, results/imaging_v2_D.json.
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
from cf_style import (MM, PAL, API3, OW, OW_LABEL, MODEL, LEGEND_PT, ANNOT_PT, load, ms, signed)  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment                          # noqa: E402

from matplotlib.lines import Line2D                                                              # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "A5_scatter"), str(QA / "A5_scatter")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 72

A, D = load("A"), load("D")
pts = []
for m in API3:
    pts.append((MODEL[m]["label"], A[f"{m}|slake_target"]["panel9_region_dependence"],
                A[f"{m}|slake_pairs"]["single_both_correct"], MODEL[m]["color"], MODEL[m]["marker"]))
for m in OW:
    pts.append((OW_LABEL[m], D[f"{m}|slake_target"]["region_dependence_flip"], D[f"{m}|slake_pairs"]["both_correct"],
                PAL["neutral_dark"], "o"))
# label offsets (points) chosen at final size so labels clear the crosshairs and each other
OFF = {"GPT-5 mini": (3, 19), "Gemini 3.7 Flash": (-6, 6), "Claude Haiku 4.5": (6, -9), "MedGemma 4B": (0, 18),
       "Qwen3-VL 8B": (6, 9), "Qwen2-VL 7B": (6, -12), "HuatuoGPT-Vision 7B": (6, -9), "LLaVA-1.5 7B": (6, -8)}
HA = {"Gemini 3.7 Flash": "right", "MedGemma 4B": "center"}

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
ax = fig.add_axes([0.15, 0.17, 0.80, 0.72])
ax.axvline(0, color=PAL["neutral_light"], lw=0.7, zorder=0)
for lab, x, y, col, mk in pts:
    ax.plot([x[1], x[2]], [y[0], y[0]], color=col, lw=0.8, solid_capstyle="butt", zorder=2)
    ax.plot([x[0], x[0]], [y[1], y[2]], color=col, lw=0.8, solid_capstyle="butt", zorder=2)
    ax.plot([x[0]], [y[0]], ls="none", marker=mk, ms=ms(mk, 4.4), mfc=col, mec="white", mew=0.5, zorder=3)
    dx, dy = OFF[lab]
    ax.annotate(lab, (x[0], y[0]), xytext=(dx, dy), textcoords="offset points", ha=HA.get(lab, "left"),
                va="center", fontsize=ANNOT_PT, color=PAL["ink"])
ax.set_xlim(-6, 42)
ax.set_ylim(20, 85)
ax.set_xticks([0, 10, 20, 30, 40])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: signed(v)))
ax.set_xlabel("Occlusion sensitivity, target − control (pp)")
ax.set_ylabel("Pair accuracy (%)")
ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.4, mfc=PAL["ink"], mec="white", label="Proprietary (colour)"),
                   Line2D([], [], ls="none", marker="o", ms=4.4, mfc=PAL["neutral_dark"], mec="white", label="Open-weight")],
          loc="lower right", handlelength=0.9, handletextpad=0.3, fontsize=LEGEND_PT, borderaxespad=0.3)

require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
