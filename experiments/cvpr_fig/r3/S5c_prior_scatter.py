"""S5c_prior_scatter -- round-3 candidate for the language-prior slot, one column (83.34 mm).

Claim: replacing the image with a blank image costs single agents most of their accuracy above
chance on VQA-RAD and SLAKE, while on MedXpertQA-MM three of the four models stay close to their
original-image accuracy (points near the diagonal answer without using the image).
Encodings: x, single-agent accuracy with the original image (%); y, with a blank image (uniform
gray, same size) (%); crosshairs, item-bootstrap 95% CIs on both axes; colour, model; marker, data
set (VQA-RAD circle, SLAKE square, MedXpertQA-MM triangle); dashed diagonal, equal accuracy (image
unused); dotted lines, chance (50% for the yes/no sets, 20% for the five-option set).
Data: results/imaging_v2_A.json ("real" and "blank" single-agent accuracy, 250 items per set).
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
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, ms)             # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment               # noqa: E402
import s25_common as c                                                               # noqa: E402

import numpy as np                                                                   # noqa: E402
from matplotlib.lines import Line2D                                                  # noqa: E402
from matplotlib.patches import Patch                                                 # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
OUT.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
BASE, QBASE = str(OUT / "S5c_prior_scatter"), str(QA / "S5c_prior_scatter")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 80.0

pts = c.prior_points()
models = c.ordered({p[0] for p in pts})
W, H = width_mm, height_mm
L, B, S = 13.0, 10.5, 67.0                            # left and bottom margins, square plot side (mm)
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([L / W, B / H, S / W, S / H])
LIM = (15, 100)
ax.set_xlim(*LIM); ax.set_ylim(*LIM)

ax.plot(LIM, LIM, color=PAL["neutral_mid"], lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(96.0, 97.2, "blank = original", ha="right", va="bottom", fontsize=ANNOT_PT,   # above the line's end
        color=PAL["neutral_dark"])
for y, lab, x, ha in ((50, "chance, yes/no", 16.0, "left"), (20, "chance, five options", 99.0, "right")):
    ax.axhline(y, color=PAL["neutral_mid"], lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    ax.text(x, y + 0.9, lab, ha=ha, va="bottom", fontsize=ANNOT_PT, color=PAL["neutral_dark"])
for m, s, o, b in pts:
    col = c.style(m)["color"]
    ax.plot([o[1], o[2]], [b[0], b[0]], color=col, lw=0.6, alpha=0.5, solid_capstyle="butt", zorder=3)
    ax.plot([o[0], o[0]], [b[1], b[2]], color=col, lw=0.6, alpha=0.5, solid_capstyle="butt", zorder=3)
    mk = c.SET_MARKER[s]
    # smaller markers drawn later, so coincident points (GPT-5 mini SLAKE, Gemini VQA-RAD) both show
    ax.plot([o[0]], [b[0]], ls="none", marker=mk, ms=ms(mk, 4.4), mfc=col, mec="white", mew=0.4,
            zorder=4 + c.SETS.index(s))
ax.set_xticks(np.arange(20, 101, 20)); ax.set_yticks(np.arange(20, 101, 20))
ax.tick_params(labelsize=TICK_PT)
ax.set_xlabel("Accuracy with original image (%)", fontsize=BASE_PT)
ax.set_ylabel("Accuracy with blank image (%)", fontsize=BASE_PT)

# keys in the empty corner above the diagonal
h_set = [Line2D([], [], ls="none", marker=c.SET_MARKER[s], ms=ms(c.SET_MARKER[s], 4.4), mfc=PAL["neutral_dark"],
                mec="white", mew=0.4, label=c.SET_LABEL[s]) for s in c.SETS]
h_mod = [Patch(fc=c.style(m)["color"], ec="none", label=c.style(m)["label"]) for m in models]
leg1 = ax.legend(handles=h_set, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=LEGEND_PT,
                 handlelength=1.0, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
ax.add_artist(leg1)
ax.legend(handles=h_mod, loc="upper left", bbox_to_anchor=(0.0, 0.79), frameon=False, fontsize=LEGEND_PT,
          handlelength=1.0, handleheight=0.7, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)

require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE, "| points", len(pts))
