"""S1b_neff_saturation — candidate for the agent-scaling display (Sec. 4.3), column width (83.34 mm).

Claim: nine agents of one VLM act as 1.2 to 2.0 independent agents, and no number of agents can
exceed 1/phi.
Encoding: x, number of agents k; y, effective number of independent agents N_eff(k) = k / (1 + (k - 1) phi)
on a log2 axis, for every model and data set (phi of the Independent MAS of 9 with original images,
250 items per data set). Dashed grey line, N_eff = k (independent agents). Line style, data set; colour
and marker, model. At "inf", the asymptote 1/phi, joined to N_eff(9) by a dotted segment.
Data: results/cvpr_nmi.json ("scaling"). Nature-figure skill, Python backend.
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
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, ms)        # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment             # noqa: E402
sys.path.insert(0, str(ALT))
from s1_common import SETS, SET_LABEL, style, models, cell, ROOT                 # noqa: E402

import numpy as np                                                              # noqa: E402
from matplotlib.lines import Line2D                                             # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator                         # noqa: E402

OUT = ROOT / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
OUT.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
BASE, QBASE = str(OUT / "S1b_neff_saturation"), str(QA / "S1b_neff_saturation")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 72.0
W, H = width_mm, height_mm
LS = {"vqarad": "-", "slake": (0, (4, 1.6)), "mxqa_mm_rad": (0, (1, 1.4))}
XINF = 11.0                                     # x position of the asymptote column
MS_ = models()

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([13.5 / W, 11.5 / H, (W - 16.0) / W, (H - 14.0) / H])
k = np.linspace(1, 9, 200)
ax.plot([1, 9], [1, 9], color=PAL["neutral_mid"], lw=1.0, ls=(0, (3.2, 1.6)), zorder=1)
ax.text(8.9, 9.25, "independent agents", ha="right", va="bottom", fontsize=ANNOT_PT, color=PAL["neutral_dark"])
for m in MS_:
    st = style(m)
    for s in SETS:
        c = cell(m, s)
        if not c or not c["phi"]:
            continue
        ph = c["phi"]
        ax.plot(k, k / (1 + (k - 1) * ph), color=st["color"], lw=1.1, ls=LS[s], zorder=3, solid_capstyle="butt")
        ax.plot([9, XINF], [9 / (1 + 8 * ph), 1 / ph], color=st["color"], lw=0.8, ls=(0, (1, 1.2)), zorder=2)
        for x, y in ((9, 9 / (1 + 8 * ph)), (XINF, 1 / ph)):
            ax.plot([x], [y], marker=st["marker"], ms=ms(st["marker"], 3.4), color=st["color"], zorder=4,
                    mfc=st["color"] if s == "vqarad" else "white", mec=st["color"], mew=0.8, lw=0)
ax.set_yscale("log", base=2)        # every plotted value is >= 1 (N_eff of one agent), no guard needed
ax.set_ylim(0.9, 12.5)
ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
ax.set_xlim(0.5, XINF + 0.6)
ax.set_xticks([1, 3, 5, 7, 9, XINF], ["1", "3", "5", "7", "9", r"$\infty$"])
ax.axvspan(9.8, XINF + 0.6, color=PAL["neutral_faint"], lw=0, zorder=0)
ax.set_xlabel("Number of agents", fontsize=BASE_PT)
ax.set_ylabel(r"Effective number of agents, $N_{\mathrm{eff}}$", fontsize=BASE_PT)
ax.tick_params(labelsize=TICK_PT)
h_models = [Line2D([], [], color=style(m)["color"], lw=1.1, marker=style(m)["marker"], ms=ms(style(m)["marker"], 3.4),
                   mfc=style(m)["color"], mec=style(m)["color"], label=style(m)["label"]) for m in MS_]
h_sets = [Line2D([], [], color=PAL["neutral_dark"], lw=1.1, ls=LS[s], label=SET_LABEL[s]) for s in SETS]
leg1 = ax.legend(handles=h_models, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=LEGEND_PT, frameon=False,
                 handlelength=2.0, borderaxespad=0.3, labelspacing=0.3)
ax.add_artist(leg1)
ax.legend(handles=h_sets, loc="upper left", bbox_to_anchor=(0.355, 0.56), fontsize=LEGEND_PT, frameon=False,
          handlelength=2.0, borderaxespad=0.3, labelspacing=0.3)

require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg", dpi=600)
fig.savefig(BASE + ".pdf", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
