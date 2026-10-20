"""S5d_prior_slope -- round-3 candidate for the language-prior slot, one column (83.34 mm).

Claim: a blank image costs single agents far more accuracy on VQA-RAD and SLAKE than on
MedXpertQA-MM, where three of the four models keep nearly all of their original-image accuracy
(their answers come from the question and options, not from the image).
Layout: one axes with three groups (VQA-RAD, SLAKE, MedXpertQA-MM; 250 items each) on the same y
axis. In each group, one line per model joins single-agent accuracy with the original image
(filled marker) and with a blank image (uniform gray, same size; open marker); thin bars,
item-bootstrap 95% CIs (models dodged slightly in x); dotted segment, chance (50% yes/no, 20% five
options). Colour and marker, model (as in Fig. 1).
Data: results/imaging_v2_A.json ("real" and "blank" single-agent accuracy).
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
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ms)                      # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment               # noqa: E402
import s25_common as c                                                               # noqa: E402

import numpy as np                                                                   # noqa: E402
from matplotlib.lines import Line2D                                                  # noqa: E402
from matplotlib.transforms import blended_transform_factory                          # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
OUT.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
BASE, QBASE = str(OUT / "S5d_prior_slope"), str(QA / "S5d_prior_slope")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 70.0

pts = c.prior_points()
models = c.ordered({p[0] for p in pts})
W, H = width_mm, height_mm
L, R, B, T = 12.0, 2.0, 23.0, 3.0                    # mm; the bottom band holds two label rows and the key
STEP = 2.6                                            # x distance between data-set groups
DX = {m: (i - (len(models) - 1) / 2) * 0.08 for i, m in enumerate(models)}   # x dodge between models

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
ax.set_xlim(-0.45, 2 * STEP + 1.75); ax.set_ylim(15, 100)
lab_tf = blended_transform_factory(ax.transData, ax.transAxes)
ticks, ticklabels = [], []
for g, s in enumerate(c.SETS):
    g0 = g * STEP
    ax.plot([g0 - 0.38, g0 + 1.38], [c.CHANCE[s]] * 2, color=PAL["neutral_mid"], lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    if g:
        ax.axvline(g0 - (STEP - 1) / 2, color=PAL["neutral_light"], lw=0.6, zorder=0)
    for m, s2, o, b in pts:
        if s2 != s:
            continue
        st = c.style(m); x0, x1 = g0 + DX[m], g0 + 1 + DX[m]
        ax.plot([x0, x1], [o[0], b[0]], color=st["color"], lw=1.1, zorder=3)
        for x, ci, filled in ((x0, o, True), (x1, b, False)):
            ax.plot([x, x], [ci[1], ci[2]], color=st["color"], lw=0.6, alpha=0.55, solid_capstyle="butt", zorder=2)
            ax.plot([x], [ci[0]], ls="none", marker=st["marker"], ms=ms(st["marker"], 4.0),
                    mfc=st["color"] if filled else "white", mec=st["color"], mew=0.8, zorder=4)
    ticks += [g0, g0 + 1]; ticklabels += ["Original", "Blank"]
    ax.text(g0 + 0.5, -0.135, c.SET_LABEL[s], transform=lab_tf, ha="center", va="top", fontsize=TICK_PT,
            color=PAL["ink"])
ax.set_xticks(ticks, ticklabels)
ax.set_yticks(np.arange(20, 101, 20))
ax.tick_params(labelsize=TICK_PT)
ax.tick_params(axis="x", labelsize=LEGEND_PT, length=0, pad=3, labelcolor=PAL["neutral_dark"])
ax.set_ylabel("Single-agent accuracy (%)", fontsize=BASE_PT)

h_mod = [Line2D([], [], color=c.style(m)["color"], lw=1.1, marker=c.style(m)["marker"],
                ms=ms(c.style(m)["marker"], 4.0), mfc=c.style(m)["color"], mec=c.style(m)["color"],
                label=c.style(m)["label"]) for m in models]
h_ref = [Line2D([], [], color=PAL["neutral_mid"], lw=0.7, ls=(0, (1, 1.6)), label="Chance")]
fig.legend(handles=h_mod + h_ref, loc="lower center", bbox_to_anchor=(0.5, 0.6 / H), ncol=3, frameon=False,
           fontsize=LEGEND_PT, handlelength=1.6, handletextpad=0.4, columnspacing=0.9, borderaxespad=0)

require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE, "| points", len(pts))
