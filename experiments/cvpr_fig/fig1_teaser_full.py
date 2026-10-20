"""fig1_teaser_full — candidate teaser, full width (174.6 mm); alternative to fig1_teaser_col.

Claim: the readers' mean accuracy and their error correlation phi predict every panel's majority
accuracy, and at the phi that one model's readers have, majority voting adds at most a few points.
Every 9-reader panel on yes/no items (VQA-RAD, SLAKE, SLAKE-target, SLAKE-pairs) in every
condition (108 panels).
a  observed majority accuracy against the beta-binomial prediction from mean reader accuracy and phi
b  majority minus mean reader accuracy against phi; curves, the beta-binomial gain at mean reader
   accuracy 60%, 75%, 90%; top axis N_eff = 9 / (1 + 8 phi)
Data: results/imaging_v2_A.json, key panel_arithmetic (experiments/imaging_v2_analysis.py A).
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

CFD = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CFD))
sys.path.insert(0, str(pathlib.Path.home() / ".claude" / "skills" / "nature-figure" / "scripts"))

from cf_style import (MM, PAL, MODELS, MODEL, ANNOT_PT, LEGEND_PT, OUT, QA,  # noqa: E402
                      load, ms, minus, signed, beta_binomial_gain, phi_from_neff, add_panel_label)
from audit_panel_alignment import require_matplotlib_panel_alignment       # noqa: E402

import numpy as np                                                          # noqa: E402
from matplotlib.lines import Line2D                                         # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator                     # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig1_teaser_full")
QBASE = str(QA / "fig1_teaser_full")

width_mm = 174.6            # final printed width: CVPR \textwidth (496.86 pt)
height_mm = 68

# ── data (read-only) ──
P = load("A")["panel_arithmetic"]
cells = P["cells"]
assert P["n_cells"] == len(cells) == 108
phi = np.array([c["phi"] for c in cells])
gain = np.array([c["majority"] - c["reader_mean"] for c in cells])

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
LEFT, RIGHT, BOTTOM, TOP, WSPACE = 0.072, 0.985, 0.142, 0.8, 0.2
plot_h = (TOP - BOTTOM) * height_mm
# width ratio that makes panel a's plot area square at the final size
ratio = 1.0
for _ in range(30):
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, ratio], left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP,
                          wspace=WSPACE)
    l_, r_ = gs.get_grid_positions(fig)[2:]
    a_w = (r_[0] - l_[0]) * width_mm
    ratio *= a_w / plot_h
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])


def dots(ax, x, y):
    for m in MODELS:
        st = MODEL[m]
        sel = np.array([c["model"] == m for c in cells])
        ax.plot(x[sel], y[sel], ls="none", marker=st["marker"], ms=ms(st["marker"], 4.0), mfc=st["color"],
                mec="white", mew=0.35, alpha=0.92, zorder=4)


# a: observed against predicted majority accuracy
pred = np.array([c["predicted"] for c in cells])
obs = np.array([c["majority"] for c in cells])
lo, hi = 35, 100
ax_a.plot([lo, hi], [lo, hi], color=PAL["neutral_mid"], lw=0.8, ls=(0, (3, 2)), zorder=1)
dots(ax_a, pred, obs)
ax_a.set_xlim(lo, hi)
ax_a.set_ylim(lo, hi)
for axis in (ax_a.xaxis, ax_a.yaxis):
    axis.set_major_locator(FixedLocator([40, 60, 80, 100]))
ax_a.set_xlabel("Predicted majority accuracy (%)")
ax_a.set_ylabel("Observed majority accuracy (%)")
d = P["observed_minus_predicted_mean"], P["observed_minus_predicted_sd"]
ax_a.text(0.97, 0.04, f"Observed − predicted\n{minus(f'{d[0]:+.2f}')} pp, SD {d[1]:.2f} pp",
          transform=ax_a.transAxes, ha="right", va="bottom", fontsize=ANNOT_PT, color=PAL["neutral_dark"])

# b: majority minus mean reader against phi
grid = np.linspace(0.002, 0.998, 400)
CURVES = [(0.60, (0, (4, 1.6)), "60%"), (0.75, "-", "75%"), (0.90, (0, (1, 1.4)), "90%")]
keys = []
for pbar, ls, lab in CURVES:
    ax_b.plot(grid, beta_binomial_gain(pbar, grid) * 100, color=PAL["neutral_mid"], lw=1.0, ls=ls, zorder=2)
    keys.append(Line2D([], [], color=PAL["neutral_mid"], lw=1.0, ls=ls, label=lab))
ax_b.axhline(0, color=PAL["neutral_light"], lw=0.7, zorder=0)
dots(ax_b, phi, gain)
ax_b.set_xlim(0, 1.0)
ax_b.set_ylim(-4, 22)
ax_b.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_b.yaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20]))
ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: signed(v)))
ax_b.set_xlabel(r"Error correlation, $\phi$")
ax_b.set_ylabel("Majority − mean reader (pp)")
top = ax_b.secondary_xaxis("top", functions=(lambda p: p, lambda p: p))
NE = [9, 4, 3, 2, 1.5, 1.2, 1]
top.xaxis.set_major_locator(FixedLocator(phi_from_neff(NE).tolist()))
top.xaxis.set_minor_locator(NullLocator())
top.set_xticklabels([f"{v:g}" for v in NE])
top.set_xlabel(r"$N_{\mathrm{eff}} = 9\,/\,(1 + 8\phi)$", labelpad=3)
top.tick_params(axis="x", length=2.5, width=0.8)

k = ax_b.legend(handles=keys, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=3, handlelength=2.2,
                handletextpad=0.4, columnspacing=0.9, borderaxespad=0.3, fontsize=LEGEND_PT,
                title="Prediction at mean reader accuracy", title_fontsize=LEGEND_PT)
k._legend_box.align = "left"
ax_b.add_artist(k)
leg = ax_b.legend(handles=[Line2D([], [], ls="none", marker=MODEL[m]["marker"], ms=ms(MODEL[m]["marker"], 4.0),
                                  mfc=MODEL[m]["color"], mec="white", mew=0.35, label=MODEL[m]["label"])
                           for m in MODELS],
                  loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.3, handletextpad=0.25,
                  labelspacing=0.3, fontsize=LEGEND_PT, title=f"Reading panels of 9 (n = {len(cells)})",
                  title_fontsize=LEGEND_PT)
leg._legend_box.align = "left"

LABEL_DY = 27     # above panel b top axis label
for ax, letter in zip((ax_a, ax_b), "ab"):
    add_panel_label(ax, letter, x_offset_pt=-30, y_offset_pt=LABEL_DY)

# ── render-time alignment gate, then exports ──
require_matplotlib_panel_alignment(
    fig,
    json_out=QBASE + ".alignment.json",
    overlay_svg=QBASE + ".alignment.svg",
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    require_panel_labels=True,
    strict=True,
)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print(f"panel a {a_w:.1f} x {plot_h:.1f} mm; ratio {ratio:.3f}")
