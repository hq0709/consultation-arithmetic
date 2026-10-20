"""fig1_teaser_col — candidate teaser, one column (83.34 mm).

Claim: majority voting over nine readers of one VLM adds at most a few points over its own
readers, because their errors are correlated; the beta-binomial curve fixed by the readers'
mean accuracy and their error correlation phi sets the gain.
Every 9-reader panel on yes/no items (VQA-RAD, SLAKE, SLAKE-target, SLAKE-pairs) in every
condition (108 panels). Curves: the beta-binomial gain at mean reader accuracy 60%, 75%, 90%.
Top axis: N_eff = 9 / (1 + 8 phi).
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

from cf_style import (MM, COL_W_MM, PAL, MODELS, MODEL, ANNOT_PT, LEGEND_PT, OUT, QA,  # noqa: E402
                      load, ms, signed, beta_binomial_gain, neff, phi_from_neff)
from audit_panel_alignment import require_matplotlib_panel_alignment                 # noqa: E402

import numpy as np                                                                    # noqa: E402
from matplotlib.lines import Line2D                                                   # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator                               # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig1_teaser_col")
QBASE = str(QA / "fig1_teaser_col")

width_mm = 83.34             # final printed width: CVPR \columnwidth (237.14 pt)
height_mm = 66

# ── data (read-only) ──
P = load("A")["panel_arithmetic"]
cells = P["cells"]
assert P["n_cells"] == len(cells) == 108
phi = np.array([c["phi"] for c in cells])
gain = np.array([c["majority"] - c["reader_mean"] for c in cells])

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
ax = fig.add_axes([0.155, 0.155, 0.815, 0.68])

# beta-binomial gain curves at fixed mean reader accuracy; the 60% and 90% curves cross near
# phi = 0.08, so they get a line-style key instead of direct labels
grid = np.linspace(0.002, 0.998, 400)
CURVES = [(0.60, (0, (4, 1.6)), "60%"), (0.75, "-", "75%"), (0.90, (0, (1, 1.4)), "90%")]
curve_keys = []
for pbar, ls, lab in CURVES:
    g = beta_binomial_gain(pbar, grid) * 100
    ax.plot(grid, g, color=PAL["neutral_mid"], lw=1.0, ls=ls, zorder=2)
    curve_keys.append(Line2D([], [], color=PAL["neutral_mid"], lw=1.0, ls=ls, label=lab))
ax.axhline(0, color=PAL["neutral_light"], lw=0.7, zorder=0)

# every 9-reader panel; model order fixed, Qwen last so its few panels stay visible
for m in MODELS:
    st = MODEL[m]
    sel = np.array([c["model"] == m for c in cells])
    ax.plot(phi[sel], gain[sel], ls="none", marker=st["marker"], ms=ms(st["marker"], 4.0),
            mfc=st["color"], mec="white", mew=0.35, alpha=0.92, zorder=4)

ax.set_xlim(0, 1.0)
ax.set_ylim(-4, 24.5)
ax.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20]))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: signed(v)))
ax.set_xlabel(r"Error correlation, $\phi$")
ax.set_ylabel("Majority − mean reader (pp)")

top = ax.secondary_xaxis("top", functions=(lambda p: p, lambda p: p))
NE = [9, 4, 2, 1.5, 1.2, 1]
top.xaxis.set_major_locator(FixedLocator(phi_from_neff(NE).tolist()))
top.xaxis.set_minor_locator(NullLocator())
top.set_xticklabels([f"{v:g}" for v in NE])
top.set_xlabel(r"$N_{\mathrm{eff}} = 9\,/\,(1 + 8\phi)$", labelpad=3)
top.spines["top"].set_visible(True)
top.tick_params(axis="x", length=2.5, width=0.8)

handles = [Line2D([], [], ls="none", marker=MODEL[m]["marker"], ms=ms(MODEL[m]["marker"], 4.0),
                  mfc=MODEL[m]["color"], mec="white", mew=0.35, label=MODEL[m]["label"]) for m in MODELS]
key = ax.legend(handles=curve_keys, loc="upper left", bbox_to_anchor=(0.075, 1.0), borderaxespad=0.3,
                ncol=3, handlelength=2.2, handletextpad=0.4, columnspacing=0.9, fontsize=LEGEND_PT,
                title="Prediction at mean reader accuracy", title_fontsize=LEGEND_PT)
key._legend_box.align = "left"
ax.add_artist(key)
leg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.79), borderaxespad=0.3,
                handletextpad=0.25, labelspacing=0.3, fontsize=LEGEND_PT,
                title=f"Reading panels of 9 (n = {len(cells)})", title_fontsize=LEGEND_PT)
leg._legend_box.align = "left"

# ── render-time alignment gate, then exports ──
require_matplotlib_panel_alignment(
    fig,
    json_out=QBASE + ".alignment.json",
    overlay_svg=QBASE + ".alignment.svg",
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    require_panel_labels=False,
    strict=True,
)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print(f"phi {phi.min():.3f}-{phi.max():.3f}  gain {gain.min():.2f}-{gain.max():.2f} pp  "
      f"N_eff {neff(phi.max()):.2f}-{neff(phi.min()):.2f}")
