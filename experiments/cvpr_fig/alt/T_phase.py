"""T_phase — candidate teaser, one column (83.34 mm).

Claim: majority voting gains only where error correlation is low; every observed system of 9 agents
sits in the high-correlation band where the predicted gain is near zero.
Colour: beta-binomial gain of a majority of 9 over its mean agent (pp), diverging at 0.
Data: results/imaging_v2_A.json, key panel_arithmetic.
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
from cf_style import (MM, PAL, MODELS, MODEL, LEGEND_PT, ANNOT_PT, load, ms, beta_binomial_gain,  # noqa: E402
                      phi_from_neff)
from audit_panel_alignment import require_matplotlib_panel_alignment                          # noqa: E402

import numpy as np                                                                               # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm                              # noqa: E402
from matplotlib.lines import Line2D                                                              # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator                                          # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "T_phase"), str(QA / "T_phase")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 82

P = load("A")["panel_arithmetic"]
cells = P["cells"]
assert len(cells) == 108

# diverging: red (voting loses) - white (no gain) - blue (voting gains)
CMAP = LinearSegmentedColormap.from_list("div", ["#B2453A", "#E6B3AC", "#F7F7F7", "#A9C4DE", "#5B8DBE", "#2F5D8C"])
phi = np.linspace(0.005, 0.995, 300)
pbar = np.linspace(0.30, 0.995, 300)
PH, PB = np.meshgrid(phi, pbar)
G = beta_binomial_gain(PB, PH) * 100

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
ax = fig.add_axes([0.15, 0.25, 0.62, 0.59])
cax = fig.add_axes([0.80, 0.25, 0.03, 0.59])
norm = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=20)
im = ax.pcolormesh(PH, PB * 100, G, cmap=CMAP, norm=norm, shading="auto", rasterized=True, zorder=0)
cs = ax.contour(PH, PB * 100, G, levels=[1, 5, 10, 15], colors=[PAL["neutral_dark"]], linewidths=0.6, zorder=1)
# labels placed by hand on the 70% accuracy line, where the contours are well separated; the 1 pp
# label sits lower, clear of the observed systems
from scipy.optimize import brentq                                                                # noqa: E402
PH1 = brentq(lambda v: beta_binomial_gain(0.55, v) * 100 - 1, 0.001, 0.99)
ax.clabel(cs, fmt=lambda v: f"{v:g} pp", fontsize=ANNOT_PT, inline=True, inline_spacing=3,
          manual=[(0.063, 70), (0.157, 70), (0.323, 70), (PH1, 55)])
for m in MODELS:
    st = MODEL[m]
    sel = [c for c in cells if c["model"] == m]
    ax.plot([c["phi"] for c in sel], [c["reader_mean"] for c in sel], ls="none", marker=st["marker"],
            ms=ms(st["marker"], 3.8), mfc=st["color"], mec="white", mew=0.45, zorder=3)
ax.set_xlim(0, 1)
ax.set_ylim(30, 100)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_yticks([40, 60, 80, 100])
ax.set_xlabel(r"Error correlation, $\phi$")
ax.set_ylabel("Mean agent accuracy (%)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(True)
top = ax.secondary_xaxis("top", functions=(lambda p: p, lambda p: p))
NE = [9, 3, 2, 1.5, 1.2, 1]
top.xaxis.set_major_locator(FixedLocator(phi_from_neff(NE).tolist()))
top.xaxis.set_minor_locator(NullLocator())
top.set_xticklabels([f"{v:g}" for v in NE])
top.set_xlabel(r"$N_{\mathrm{eff}}$", labelpad=2)
top.tick_params(axis="x", length=2.5, width=0.8)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Majority − mean agent (pp)", fontsize=8.5)
cb.set_ticks([-10, 0, 10, 20])
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: (f"{v:+.0f}" if v else "0").replace("-", "−")))
cb.outline.set_linewidth(0.6)
fig.legend(handles=[Line2D([], [], ls="none", marker=MODEL[m]["marker"], ms=ms(MODEL[m]["marker"], 3.8),
                           mfc=MODEL[m]["color"], mec="white", mew=0.45, label=MODEL[m]["label"]) for m in MODELS],
           loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, handlelength=0.9, handletextpad=0.3,
           columnspacing=1.2, labelspacing=0.25, borderaxespad=0.2, fontsize=LEGEND_PT)

# ── alignment gate (one data panel; the colorbar is excluded by the auditor), then exports ──
require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg", dpi=600)
fig.savefig(BASE + ".pdf", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
