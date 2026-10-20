"""fig7_contrasts — candidate, one column (83.34 mm), one panel.

Claim: the error correlation of two readers falls most when they do not share the model (-0.47 to
-0.49) or the perceiver (-0.26 to -0.31); roles, renderings and reference cases lower it by 0.03 to 0.15.
Rows: the component the two readers do not share, matched within model (docs/IMAGING_PROTOCOL.md
section 12): role (specialty vs generic panel), rendering (different renderings vs the original
image), reference cases (private vs shared), report (own vs shared report), perceiver (reports from
different perceivers vs separate reports from one perceiver), reasoner (different reasoner models
reading one shared report vs one reasoner), model (different models vs one model, reading images).
Black: mean over models (or model pairs) per data set, item bootstrap 95% CI; coloured dots: each
model; grey dots: each model pair. Reference cases were run on VQA-RAD and SLAKE only.
Data: results/imaging_v2_C.json (experiments/imaging_v2_analysis.py C).
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

from cf_style import (MM, PAL, MODELS, MODEL, SETS, SET_LABEL, TICK_PT, LEGEND_PT,  # noqa: E402
                      OUT, QA, load, ms, signed)
from audit_panel_alignment import require_matplotlib_panel_alignment              # noqa: E402

from matplotlib.lines import Line2D                                                # noqa: E402
from matplotlib.ticker import FixedLocator                                         # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig7_contrasts")
QBASE = str(QA / "fig7_contrasts")

width_mm = 83.34            # final printed width: CVPR \columnwidth (237.14 pt)
height_mm = 78

# ── data (read-only) ──
C = load("C")
ROWS = [("role", "Role"), ("rendering", "Rendering"), ("reference cases", "Reference cases"),
        ("report", "Report"), ("perceiver", "Perceiver"), ("reasoner", "Reasoner"), ("model", "Model")]
SET_MARK = {"vqarad": "o", "slake": "s", "mxqa_mm_rad": "^"}
SET_DY = {"vqarad": 0.27, "slake": 0.0, "mxqa_mm_rad": -0.27}

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
ax = fig.add_axes([0.3, 0.13, 0.675, 0.68])

for ri, (key, _) in enumerate(ROWS):
    y0 = -ri
    for s in SETS:
        d = (C[s]["contrasts"]).get(key)
        if not d:
            continue
        y = y0 + SET_DY[s]
        for k, v in sorted(d.items()):
            if k == "mean":
                continue
            st = MODEL.get(k)
            ax.plot([v[0]], [y], ls="none", marker="o", ms=2.7,
                    color=st["color"] if st else PAL["neutral_mid"], alpha=0.9, mew=0, zorder=3)
        est, lo, hi = d["mean"]
        ax.plot([lo, hi], [y, y], color=PAL["ink"], lw=1.1, solid_capstyle="butt", zorder=5)
        ax.plot([est], [y], ls="none", marker=SET_MARK[s], ms=ms(SET_MARK[s], 4.2), mfc=PAL["ink"],
                mec="white", mew=0.5, zorder=6)
    if ri:
        ax.axhline(y0 + 0.5, color=PAL["neutral_faint"], lw=0.6, zorder=0)

ax.axvline(0, color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.set_xlim(-0.82, 0.1)
ax.xaxis.set_major_locator(FixedLocator([-0.8, -0.6, -0.4, -0.2, 0]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: signed(v, 1)))
ax.set_xlabel(r"Change in error correlation, $\Delta\phi$")
ax.set_yticks([-i for i in range(len(ROWS))], [lab for _, lab in ROWS], fontsize=TICK_PT)
ax.set_ylim(-len(ROWS) + 0.5, 0.5)
ax.tick_params(axis="y", length=0, pad=4)
ax.spines["left"].set_visible(False)

# one key in three columns: data sets (black mean and CI), then models and model pairs (dots)
handles = [Line2D([], [], ls="-", lw=1.1, color=PAL["ink"], marker=SET_MARK[s], ms=ms(SET_MARK[s], 4.2),
                  mfc=PAL["ink"], mec="white", mew=0.5, label=SET_LABEL[s]) for s in SETS]
handles += [Line2D([], [], ls="none", marker="o", ms=3.0, mew=0, color=MODEL[m]["color"], label=MODEL[m]["label"])
            for m in MODELS]
handles += [Line2D([], [], ls="none", marker="o", ms=3.0, mew=0, color=PAL["neutral_mid"], label="Model pair")]
leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.995), ncol=3, handlelength=1.35,
           handletextpad=0.3, columnspacing=0.55, labelspacing=0.3, borderaxespad=0.0, borderpad=0.2,
                 fontsize=LEGEND_PT)
fig.canvas.draw()
_bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
assert _bb.x1 <= 0.995, f"legend runs off the figure: right edge {_bb.x1:.3f}"

# ── render-time alignment gate (single panel: records the plot-area geometry), then exports ──
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
print("written", BASE)
