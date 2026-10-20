"""fig5_region — candidate, one column (83.34 mm).

Claim: models differ widely in whether their answers follow the queried region (3.2 to 32.9 pp for
API models) and the image (both answers of a pair correct, 31.5% to 74.0%); among API models the two
measures have the same order.
Rows: API models (readers of the independent panel of 9; reader-level rates averaged within
item) and open-weight models (one forward pass per item).
a  SLAKE-target (250 items): share of answers that change when a mid-gray box covers the queried
   organ or lesion minus the share that change when an equal-size box covers another part of the
   body (paired item bootstrap 95% CI)
b  SLAKE-pairs (200 pairs, the same question on two images with opposite reference answers): both
   answers correct (single reader for API models; complete pairs resampled). A reader that ignores
   the image scores 0.
Data: results/imaging_v2_A.json, results/imaging_v2_D.json (experiments/imaging_v2_analysis.py A, D).
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

from cf_style import (MM, PAL, API3, MODEL, OW, OW_LABEL, OW_STYLE, BASE_PT, TICK_PT,  # noqa: E402
                      OUT, QA, load, signed, hci, add_panel_label)
from audit_panel_alignment import require_matplotlib_panel_alignment                  # noqa: E402

from matplotlib.ticker import FixedLocator                                             # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig5_region")
QBASE = str(QA / "fig5_region")

width_mm = 83.34            # final printed width: CVPR \columnwidth (237.14 pt)
height_mm = 60

# ── data (read-only) ──
A, D = load("A"), load("D")


def rates(m):
    """-> (region dependence, both answers correct), each [est, lo, hi]."""
    if m in API3:
        assert A[f"{m}|slake_pairs"]["n_pairs"] == 200
        return A[f"{m}|slake_target"]["panel9_region_dependence"], A[f"{m}|slake_pairs"]["single_both_correct"]
    assert D[f"{m}|slake_pairs"]["n_pairs"] == 200
    return D[f"{m}|slake_target"]["region_dependence_flip"], D[f"{m}|slake_pairs"]["both_correct"]


GROUPS = [("Proprietary models", API3), ("Open-weight models", OW)]
GAP = 1.5
ypos, y_head, start = {}, {}, 0.0
for g, ms_ in GROUPS:
    y_head[g] = start + 1.0
    for i, m in enumerate(ms_):
        ypos[m] = start - i
    start -= len(ms_) + GAP
lo_y = min(ypos.values()) - 0.7
hi_y = max(y_head.values()) + 0.5
label = {m: MODEL[m]["label"] if m in MODEL else OW_LABEL[m] for m in ypos}

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
gs = fig.add_gridspec(1, 2, left=0.385, right=0.958, bottom=0.25, top=0.925, wspace=0.24)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
axes = (ax_a, ax_b)

for m, y in ypos.items():
    st = MODEL.get(m) or OW_STYLE
    rd, bc = rates(m)
    hci(ax_a, y, rd, st["color"], st["marker"])
    hci(ax_b, y, bc, st["color"], st["marker"])

ax_a.set_yticks(list(ypos.values()), [label[m] for m in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
# group headers start at the figure's left edge, on rows of their own
for g, _ in GROUPS:
    ax_a.text(0.008, y_head[g], g, transform=mpl.transforms.blended_transform_factory(fig.transFigure, ax_a.transData),
              ha="left", va="center", fontsize=BASE_PT, fontweight="bold")
for ax in axes:
    ax.set_ylim(lo_y, hi_y)
    ax.spines["left"].set_visible(False)
    ax.axhline(y_head["Open-weight models"] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)
plt.setp(ax_b.get_yticklabels(), visible=False)
ax_b.tick_params(axis="y", length=0)

ax_a.plot([0, 0], [lo_y, y_head["Proprietary models"] - 0.5], color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax_a.set_xlim(-5, 46)
ax_a.xaxis.set_major_locator(FixedLocator([0, 20, 40]))
ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: signed(v)))
ax_a.set_xlabel("Occlusion\nsensitivity (pp)")

ax_b.set_xlim(0, 100)
ax_b.xaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax_b.set_xlabel("Pair accuracy (%)")

for ax, letter in zip(axes, "ab"):
    add_panel_label(ax, letter, x_offset_pt=-2, y_offset_pt=2)

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
print("written", BASE)
