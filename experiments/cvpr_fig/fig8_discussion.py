"""fig8_discussion — candidate, one column (83.34 mm).

Claim: discussion overturns 41.5-100% of correct round-0 minority answers and 1.0-6.5% of correct
majority answers; on MedXpertQA-MM it overturns correct answers that differ from the reader's
gray-image answer more often than those that match it.
Readers of the discussion panel of 9 who were correct in round 0; share wrong after the final round
(ratios of reader counts, item bootstrap 95% CI). Rows: model within data set (250 items each).
a  split by whether the reader's round-0 answer was the round-0 majority answer
b  split by whether the round-0 answer differed from the same reader's answer on the gray image
Data: results/imaging_v2_A.json, key discussion.conformity (experiments/imaging_v2_analysis.py A).
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

from cf_style import (MM, PAL, MODELS, MODEL, SETS, SET_LABEL, BASE_PT, TICK_PT, LEGEND_PT,  # noqa: E402
                      OUT, QA, load, hci, add_panel_label)
from audit_panel_alignment import require_matplotlib_panel_alignment                       # noqa: E402

from matplotlib.lines import Line2D                                                         # noqa: E402
from matplotlib.ticker import FixedLocator                                                  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig8_discussion")
QBASE = str(QA / "fig8_discussion")

width_mm = 83.34            # final printed width: CVPR \columnwidth (237.14 pt)
height_mm = 80

# ── data (read-only) ──
A = load("A")
conf = lambda m, s: A[f"{m}|{s}"]["discussion"]["conformity"]
for m in MODELS:
    for s in SETS:
        assert conf(m, s)["n_items"] == 250

GAP = 1.5
ypos = {(s, m): -(si * (len(MODELS) + GAP) + mi) for si, s in enumerate(SETS) for mi, m in enumerate(MODELS)}
y_head = {s: -(si * (len(MODELS) + GAP)) + 1.0 for si, s in enumerate(SETS)}
lo_y = min(ypos.values()) - 0.7
hi_y = max(y_head.values()) + 0.5
DY = 0.19

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
gs = fig.add_gridspec(1, 2, left=0.335, right=0.965, bottom=0.163, top=0.865, wspace=0.2)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
axes = (ax_a, ax_b)

for (s, m), y in ypos.items():
    st, c = MODEL[m], conf(m, s)
    hci(ax_a, y + DY, c["right_to_wrong_minority"], st["color"], st["marker"])
    hci(ax_a, y - DY, c["right_to_wrong_majority"], st["color"], st["marker"], filled=False)
    hci(ax_b, y + DY, c["right_to_wrong_image_dependent"], st["color"], st["marker"])
    hci(ax_b, y - DY, c["right_to_wrong_prior_consistent"], st["color"], st["marker"], filled=False)

ax_a.set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.06, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=BASE_PT, fontweight="bold")
for ax in axes:
    ax.set_ylim(lo_y, hi_y)
    ax.spines["left"].set_visible(False)
    for s in SETS[1:]:
        ax.axhline(y_head[s] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)
plt.setp(ax_b.get_yticklabels(), visible=False)
ax_b.tick_params(axis="y", length=0)

ax_a.set_xlim(-4, 104)                       # room for markers at 0 and 100
ax_a.xaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax_b.set_xlim(-1.6, 41.6)
ax_b.xaxis.set_major_locator(FixedLocator([0, 20, 40]))
fig.text((ax_a.get_position().x0 + ax_b.get_position().x1) / 2, 0.012,
         "Conformity (%)", ha="center", va="bottom", fontsize=BASE_PT)

# keys above each panel, to the right of its letter
for ax, entries in ((ax_a, [("Minority", True), ("Majority", False)]),
                    (ax_b, [("Differs from blank", True), ("Same as blank", False)])):
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, color=PAL["ink"],
                              mfc=PAL["ink"] if filled else "white", mew=0.9, label=lab)
                       for lab, filled in entries],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, handlelength=0.9, handletextpad=0.3,
              labelspacing=0.2, borderaxespad=0.15, borderpad=0.0, fontsize=LEGEND_PT)
for ax, letter in zip(axes, "ab"):
    add_panel_label(ax, letter, x_offset_pt=-11, y_offset_pt=12)

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
