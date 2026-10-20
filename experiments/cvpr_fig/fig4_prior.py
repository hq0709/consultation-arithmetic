"""fig4_prior — candidate, full width (174.6 mm).

Claim: a gray image lowers accuracy on VQA-RAD and SLAKE and raises the error correlation of three
models; on MedXpertQA-MM the image changes the correctness of 13-27% of reader answers, and 64-76% of
consensus errors repeat the gray-image majority answer.
Rows: model within data set (250 items each), as in fig3_panels.
a  single-reader accuracy with the original image and with a uniform gray image of the same size
   (item bootstrap 95% CI); dashed, chance accuracy
b  error correlation phi of the independent panel of 9 with each image
c  every reader answer of the independent panel of 9, cross-tabulated by its correctness with the
   original image and with the gray image (same reader slot, same prompt); right of zero, correct
   with the original image
d  consensus errors (at least 7 of 9 readers wrong) whose majority answer is also the gray-image
   majority answer (item bootstrap 95% CI); n, number of consensus errors
Data: results/imaging_v2_A.json (experiments/imaging_v2_analysis.py A).
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
                      ANNOT_PT, OUT, QA, load, hci, add_panel_label)
from audit_panel_alignment import require_matplotlib_panel_alignment                       # noqa: E402

from matplotlib.lines import Line2D                                                         # noqa: E402
from matplotlib.patches import Patch                                                        # noqa: E402
from matplotlib.ticker import FixedLocator                                                  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig4_prior")
QBASE = str(QA / "fig4_prior")

width_mm = 174.6            # final printed width: CVPR \textwidth (496.86 pt)
height_mm = 84

# ── data (read-only) ──
A = load("A")
get = lambda m, s: A[f"{m}|{s}"]
CHANCE = {"vqarad": 50, "slake": 50, "mxqa_mm_rad": 20}

GAP = 1.6
ypos = {(s, m): -(si * (len(MODELS) + GAP) + mi) for si, s in enumerate(SETS) for mi, m in enumerate(MODELS)}
y_head = {s: -(si * (len(MODELS) + GAP)) + 1.0 for si, s in enumerate(SETS)}
lo_y = min(ypos.values()) - 0.75
hi_y = max(y_head.values()) + 0.55
DY = 0.19
H = 0.64
RIGHT_IMG, RIGHT_BOTH = PAL["pos_dark"], PAL["pos_soft"]
WRONG_IMG, WRONG_BOTH = PAL["neg_dark"], PAL["neg_soft"]

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
# one grid of nine equal columns: a, b and d span two, c spans three (it runs -100..100)
gs = fig.add_gridspec(1, 9, left=0.168, right=0.985, bottom=0.235, top=0.9, wspace=0.55)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4], sharey=ax_a)
ax_c = fig.add_subplot(gs[0, 4:7], sharey=ax_a)
ax_d = fig.add_subplot(gs[0, 7:9], sharey=ax_a)
axes = (ax_a, ax_b, ax_c, ax_d)

for (s, m), y in ypos.items():
    st, d = MODEL[m], get(m, s)
    hci(ax_a, y + DY, d["real"]["single"], st["color"], st["marker"])
    hci(ax_a, y - DY, d["blank"]["single"], st["color"], st["marker"], filled=False)
    hci(ax_b, y + DY, d["real"]["phi"], st["color"], st["marker"])
    hci(ax_b, y - DY, d["blank"]["phi"], st["color"], st["marker"], filled=False)
    pq = d["prior_decomposition"]
    rr, wr = pq["prior_right_stays_right"], pq["image_corrects_prior"]
    rw, ww = pq["image_misleads_prior"], pq["prior_wrong_stays_wrong"]
    ax_c.barh(y, wr, left=0, height=H, color=RIGHT_IMG, edgecolor="white", lw=0.5, zorder=3)
    ax_c.barh(y, rr, left=wr, height=H, color=RIGHT_BOTH, edgecolor="white", lw=0.5, zorder=3)
    ax_c.barh(y, -rw, left=0, height=H, color=WRONG_IMG, edgecolor="white", lw=0.5, zorder=3)
    ax_c.barh(y, -ww, left=-rw, height=H, color=WRONG_BOTH, edgecolor="white", lw=0.5, zorder=3)
    ax_c.text(wr + rr + 2, y, f"{wr + rr:.0f}", ha="left", va="center", fontsize=ANNOT_PT,
              color=PAL["neutral_dark"])
    ce = pq["consensus_errors_matching_blank_majority"]
    hci(ax_d, y, ce, st["color"], st["marker"])
    ax_d.text(1.0, y, f"{pq['consensus_errors']}", transform=ax_d.get_yaxis_transform(), ha="right",
              va="center", fontsize=ANNOT_PT, color=PAL["neutral_dark"])

# shared rows
ax_a.set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.035 * 46 / 30, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right",
              va="center", fontsize=BASE_PT, fontweight="bold")
for ax in axes:
    ax.set_ylim(lo_y, hi_y)
    ax.spines["left"].set_visible(False)
    for s in SETS[1:]:
        ax.axhline(y_head[s] + 0.8, color=PAL["neutral_light"], lw=0.6, zorder=0)
for ax in axes[1:]:
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", length=0)

# a: accuracy, chance per data set as a dashed segment over its rows
for s in SETS:
    ys = [ypos[(s, m)] for m in MODELS]
    ax_a.plot([CHANCE[s]] * 2, [min(ys) - 0.5, max(ys) + 0.5], color=PAL["neutral_mid"], lw=0.7,
              ls=(0, (3, 2)), zorder=1)
ax_a.set_xlim(15, 100)
ax_a.xaxis.set_major_locator(FixedLocator([20, 40, 60, 80, 100]))
ax_a.set_xlabel("Accuracy (%)")

ax_b.set_xlim(0.35, 1.03)
ax_b.xaxis.set_major_locator(FixedLocator([0.4, 0.6, 0.8, 1.0]))
ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_b.set_xlabel(r"Error correlation, $\phi$")

ax_c.axvline(0, color=PAL["ink"], lw=0.8, zorder=5)
ax_c.set_xlim(-104, 112)
ax_c.xaxis.set_major_locator(FixedLocator([-100, -50, 0, 50, 100]))
ax_c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}"))
ax_c.set_xlabel("Reader answers (%)")
ax_c.text(-52, y_head[SETS[0]], "Wrong", ha="center", va="center", fontsize=LEGEND_PT)
ax_c.text(52, y_head[SETS[0]], "Right", ha="center", va="center", fontsize=LEGEND_PT)
# the four answer classes of c, in one row under the whole figure (no panel has room for it)
fig.legend(handles=[Patch(color=WRONG_BOTH, label="Wrong with both images"),
                    Patch(color=WRONG_IMG, label="Wrong only with original"),
                    Patch(color=RIGHT_IMG, label="Right only with original"),
                    Patch(color=RIGHT_BOTH, label="Right with both images")],
           loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, handlelength=1.0, handleheight=0.75,
           handletextpad=0.35, columnspacing=1.4, borderaxespad=0.15, fontsize=LEGEND_PT)

ax_d.set_xlim(0, 112)
ax_d.xaxis.set_major_locator(FixedLocator([0, 50, 100]))
ax_d.axvline(50, color=PAL["neutral_light"], lw=0.6, zorder=0)
ax_d.set_xlabel("Same as gray-image\nmajority (%)")
ax_d.text(1.0, y_head[SETS[0]], "n", transform=ax_d.get_yaxis_transform(), ha="right", va="center",
          fontsize=ANNOT_PT, color=PAL["neutral_dark"])


def key(ax, entries):
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, color=PAL["ink"],
                              mfc=PAL["ink"] if filled else "white", mew=0.9, label=lab)
                       for lab, filled in entries],
              loc="center left", bbox_to_anchor=(0.0, y_head[SETS[0]]), bbox_transform=ax.get_yaxis_transform(),
              ncol=2, handlelength=0.9, handletextpad=0.35, columnspacing=1.1, borderaxespad=0.0, borderpad=0.0,
              fontsize=LEGEND_PT)


key(ax_a, [("Original", True), ("Gray", False)])
key(ax_b, [("Original", True), ("Gray", False)])

for ax, letter in zip(axes, "abcd"):
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
