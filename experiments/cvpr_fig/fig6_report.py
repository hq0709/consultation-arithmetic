"""fig6_report — candidate, full width (174.6 mm).

Claim: reports from Gemini 3.7 Flash raise every other reasoner above its own image reading, and one
shared report raises the readers' error correlation to 0.80-0.99.
Rows: the reasoner model (the reader that chooses the answer) within each data set (250 items).
a  mean reader accuracy; columns: the perceiver that wrote the reports (each of the 9 readers reads
   its own report, without the image), and the reasoner reading the image itself. Shading runs
   within each data set; bold, the reasoner reads its own model's reports.
b  error correlation phi of the 9 readers: reading the image, each reading its own report from the
   same model, and all reading one shared report (item bootstrap 95% CI)
Data: results/imaging_v2_B.json (experiments/imaging_v2_analysis.py B).
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

from cf_style import (MM, PAL, API3, MODEL, SETS, SET_LABEL, BASE_PT, TICK_PT, LEGEND_PT,  # noqa: E402
                      ANNOT_PT, OUT, QA, load, ms, add_panel_label)
from audit_panel_alignment import require_matplotlib_panel_alignment                      # noqa: E402

from matplotlib.colors import LinearSegmentedColormap                                      # noqa: E402
from matplotlib.lines import Line2D                                                        # noqa: E402
from matplotlib.ticker import FixedLocator                                                 # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig6_report")
QBASE = str(QA / "fig6_report")

width_mm = 174.6            # final printed width: CVPR \textwidth (496.86 pt)
height_mm = 70

# ── data (read-only) ──
B = load("B")
# sequential blue (dataviz reference ramp, steps 100-600), as in the radiology figure
CMAP = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#184f95"])

GAP = 1.6
ypos = {(s, r): -(si * (len(API3) + GAP) + ri) for si, s in enumerate(SETS) for ri, r in enumerate(API3)}
y_head = {s: -(si * (len(API3) + GAP)) + 1.0 for si, s in enumerate(SETS)}
lo_y = min(ypos.values()) - 0.7
hi_y = max(y_head.values()) + 0.5
COLS = API3 + ["image"]
XC = {c: i + (0.3 if c == "image" else 0) for i, c in enumerate(COLS)}      # gap before the image column
COL_LABEL = {"gpt-5-mini": "GPT-5\nmini", "gemini-3.7-flash": "Gemini\n3.7 Flash",
             "claude-haiku-4-5-20251001": "Claude\nHaiku 4.5", "image": "Image"}
SERIES = [("direct", "s", True, 0.25, "Image"), ("own", "o", True, 0.0, "Own report"),
          ("shared", "o", False, -0.25, "Shared report")]

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], left=0.168, right=0.985, bottom=0.165, top=0.78,
                      wspace=0.07)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

for s in SETS:
    vals = {}
    for r in API3:
        row = B[s][r]
        for c in COLS:
            vals[(r, c)] = (row["direct"] if c == "image" else row[f"{c}|own"])["reader_acc"][0]
    lo, hi = min(vals.values()), max(vals.values())
    for r in API3:
        y = ypos[(s, r)]
        for c in COLS:
            v = vals[(r, c)]
            t = (v - lo) / (hi - lo)
            ax_a.add_patch(plt.Rectangle((XC[c] - 0.47, y - 0.44), 0.94, 0.88, facecolor=CMAP(0.1 + 0.8 * t),
                                         lw=0, zorder=2))
            ax_a.text(XC[c], y, f"{v:.1f}", ha="center", va="center", fontsize=ANNOT_PT, zorder=3,
                      color="white" if t > 0.55 else PAL["ink"], fontweight="bold" if c == r else "normal")
        st = MODEL[r]
        for key, marker, filled, dy, _ in SERIES:
            e = B[s][r]["direct"] if key == "direct" else B[s][r][f"{r}|{key}"]
            est, l_, h_ = e["phi"]
            ax_b.plot([l_, h_], [y + dy, y + dy], color=st["color"], lw=0.9, solid_capstyle="butt", zorder=3)
            ax_b.plot([est], [y + dy], marker=marker, ms=ms(marker, 3.6), color=st["color"],
                      mfc=st["color"] if filled else "white", mec=st["color"], mew=0.9, lw=0, zorder=4)

# a: heat map grid; perceivers on top
ax_a.set_xlim(-0.55, XC["image"] + 0.55)
ax_a.set_xticks([XC[c] for c in COLS], [COL_LABEL[c] for c in COLS], fontsize=TICK_PT)
ax_a.xaxis.tick_top()
ax_a.tick_params(axis="x", length=0, pad=2)
ax_a.text(1.0, 1.215, "Reports by perception module", transform=ax_a.get_xaxis_transform(), ha="center", va="bottom",
          fontsize=LEGEND_PT, color=PAL["neutral_dark"])
ax_a.plot([-0.45, 2.45], [1.2, 1.2], transform=ax_a.get_xaxis_transform(), color=PAL["neutral_mid"], lw=0.6,
          clip_on=False)
ax_a.set_xlabel("Agent accuracy (%)")
ax_a.set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.035, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=BASE_PT, fontweight="bold")
for sp in ("left", "bottom", "top", "right"):
    ax_a.spines[sp].set_visible(False)

# b: error correlation
ax_b.set_ylim(lo_y, hi_y)
ax_b.spines["left"].set_visible(False)
plt.setp(ax_b.get_yticklabels(), visible=False)
ax_b.tick_params(axis="y", length=0)
for s in SETS[1:]:
    ax_b.axhline(y_head[s] + 0.8, color=PAL["neutral_light"], lw=0.6, zorder=0)
ax_b.set_xlim(0.55, 1.01)
ax_b.xaxis.set_major_locator(FixedLocator([0.6, 0.7, 0.8, 0.9, 1.0]))
ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_b.set_xlabel(r"Error correlation, $\phi$")
ax_b.legend(handles=[Line2D([], [], ls="none", marker=mk, ms=ms(mk, 3.6), color=PAL["ink"],
                            mfc=PAL["ink"] if f else "white", mew=0.9, label=lab)
                     for _, mk, f, _, lab in SERIES],
            loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=0.9, handletextpad=0.35,
            columnspacing=1.2, borderaxespad=0.25, fontsize=LEGEND_PT)

LABEL_DY = 27     # letters level with the "Reports by perception module" header of a
for ax, letter in zip((ax_a, ax_b), "ab"):
    add_panel_label(ax, letter, x_offset_pt=-2, y_offset_pt=LABEL_DY)

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
