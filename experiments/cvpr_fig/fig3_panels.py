"""fig3_panels — candidate, full width (174.6 mm).

Claim: reading panels of nine readers from one model change accuracy little; the Qwen 3.8 Flash gain
comes mostly from its panel readers rather than from voting; discussion raises the error correlation
to 0.995-1.000 (Qwen 3.8 Flash, 0.82-0.91).
Rows: model within data set (250 items each).
a  accuracy of the independent panel of 9 and of the discussion panel of 9 minus the single
   reader (paired item bootstrap 95% CI)
b  the independent-panel difference split into mean panel reader minus single reader (Readers)
   and majority minus mean panel reader (Majority vote); same scale as a
c  error correlation phi of the 9 readers in round 0 and after discussion; top axis
   N_eff = 9 / (1 + 8 phi)
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
                      OUT, QA, load, signed, phi_from_neff, hci, add_panel_label)
from audit_panel_alignment import require_matplotlib_panel_alignment                       # noqa: E402

from matplotlib.lines import Line2D                                                         # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator                                     # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig3_panels")
QBASE = str(QA / "fig3_panels")

width_mm = 174.6            # final printed width: CVPR \textwidth (496.86 pt)
height_mm = 76

# ── data (read-only) ──
A = load("A")
get = lambda m, s: A[f"{m}|{s}"]
for m in MODELS:
    for s in SETS:
        assert get(m, s)["n_items"] == 250

GAP = 1.6
ypos = {(s, m): -(si * (len(MODELS) + GAP) + mi) for si, s in enumerate(SETS) for mi, m in enumerate(MODELS)}
y_head = {s: -(si * (len(MODELS) + GAP)) + 1.0 for si, s in enumerate(SETS)}
lo_y = min(ypos.values()) - 0.75
hi_y = max(y_head.values()) + 0.55
DY = 0.19

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
gs = fig.add_gridspec(1, 3, left=0.168, right=0.985, bottom=0.135, top=0.865, wspace=0.12)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
ax_c = fig.add_subplot(gs[0, 2], sharey=ax_a)
axes = (ax_a, ax_b, ax_c)

for (s, m), y in ypos.items():
    st, d = MODEL[m], get(m, s)
    p = d["paired"]
    hci(ax_a, y + DY, p["panel9_minus_single"], st["color"], st["marker"])
    hci(ax_a, y - DY, p["disc9_minus_single"], st["color"], st["marker"], filled=False)
    hci(ax_b, y + DY, p["reader_mean_minus_single"], st["color"], st["marker"])
    hci(ax_b, y - DY, p["panel9_minus_reader_mean"], st["color"], st["marker"], filled=False)
    hci(ax_c, y + DY, d["real"]["phi"], st["color"], st["marker"])
    hci(ax_c, y - DY, d["discussion"]["phi_last"], st["color"], st["marker"], filled=False)

# shared rows: model names as tick labels, data set names as group headers
ax_a.set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.035, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right",
              va="center", fontsize=BASE_PT, fontweight="bold")
for ax in axes:
    ax.set_ylim(lo_y, hi_y)
    ax.spines["left"].set_visible(False)
    for s in SETS[1:]:
        ax.axhline(y_head[s] + 0.8, color=PAL["neutral_light"], lw=0.6, zorder=0)
for ax in (ax_b, ax_c):
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", length=0)

# a and b: one scale, so the two parts in b add up to the filled points in a
top_line = y_head[SETS[0]] - 0.55
for ax in (ax_a, ax_b):
    ax.set_xlim(-8, 21)
    ax.xaxis.set_major_locator(FixedLocator([-5, 0, 5, 10, 15, 20]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: signed(v)))
    ax.plot([0, 0], [lo_y, top_line], color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax_a.set_xlabel("Panel − single reader (pp)")
ax_b.set_xlabel("Difference (pp)")

ax_c.set_xlim(0.35, 1.03)
ax_c.xaxis.set_major_locator(FixedLocator([0.4, 0.6, 0.8, 1.0]))
ax_c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_c.set_xlabel(r"Error correlation, $\phi$")
top = ax_c.secondary_xaxis("top", functions=(lambda p: p, lambda p: p))
NE = [2, 1.5, 1.2, 1]
top.xaxis.set_major_locator(FixedLocator(phi_from_neff(NE).tolist()))
top.xaxis.set_minor_locator(NullLocator())
top.set_xticklabels([f"{v:g}" for v in NE])
top.set_xlabel(r"$N_{\mathrm{eff}}$", labelpad=2)
top.tick_params(axis="x", length=2.5, width=0.8, labelsize=TICK_PT)


def key(ax, entries):
    """Filled/open key in the first data set's header row."""
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, color=PAL["ink"],
                              mfc=PAL["ink"] if filled else "white", mew=0.9, label=lab)
                       for lab, filled in entries],
              loc="center left", bbox_to_anchor=(0.0, y_head[SETS[0]]), bbox_transform=ax.get_yaxis_transform(),
              ncol=2, handlelength=0.9, handletextpad=0.35, columnspacing=1.1, borderaxespad=0.0, borderpad=0.0,
              fontsize=LEGEND_PT)


key(ax_a, [("Independent", True), ("Discussion", False)])
key(ax_b, [("Readers", True), ("Majority vote", False)])
key(ax_c, [("Round 0", True), ("Final", False)])

# letters at one physical offset from each plot area's top-left corner
for ax, letter in zip(axes, "abc"):
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
