"""fig5_distribution — candidate A: panels shift mean accuracy by at most 6.3 % and overlap the single doctor.

Nature-figure skill, Python backend. Same values, box statistics and estimator as
fig_rest.py::fig_box, rebuilt to the skill's standard: horizontal boxes so the full
architecture names label the rows, every configuration drawn, the mean marked on each
box, and the relative change in mean accuracy vs the single doctor beside each row.
"""
import sys
import pathlib

sys.dont_write_bytecode = True     # read-only on the repository: no __pycache__ writes on import

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
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
})

NF = pathlib.Path(__file__).resolve().parent                     # experiments/nature_fig
REPO = NF.parents[1]
for p in (NF, REPO):
    sys.path.insert(0, str(p))

from nf_style import (MM, PAL, ARCH, MAS_ORDER, BENCH_ORDER, BENCH_LABEL, BASE_PT,  # noqa: E402
                      TICK_PT, ANNOT_PT, add_panel_label, add_panel_title, minus)
from audit_panel_alignment import require_matplotlib_panel_alignment               # noqa: E402
from rest_helpers import swarm                                                     # noqa: E402

import numpy as np                                                                 # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter                         # noqa: E402
from matplotlib.patches import PathPatch                                           # noqa: E402
from experiments.fig_rest import build, acc                                        # noqa: E402  (read-only)
from experiments.vizstyle import MODEL_ORDER                                       # noqa: E402

OUT = REPO / "paper" / "figures"
QA = NF / "qa"
QA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig5_distribution")
QBASE = str(QA / "fig5_distribution")

width_mm = 160          # final printed width: \textwidth
height_mm = 60

# ── data: identical inclusion and estimator to fig_rest.fig_box (read-only) ──
cells = build()
ROWS = ["cot"] + MAS_ORDER
vals = {}
for b in BENCH_ORDER:
    vals[(b, "cot")] = [acc(cells[(m, b, "cot", 1)]) for m in MODEL_ORDER if (m, b, "cot", 1) in cells]
    for a in MAS_ORDER:
        vals[(b, a)] = [acc(v) for k, v in cells.items() if k[1] == b and k[2] == a]
counts = {k: len(v) for k, v in vals.items()}
assert all(counts[(b, "cot")] == 8 for b in BENCH_ORDER)
assert all(counts[(b, a)] == 40 for b in BENCH_ORDER for a in MAS_ORDER)
rel = {}                                   # relative change of the mean vs the single doctor, %
for b in BENCH_ORDER:
    base = np.mean(vals[(b, "cot")])
    for a in MAS_ORDER:
        rel[(b, a)] = (np.mean(vals[(b, a)]) - base) / base * 100
assert [round(rel[("medxpertqa", a)], 1) for a in ("centralized", "discussion", "independent", "tiered")] == [6.3, 6.0, 2.5, 1.0]

# ── encoding ──
Y = {a: len(ROWS) - 1 - i for i, a in enumerate(ROWS)}        # single doctor on top
BOX_H = 0.52
PT_MS, PT_MEW = 2.7, 0.3
REF_C, REF_LW, REF_LS = PAL["neutral_mid"], 0.7, (0, (3, 2))
# The single doctor is the reference row: drawn in a neutral derived from PAL (not ARCH's ink)
# so the baseline stays quieter than the panels it frames; the row label names it.
SD_C = PAL["neutral_dark"]
XTICKS = {"medxpertqa": [20, 40, 60], "medagentsbench": [20, 40, 60], "medqa": [60, 70, 80, 90]}
XLIM = {"medxpertqa": (10, 67), "medagentsbench": (10, 67), "medqa": (49, 99)}


def draw_box(ax, x, yc, color, sd=False):
    """Median/IQR box with 1.5 x IQR whiskers (matplotlib's default, as in the current figure).

    Layering: box tint (z 1) < points (z 2) < box outline, whiskers, median (z 3) < mean (z 5),
    so the overlaid configurations never hide the summary they are summarised by.
    """
    edge = SD_C if sd else color
    bp = ax.boxplot([x], positions=[yc], orientation="horizontal", widths=BOX_H, whis=1.5,
                    showfliers=False, patch_artist=True, manage_ticks=False,
                    medianprops=dict(color=PAL["ink"], lw=1.1, solid_capstyle="butt"),
                    whiskerprops=dict(color=edge, lw=0.8), capprops=dict(color=edge, lw=0.8),
                    boxprops=dict(lw=0.8))
    box = bp["boxes"][0]
    box.set_facecolor("white" if sd else mpl.colors.to_rgba(color, 0.20))
    box.set_edgecolor("none")
    box.set_zorder(1)
    ax.add_patch(PathPatch(box.get_path(), facecolor="none", edgecolor=edge, lw=0.8, zorder=3))
    for art in bp["whiskers"] + bp["caps"] + bp["medians"]:
        art.set_zorder(3)


fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), layout="constrained")
fig.get_layout_engine().set(w_pad=1.5 / 72, h_pad=1.5 / 72, wspace=0.06, hspace=0.0)
gs = fig.add_gridspec(1, len(BENCH_ORDER))
axes = [fig.add_subplot(gs[0, i]) for i in range(len(BENCH_ORDER))]
for i, (ax, b) in enumerate(zip(axes, BENCH_ORDER)):
    base = float(np.mean(vals[(b, "cot")]))
    ax.axvline(base, color=REF_C, lw=REF_LW, ls=REF_LS, zorder=1)
    for a in ROWS:
        draw_box(ax, vals[(b, a)], Y[a], ARCH[a]["color"], sd=(a == "cot"))
    ax.set_xlim(*XLIM[b])
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS[b]))
    ax.yaxis.set_major_locator(FixedLocator([Y[a] for a in ROWS]))
    if i == 0:
        ax.yaxis.set_major_formatter(FixedFormatter([ARCH[a]["label"] for a in ROWS]))
        ax.tick_params(axis="y", length=0, pad=3, labelsize=TICK_PT)
    else:
        ax.tick_params(axis="y", length=0, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Accuracy (%)", fontsize=BASE_PT)
    add_panel_label(ax, "abc"[i])
    add_panel_title(ax, BENCH_LABEL[b])
    # relative change of the mean, in a column just right of the plot area
    for a in MAS_ORDER:
        v = rel[(b, a)]
        ax.text(1.0, Y[a], " " + minus(f"{v:+.1f}%"), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=ANNOT_PT, fontweight="bold",
                color=PAL["delta_up"] if v >= 0 else PAL["delta_down"], clip_on=False)

# every configuration as a point, spread as a deterministic beeswarm (layout is final here)
fig.canvas.draw()
k = fig.dpi / 72
for ax, b in zip(axes, BENCH_ORDER):
    to_px = ax.transData
    row_px = abs(to_px.transform((0, 1))[1] - to_px.transform((0, 0))[1])
    for a in ROWS:
        x = np.asarray(vals[(b, a)])
        xpx = to_px.transform(np.column_stack([x, np.zeros_like(x)]))[:, 0]
        off = swarm(xpx, diameter_px=(PT_MS + PT_MEW) * k, max_half_px=0.36 * row_px)
        y = Y[a] + off / row_px
        if a == "cot":
            ax.plot(x, y, ls="none", marker="o", ms=PT_MS + 0.3, mfc="white", mec=SD_C,
                    mew=0.6, zorder=2)
        else:
            ax.plot(x, y, ls="none", marker="o", ms=PT_MS, mfc=ARCH[a]["color"], mec="white",
                    mew=PT_MEW, alpha=0.9, zorder=2)
        ax.plot([np.mean(x)], [Y[a]], ls="none", marker="D", ms=3.6, mfc="white",
                mec=PAL["ink"], mew=0.8, zorder=5)

# ── render-time alignment gate (multi-panel figures), then exports ──
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
fig.savefig(QBASE + ".tiff", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
plt.close(fig)
print("counts", {f"{b}/{a}": counts[(b, a)] for b in BENCH_ORDER for a in ROWS})
print("relative change %", {f"{b}/{a}": round(rel[(b, a)], 1) for b in BENCH_ORDER for a in MAS_ORDER})
