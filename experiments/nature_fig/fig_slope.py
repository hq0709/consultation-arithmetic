"""fig8_slope — candidate A: phi ~ 0.76 leaves a nine-member panel with 1.27 effective opinions,
and discussion drives phi to 0.99.

Nature-figure skill, Python backend. Structural adaptation of experiments/fig_forest.py::slope:
same JSON and estimators; all 340 configurations shown (the current x-limit hid three).
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
    # math glyphs (phi, N_eff) in the same face as the text
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans",
    "mathtext.cal": "Liberation Sans",
    "axes.labelcolor": "#1F1F1F",
    "xtick.color": "#1F1F1F",
    "ytick.color": "#1F1F1F",
})

NF = pathlib.Path(__file__).resolve().parent                     # experiments/nature_fig
REPO = NF.parents[1]
for p in (NF, REPO):
    sys.path.insert(0, str(p))

from nf_style import (MM, TEXT_W_MM, PAL, ARCH, LABEL_MATH_PT, ANNOT_PT, LEGEND_PT,  # noqa: E402
                      add_panel_label, add_panel_title)
from audit_panel_alignment import require_matplotlib_panel_alignment               # noqa: E402

import json                                                                          # noqa: E402
import collections                                                                   # noqa: E402
import numpy as np                                                                   # noqa: E402
from matplotlib.lines import Line2D                                                  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator                              # noqa: E402
from matplotlib.transforms import ScaledTranslation                                  # noqa: E402

OUT = REPO / "paper" / "figures"
QA = NF / "qa"
QA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig8_slope")
QBASE = str(QA / "fig8_slope")

width_mm = 160          # final printed width: 160 (\textwidth)
height_mm = 62

# ── data (read-only): the JSON the current script reads, all 340 configurations ──
ind = json.loads((REPO / "results/independence.json").read_text())
assert len(ind) == 340
FULL_PANEL = ("independent", "centralized", "discussion")     # share the same first opinions
by_n = collections.defaultdict(list)
for r in ind:
    if r["arch"] in FULL_PANEL and r["N"] >= 3:
        by_n[r["N"]].append(r["phi0"])
ns = np.array(sorted(by_n), float)
phi_by_n = np.array([np.mean(by_n[n]) for n in sorted(by_n)])
neff_meas = ns / (1 + (ns - 1) * phi_by_n)
phibar = float(np.mean([v for n in by_n for v in by_n[n]]))
first = {a: np.array([r["phi0"] for r in ind if r["arch"] == a])
         for a in ("independent", "centralized", "tiered", "discussion")}
after = np.array([r["phi_last"] for r in ind if r.get("phi_last") is not None])
assert sum(len(v) for v in first.values()) == 340 and len(after) == 96

INK, GREY, DARK, LIGHT = PAL["ink"], PAL["neutral_mid"], PAL["neutral_dark"], PAL["neutral_light"]
EDGE = dict(color=DARK, lw=0.7, ls=(0, (3, 2)))
MEAS = ARCH["independent"]     # a first round is, by construction, an independent panel


def at(ax, dx_pt=0.0, dy_pt=0.0):
    """Data transform shifted by a fixed physical offset."""
    return ax.transData + ScaledTranslation(dx_pt / 72, dy_pt / 72, ax.figure.dpi_scale_trans)


def right_edge(ax, dx_pt=0.0):
    """x in axes fraction, y in data, shifted by a fixed physical offset."""
    return (mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
            + ScaledTranslation(dx_pt / 72, 0, ax.figure.dpi_scale_trans))


def header(ax, letter, title):
    """Letter at the left edge of the axis decorations, title right after it (pt offsets)."""
    fig_ = ax.figure
    fig_.canvas.draw()
    r = fig_.canvas.get_renderer()
    box = ax.get_window_extent(r)
    off = (ax.get_tightbbox(r).x0 - box.x0) * 72 / fig_.dpi
    off = max(off, -0.33 * box.width * 72 / fig_.dpi)    # stay inside the auditor's label window
    add_panel_label(ax, letter, x_offset_pt=off)
    add_panel_title(ax, title, x_offset_pt=off + 10)


def spread(values, half):
    """Deterministic vertical offsets (golden-ratio sequence in value order): no RNG."""
    order = np.argsort(values, kind="stable")
    offs = np.empty(len(values))
    offs[order] = ((np.arange(len(values)) * 0.6180339887498949) % 1.0 - 0.5) * 2 * half
    return offs


fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), layout="constrained")
fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.07)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.32])
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# ── a: N_eff = N / (1 + (N - 1) phi), fixed-phi curves and the measured panels ──
ax = ax_a
grid = np.linspace(1, 9, 241)
CURVES = [0.0, 0.25, 0.5, 0.75, 0.95]
for phi in CURVES:
    neff = grid / (1 + (grid - 1) * phi)
    ax.plot(grid, neff, color=GREY, lw=0.8, ls=(0, (1, 1.6)), zorder=2)
    lab = r"$\varphi$ = 0, independent" if phi == 0 else f"{phi:g}"
    ax.text(1.0, 9 / (1 + 8 * phi), lab, transform=right_edge(ax, 3), ha="left", va="center",
            fontsize=ANNOT_PT, color=DARK, clip_on=False)
ax.plot([0.6, 9.0], [1.0, 1.0], **EDGE, zorder=1)   # ends with the data, clear of the labels
assert np.all(neff_meas > 0)            # strictly positive: safe on the log axis
ax.plot(ns, neff_meas, color=MEAS["color"], lw=1.5, zorder=5)
ax.plot(ns, neff_meas, MEAS["marker"], ms=5.0, mfc=MEAS["color"], mec="white", mew=0.6,
        zorder=6)
ax.set_yscale("log")
ax.set_xlim(0.6, 9.4)
ax.set_ylim(0.74, 11.0)            # room under the single-doctor line for its label
ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["1", "2", "5", "10"])
ax.set_xticks([1, 3, 5, 7, 9])
ax.set_xlabel(r"Panel size $N$")
ax.set_ylabel(r"Effective opinions, $N_{\mathrm{eff}}$", fontsize=LABEL_MATH_PT)
# the measured curve: identity in a one-entry key (the log axis leaves no room for an
# inline label between neighbouring curves), its N = 9 value at the end of the curve
key = [Line2D([], [], color=MEAS["color"], lw=1.5, marker=MEAS["marker"], ms=4.4,
              mfc=MEAS["color"], mec="white", mew=0.5, label="Measured panels,\n" + r"$\varphi$ = " + f"{phibar:.2f}")]
ax.legend(handles=key, loc="upper left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2,
          borderpad=0.2, handlelength=2.0, handletextpad=0.5, fontsize=LEGEND_PT)
ax.text(ns[-1], neff_meas[-1], f"{neff_meas[-1]:.2f}", transform=at(ax, -4, 4.5), ha="right",
        va="bottom", fontsize=ANNOT_PT, color=INK)
ax.text(1.05, 1.0, "single doctor", transform=at(ax, 0, -2.5), ha="left", va="top",
        fontsize=ANNOT_PT, color=DARK)

# ── b: phi of every configuration; first round for all architectures, then after discussion ──
ax = ax_b
rows = [("Independent", first["independent"], ARCH["independent"]),
        ("Centralized", first["centralized"], ARCH["centralized"]),
        ("Hybrid,\nescalated items", first["tiered"], ARCH["tiered"]),
        ("Decentralized", first["discussion"], ARCH["discussion"]),
        ("Decentralized,\nafter discussion", after, ARCH["discussion"])]
ys = np.array([5.5, 4.5, 3.5, 2.5, 1.0])         # an extra gap marks the change of round
HALF = 0.30
ax.set_xlim(0.28, 1.012)
ax.set_ylim(0.45, 6.15)
for y, (lab, vals, st) in zip(ys, rows):
    ax.scatter(vals, y + spread(vals, HALF), s=9.5, marker=st["marker"], c=st["color"],
               alpha=0.78, edgecolors="white", linewidths=0.25, zorder=3)
    med = float(np.median(vals))
    ax.plot([med, med], [y - 0.40, y + 0.40], color=INK, lw=1.3, solid_capstyle="butt", zorder=5)
    ax.text(1.0, y, f"{med:.2f}", transform=at(ax, 9, 0), ha="left", va="center", fontsize=ANNOT_PT,
            color=INK, clip_on=False)
ax.text(1.0, ys[0] + 0.62, "median", transform=at(ax, 9, 0), ha="left", va="bottom",
        fontsize=ANNOT_PT, color=DARK, clip_on=False)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows])
ax.tick_params(axis="y", length=0, pad=4)
ax.spines["left"].set_visible(False)
ax.set_xticks([0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels(["0.4", "0.6", "0.8", "1.0"])
ax.set_xlabel(r"Error correlation $\varphi$ between panel members")

header(ax_a, "a", "Effective opinions against panel size")
header(ax_b, "b", r"Error correlation $\varphi$ in all 340 configurations")

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
print("N_eff", dict(zip(ns.astype(int).tolist(), np.round(neff_meas, 3).tolist())),
      f"phibar {phibar:.4f}", {k: round(float(np.median(v)), 3) for k, v in first.items()},
      f"after median {np.median(after):.4f} mean {after.mean():.4f} sd {after.std():.4f}")
