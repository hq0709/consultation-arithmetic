"""S4c_debate_lines — candidate for the debate-dynamics slot (NMI error dynamics), full width (174.6 mm).

Claim: over three rounds of debate the agents of one model converge on one answer (error correlation
to 1, unanimity to nearly every item) while majority accuracy moves by a few points at most.
Panels, against debate round (0 = initial answers of the Decentralized MAS of 9, original images):
(a) error correlation phi among the 9 agents, right axis N_eff = 9 / (1 + 8 phi); (b) items on which
all 9 agents give the same answer (%); (c) majority (plurality) accuracy minus its round-0 value (pp).
A debate that stopped at unanimity keeps its last round. Bold line, mean over the three data sets
(VQA-RAD, SLAKE, MedXpertQA-MM); thin lines, each data set. Model colours and markers as in the paper.
Data: results/cvpr_nmi.json ("debate"), experiments/cvpr_fig/extract_nmi.py.
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
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, ms, header, phi_from_neff)   # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment                           # noqa: E402
from r3_common import NMI, STYLE, SETS, present                                                 # noqa: E402

import numpy as np                                                                                # noqa: E402
from matplotlib.lines import Line2D                                                               # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator                                           # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
BASE, QBASE = str(OUT / "S4c_debate_lines"), str(QA / "S4c_debate_lines")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 60

DEB = NMI["debate"]
MODELS_P = present("debate")
R = np.arange(4)


def series(m, s, key):
    d = DEB[f"{m}|{s}"]
    v = np.array(d[key], float)
    return v - v[0] if key == "accuracy" else v


fig, axes = plt.subplots(1, 3, figsize=(width_mm * MM, height_mm * MM),
                         gridspec_kw=dict(left=0.075, right=0.985, bottom=0.2, top=0.79, wspace=0.58))
spec = [("phi", r"Error correlation, $\phi$"), ("unanimity", "Unanimous items (%)"),
        ("accuracy", "Majority accuracy change (pp)")]
for ax, (key, ylab), letter in zip(axes, spec, "abc"):
    for m in MODELS_P:
        st = STYLE[m]
        sets = [s for s in SETS if f"{m}|{s}" in DEB]
        Y = np.array([series(m, s, key) for s in sets])
        for y in Y:
            ax.plot(R, y, color=st["color"], lw=0.6, alpha=0.32, zorder=2)
        ax.plot(R, Y.mean(0), color=st["color"], lw=1.5, marker=st["marker"], ms=ms(st["marker"], 3.4),
                mfc=st["color"], mec="white", mew=0.4, zorder=4)
    ax.set_xticks(R)
    ax.set_xlim(-0.25, 3.25)
    ax.tick_params(labelsize=TICK_PT)
    ax.set_ylabel(ylab, fontsize=BASE_PT)
    header(ax, letter, y_offset_pt=5)
axes[0].set_ylim(0.4, 1.03)
axes[0].yaxis.set_major_locator(FixedLocator([0.4, 0.6, 0.8, 1.0]))
ne = axes[0].twinx()                      # N_eff on the right of panel a, same scale as phi
ne.set_ylim(axes[0].get_ylim())
ne_ticks = [1.75, 1.5, 1.25, 1.0]
ne.yaxis.set_major_locator(FixedLocator([float(phi_from_neff(x)) for x in ne_ticks]))
ne.set_yticklabels([f"{x:g}" for x in ne_ticks], fontsize=TICK_PT)
ne.text(1.0, 1.02, r"$N_{\mathrm{eff}}$", transform=ne.transAxes, ha="center", va="bottom", fontsize=BASE_PT)
ne.spines["right"].set_visible(True); ne.spines["top"].set_visible(False)
axes[1].set_ylim(35, 102)
axes[1].yaxis.set_major_locator(FixedLocator([40, 60, 80, 100]))
lo = min(np.floor(min(series(m, s, "accuracy").min() for m in MODELS_P for s in SETS if f"{m}|{s}" in DEB)) - 1, -2)
hi = max(np.ceil(max(series(m, s, "accuracy").max() for m in MODELS_P for s in SETS if f"{m}|{s}" in DEB)) + 1, 2)
axes[2].set_ylim(lo, hi)
axes[2].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
axes[2].axhline(0, color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
fig.supxlabel("Debate round (0, initial answers)", y=0.0, va="bottom", fontsize=BASE_PT)
mh = [Line2D([], [], color=STYLE[m]["color"], lw=1.5, marker=STYLE[m]["marker"], ms=ms(STYLE[m]["marker"], 3.4),
             mfc=STYLE[m]["color"], mec="white", mew=0.4, label=STYLE[m]["label"]) for m in MODELS_P]
mh.append(Line2D([], [], color=PAL["neutral_dark"], lw=0.6, alpha=0.5, label="Single data set"))
fig.legend(handles=mh, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(mh), fontsize=LEGEND_PT,
           handlelength=1.8, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.3)

require_matplotlib_panel_alignment(fig, axes=list(axes), panel_ids=["a", "b", "c"],
                                   row_groups=[{"id": "metrics", "panels": ["a", "b", "c"]}],
                                   json_out=QBASE + ".alignment.json", overlay_svg=QBASE + ".alignment.svg",
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
fin = {m: [round(DEB[f"{m}|{s}"]["phi"][-1], 3) for s in SETS if f"{m}|{s}" in DEB] for m in MODELS_P}
print("round-3 phi", fin)
print("written", BASE)
