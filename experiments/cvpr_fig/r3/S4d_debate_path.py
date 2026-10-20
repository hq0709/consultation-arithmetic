"""S4d_debate_path — candidate for the debate-dynamics slot (NMI error dynamics), one column (83.34 mm).

Claim: debate moves every system along the same path, toward unanimity and an error correlation of 1
(N_eff = 1), and leaves majority accuracy within a few points of its initial value.
x, items on which all 9 agents give the same answer (%); y, error correlation phi among the agents
(right axis, N_eff = 9 / (1 + 8 phi)). Each line is one system (model x data set) of the
Decentralized MAS of 9 on original images: open marker, initial answers (round 0); small dots,
rounds 1 and 2; filled marker, round 3. Line style, data set. The note gives the range of majority
accuracy change from round 0 to round 3 over the systems drawn. Model colours and markers as in the
paper. Data: results/cvpr_nmi.json ("debate"), experiments/cvpr_fig/extract_nmi.py.
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
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, ms, phi_from_neff, minus)   # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment                           # noqa: E402
from r3_common import NMI, STYLE, SETS, SET_LABEL, present                                      # noqa: E402

import numpy as np                                                                                # noqa: E402
from matplotlib.lines import Line2D                                                               # noqa: E402
from matplotlib.ticker import FixedLocator                                                        # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
BASE, QBASE = str(OUT / "S4d_debate_path"), str(QA / "S4d_debate_path")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 80

DEB = NMI["debate"]
MODELS_P = present("debate")
LS = {"vqarad": "-", "slake": (0, (3.2, 1.6)), "mxqa_mm_rad": (0, (1, 1.3))}

fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM),
                       gridspec_kw=dict(left=0.155, right=0.87, bottom=0.325, top=0.93))
dacc = []
for m in MODELS_P:
    st = STYLE[m]
    for s in SETS:
        d = DEB.get(f"{m}|{s}")
        if not d:
            continue
        x, y = np.array(d["unanimity"], float), np.array(d["phi"], float)
        dacc.append(d["accuracy"][-1] - d["accuracy"][0])
        ax.plot(x, y, color=st["color"], lw=0.9, ls=LS[s], alpha=0.9, zorder=3, dash_capstyle="butt")
        ax.plot(x[1:-1], y[1:-1], ls="none", marker="o", ms=1.8, color=st["color"], zorder=4)
        ax.plot([x[0]], [y[0]], ls="none", marker=st["marker"], ms=ms(st["marker"], 3.8), mfc="white",
                mec=st["color"], mew=0.9, zorder=5)
        ax.plot([x[-1]], [y[-1]], ls="none", marker=st["marker"], ms=ms(st["marker"], 3.8), mfc=st["color"],
                mec="white", mew=0.4, zorder=6)
ax.set_xlim(35, 103)
ax.set_ylim(0.4, 1.04)
ax.xaxis.set_major_locator(FixedLocator([40, 60, 80, 100]))
ax.yaxis.set_major_locator(FixedLocator([0.4, 0.6, 0.8, 1.0]))
ax.tick_params(labelsize=TICK_PT)
ax.set_xlabel("Unanimous items (%)", fontsize=BASE_PT)
ax.set_ylabel(r"Error correlation, $\phi$", fontsize=BASE_PT)
ne = ax.twinx()
ne.set_ylim(ax.get_ylim())
ne_ticks = [1.75, 1.5, 1.25, 1.0]
ne.yaxis.set_major_locator(FixedLocator([float(phi_from_neff(v)) for v in ne_ticks]))
ne.set_yticklabels([f"{v:g}" for v in ne_ticks], fontsize=TICK_PT)
ne.text(1.0, 1.015, r"$N_{\mathrm{eff}}$", transform=ne.transAxes, ha="center", va="bottom", fontsize=BASE_PT)
ne.spines["right"].set_visible(True); ne.spines["top"].set_visible(False)
lo_, hi_ = min(dacc), max(dacc)
ax.text(0.035, 0.975, "Majority accuracy, round 3 minus round 0:\n" + minus(f"{lo_:+.1f}") + " to " + minus(f"{hi_:+.1f}") + " pp",
        transform=ax.transAxes, ha="left", va="top", fontsize=ANNOT_PT, color=PAL["ink"], linespacing=1.25)
mh = [Line2D([], [], ls="none", marker=STYLE[m]["marker"], ms=ms(STYLE[m]["marker"], 3.8), mfc=STYLE[m]["color"],
             mec="white", mew=0.4, label=STYLE[m]["label"]) for m in MODELS_P]
sh = [Line2D([], [], color=PAL["neutral_dark"], lw=0.9, ls=LS[s], label=SET_LABEL[s]) for s in SETS]
kh = [Line2D([], [], ls="none", marker="o", ms=3.8, mfc="white", mec=PAL["neutral_dark"], mew=0.9, label="Round 0"),
      Line2D([], [], ls="none", marker="o", ms=3.8, mfc=PAL["neutral_dark"], mec="white", mew=0.4, label="Round 3")]
fig.legend(handles=mh + kh, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=LEGEND_PT,
           handlelength=1.0, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
fig.legend(handles=sh, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1, fontsize=LEGEND_PT,
           handlelength=2.0, handletextpad=0.4, borderaxespad=0.2)

require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=True)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("accuracy change range", round(lo_, 1), round(hi_, 1), "systems", len(dacc))
print("written", BASE)
