"""S3b_tokens_strip — candidate for the cost-accuracy slot (NMI success per 1K tokens), one column (83.34 mm).

Claim: a multi-agent system of 9 agents returns roughly a tenth of the single agent's correct answers
per 1,000 tokens, for every model and data set.
x, architecture (SAS and each multi-agent architecture present); y, correct answers per 1,000 tokens
(accuracy as a fraction divided by nominal input plus output tokens per item, in thousands; log scale).
Dots, systems (model x data set; model colour and marker); black bar, geometric mean over systems,
with its value; text above each multi-agent column, fold reduction from SAS (geometric mean of the per-system
ratios over the systems run with both).
Data: results/cvpr_nmi.json ("cost"), experiments/cvpr_fig/extract_nmi.py.
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
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, ms)   # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment    # noqa: E402
from r3_common import NMI, STYLE, ARCHS, SETS, present                   # noqa: E402

import numpy as np                                                         # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator                    # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
BASE, QBASE = str(OUT / "S3b_tokens_strip"), str(QA / "S3b_tokens_strip")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 64

COST = NMI["cost"]
MODELS_P = present("cost")
ARCH_P = [a for a in ARCHS if any(a in v for v in COST.values())]


def per_k_tokens(c):
    return (c["acc"] / 100) / ((c["in_tok"] + c["out_tok"]) / 1000)


vals = {a: [] for a in ARCH_P}
per_sys = {}                                   # (model, set) -> {architecture: value}, for matched fold changes
fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM),
                       gridspec_kw=dict(left=0.17, right=0.98, bottom=0.335, top=0.97))
spread = np.linspace(-0.26, 0.26, max(len(MODELS_P), 2))
for i, a in enumerate(ARCH_P):
    for j, m in enumerate(MODELS_P):
        for s in SETS:
            c = COST.get(f"{m}|{s}", {}).get(a)
            if not c:
                continue
            v = per_k_tokens(c)
            assert v > 0                     # positive by construction: the log axis needs no pseudocount
            vals[a].append(v)
            per_sys.setdefault((m, s), {})[a] = v
            st = STYLE[m]
            ax.plot([i + spread[j]], [v], ls="none", marker=st["marker"], ms=ms(st["marker"], 3.6),
                    mfc=st["color"], mec="white", mew=0.4, alpha=0.9, zorder=3)
gm = {a: float(np.exp(np.mean(np.log(vals[a])))) for a in ARCH_P if vals[a]}
fold = {}
for i, a in enumerate(ARCH_P):
    ax.plot([i - 0.36, i + 0.36], [gm[a], gm[a]], color=PAL["ink"], lw=1.6, solid_capstyle="butt", zorder=4)
    ax.text(i + 0.40, gm[a], f"{gm[a]:.2f}", ha="left", va="center", fontsize=ANNOT_PT, color=PAL["ink"])
    if a != "SAS":
        # fold reduction over the systems that have both SAS and this architecture
        r = [d["SAS"] / d[a] for d in per_sys.values() if "SAS" in d and a in d]
        fold[a] = float(np.exp(np.mean(np.log(r))))
        ax.text(i, max(vals[a]) * 1.55, f"{fold[a]:.0f}× fewer", ha="center", va="bottom",
                fontsize=ANNOT_PT, color=PAL["neutral_dark"])
ax.set_yscale("log")
lo, hi = min(min(v) for v in vals.values()) / 1.6, max(max(v) for v in vals.values()) * 2.6
ax.set_ylim(lo, hi)
ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1) if lo <= t <= hi]
ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=TICK_PT)
ax.set_xlim(-0.6, len(ARCH_P) - 0.2)
ax.set_xticks(range(len(ARCH_P)))
ax.set_xticklabels([a if a == "SAS" else f"{a}\nMAS" for a in ARCH_P], fontsize=TICK_PT)
ax.set_ylabel("Correct answers per 1,000 tokens", fontsize=BASE_PT)
mh = [Line2D([], [], ls="none", marker=STYLE[m]["marker"], ms=ms(STYLE[m]["marker"], 3.6), mfc=STYLE[m]["color"],
             mec="white", mew=0.4, label=STYLE[m]["label"]) for m in MODELS_P]
mh.append(Line2D([], [], color=PAL["ink"], lw=1.6, label="Geometric mean"))
fig.legend(handles=mh, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=2, fontsize=LEGEND_PT,
           handlelength=1.2, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2)

require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=True)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("geometric means", {a: round(v, 3) for a, v in gm.items()}, "n", {a: len(v) for a, v in vals.items()},
      "matched fold", {a: round(v, 1) for a, v in fold.items()})
print("written", BASE)
