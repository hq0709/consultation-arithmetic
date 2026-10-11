"""fig6_coordination (paper Fig. 7): panels coordinate in few turns and still discard correct answers.

Nature-figure skill, Python backend. Same metrics as the former fig_rest.py::fig_coord:
  a  reasoning turns T vs number of agents n for the 480 multi-agent configurations,
     their log-log fit and the published general-domain law. Self-consistency is left
     out of the points and the fit: its samples never exchange messages;
  b  mean error amplification A_e per architecture, SD across configurations.
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
    # italic variables in labels use the same sans family as the text
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
})

NF = pathlib.Path(__file__).resolve().parent                     # experiments/nature_fig
REPO = NF.parents[1]
for p in (NF, REPO):
    sys.path.insert(0, str(p))

from nf_style import (MM, PAL, ARCH, MAS_ORDER, BASE_PT, TICK_PT, LEGEND_PT,     # noqa: E402
                      ANNOT_PT, add_panel_label, add_panel_title)
from audit_panel_alignment import require_matplotlib_panel_alignment               # noqa: E402

import numpy as np                                                                 # noqa: E402
from matplotlib.lines import Line2D                                                # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter            # noqa: E402
from experiments.fig_rest import build                                             # noqa: E402  (read-only)
from mechanisms.nmi_metrics import config_metrics, fit_turn_powerlaw               # noqa: E402
from panels.architectures import NMI_CLASS                                         # noqa: E402

OUT = REPO / "paper" / "figures"
QA = NF / "qa"
QA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig6_coordination")
QBASE = str(QA / "fig6_coordination")

width_mm = 160          # final printed width: \textwidth
height_mm = 68

# ── data: identical computation to fig_rest.fig_coord (read-only) ──
cells = build()
md = []
for (m, b, a, N), v in cells.items():
    if a in ("cot", "zeroshot"):
        continue
    sas = cells.get((m, b, "cot", 1))
    if not sas:
        continue
    r = config_metrics(v, sas)
    if r:
        r.update(arch=a, cls=NMI_CLASS.get(a, a), model=m, bench=b)
        md.append(r)
assert len(md) == 600, len(md)                     # 480 multi-agent + 120 self-consistency
mas = [r for r in md if r["arch"] != "sc"]
assert len(mas) == 480, len(mas)
pl = fit_turn_powerlaw([(r["n_agents"], r["turns"]) for r in mas])  # appendix: a = 0.72, b = 1.002
assert (round(pl["a"], 2), round(pl["exponent"], 3), round(pl["r2"], 3)) == (0.72, 1.002, 0.499)
GEN_A, GEN_B, GEN_R2 = 2.72, 1.724, 0.97           # published general-domain law
C_STAR = 0.39                                      # published message-density plateau
CLS = ["Independent", "Centralized", "Decentralized", "Hybrid"]
CLS_KEY = dict(zip(CLS, MAS_ORDER))
ae = {c: np.array([r["error_amp"] for r in md if r["cls"] == c]) for c in CLS}
ae_mean = {c: float(ae[c].mean()) for c in CLS}
ae_sd = {c: float(ae[c].std(ddof=1)) for c in CLS}
assert [round(ae_mean[c], 2) for c in CLS] == [0.29, 0.32, 0.25, 0.09]
assert all(len(ae[c]) == 120 for c in CLS)

# ── encoding ──
MS_BASE = 3.4                                          # marker size (pt) for a circle
MS_SCALE = {"o": 1.0, "s": 0.88, "^": 1.15, "D": 0.86, "X": 1.15}  # optical equal size
POINT_ALPHA = 0.85
REF_C, REF_LW, REF_LS = PAL["neutral_mid"], 0.8, (0, (3, 2))
FIT_C, FIT_LW = PAL["ink"], 1.3
LABEL_C = PAL["neutral_dark"]
PT_KEYS = MAS_ORDER                                    # drawing order in panel a
# horizontal offset (agents) per architecture so coincident configurations stay visible
DODGE = {"independent": -0.32, "discussion": 0.32,
         "centralized": -0.22, "tiered": 0.22}

def ms(a):
    return MS_BASE * MS_SCALE[ARCH[a]["marker"]]

def scatter_points(ax, xs, ys, a, alpha=POINT_ALPHA):
    ax.plot(xs, ys, ls="none", marker=ARCH[a]["marker"], ms=ms(a), mfc=ARCH[a]["color"],
            mec="white", mew=0.35, alpha=alpha, zorder=3)

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), layout="constrained")
fig.get_layout_engine().set(w_pad=1.5 / 72, h_pad=1.5 / 72, wspace=0.06, hspace=0.0)
gs = fig.add_gridspec(1, 2)
axA, axC = (fig.add_subplot(gs[0, i]) for i in range(2))   # author kept panels a and c

# ── a: turn scaling ──
turns = np.array([r["turns"] for r in mas])
if np.any(turns <= 0):
    raise ValueError("log axis needs strictly positive turn counts")
for a in PT_KEYS:
    sub = [r for r in mas if r["arch"] == a]
    scatter_points(axA, [r["n_agents"] + DODGE[a] for r in sub], [r["turns"] for r in sub], a)
n_fit = np.linspace(1, 10, 120)                        # this study: the data's n range
axA.plot(n_fit, pl["a"] * (n_fit + 0.5) ** pl["exponent"], color=FIT_C, lw=FIT_LW, zorder=5,
         solid_capstyle="round")
n_gen = np.linspace(1, 10, 80)                         # published law: the current figure's range
axA.plot(n_gen, GEN_A * (n_gen + 0.5) ** GEN_B, color=REF_C, lw=REF_LW, ls=REF_LS, zorder=4)
axA.set_yscale("log")
axA.set_xlim(0, 11)
axA.set_ylim(0.75, 1600)       # headroom for the fit key above the dashed law
axA.xaxis.set_major_locator(FixedLocator([1, 5, 10]))
axA.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
axA.yaxis.set_major_formatter(FixedFormatter(["1", "10", "100"]))
axA.yaxis.set_minor_locator(NullLocator())
axA.set_xlabel("Number of agents $n$", fontsize=BASE_PT)
axA.set_ylabel("Reasoning turns $T$", fontsize=BASE_PT)
add_panel_label(axA, "a")
add_panel_title(axA, "Turn scaling")

# ── b (was c): error amplification (hero) ──
y = np.arange(len(CLS))[::-1]
for yi, c in zip(y, CLS):
    a = CLS_KEY[c]
    axC.barh(yi, ae_mean[c], height=0.62, color=ARCH[a]["color"], edgecolor="none", zorder=3)
    # A_e >= 0: the lower whisker of mean - SD is truncated at 0 (Hybrid: SD exceeds the mean)
    axC.errorbar(ae_mean[c], yi, xerr=[[min(ae_sd[c], ae_mean[c])], [ae_sd[c]]], fmt="none", ecolor=PAL["ink"], elinewidth=0.8,
                 capsize=1.6, capthick=0.8, zorder=4)
axC.set_xlim(0, 0.62)
axC.set_ylim(-0.6, len(CLS) - 0.4)
axC.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6]))
axC.xaxis.set_major_formatter(FixedFormatter(["0", "0.2", "0.4", "0.6"]))
axC.yaxis.set_major_locator(FixedLocator(list(y)))
axC.yaxis.set_major_formatter(FixedFormatter(CLS))
axC.tick_params(axis="y", length=0, pad=3, labelsize=TICK_PT)
axC.set_xlabel("Error amplification $A$ₑ", fontsize=BASE_PT)
add_panel_label(axC, "b")
add_panel_title(axC, "Information discarded")

# ── shared legend strip ──
handles = [Line2D([], [], ls="none", marker=ARCH[a]["marker"], ms=ms(a) * 1.1,
                  mfc=ARCH[a]["color"], mec="white", mew=0.35, label=ARCH[a]["label"])
           for a in MAS_ORDER]
fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), fontsize=LEGEND_PT,
           handlelength=1.6, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.2)

# ── the two fits, keyed by their own line styles in the free upper-left corner ──
fit_key = [Line2D([], [], color=FIT_C, lw=FIT_LW,
                  label="This study\n$b$ = %.2f, $a$ = %.2f, $R$² = %.2f" % (pl["exponent"], pl["a"], pl["r2"])),
           Line2D([], [], color=REF_C, lw=REF_LW, ls=REF_LS,
                  label="General domain\n$b$ = %.2f, $a$ = %.2f, $R$² = %.2f" % (GEN_B, GEN_A, GEN_R2))]
axA.legend(handles=fit_key, loc="upper left", fontsize=ANNOT_PT, handlelength=2.0, handletextpad=0.5,
           labelspacing=0.5, borderaxespad=0.3, labelcolor=[PAL["ink"], LABEL_C])
fig.canvas.draw()
for yi, c in zip(y, CLS):
    axC.text(ae_mean[c] + ae_sd[c] + 0.018, yi, f"{ae_mean[c]:.2f}", ha="left", va="center",
             fontsize=ANNOT_PT, color=PAL["ink"], zorder=5)

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
print("fit", {k: round(v, 3) for k, v in pl.items()}, "A_e", {c: (round(ae_mean[c], 3), round(ae_sd[c], 3)) for c in CLS})
