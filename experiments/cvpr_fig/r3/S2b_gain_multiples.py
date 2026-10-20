"""S2b_gain_multiples -- round-3 candidate for the capability slot, full width (174.6 mm).

Question (NMI capability ceiling, per architecture): within each multi-agent architecture, does
the gain over SAS fall as SAS accuracy rises? Each panel prints its own OLS slope (pp of gain per
pp of SAS accuracy; CI from resampling model x data-set systems).
Panels a-d: Independent, Centralized, Decentralized, Hybrid at N = 9, same axes; a panel whose
architecture has not finished running says so instead of showing points.
Encodings: x, SAS accuracy (%); y, multi-agent gain (MAS minus SAS, pp) with paired item-bootstrap
95% CI; colour and marker, model (as in Fig. 1; capability-ladder models in their family hue);
dashed line, zero gain; grey line, OLS fit.
Data: results/cvpr_nmi.json ("arch"). Nature-figure skill, Python backend.
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
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, ms, signed, header)   # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment                       # noqa: E402
import s25_common as c                                                                       # noqa: E402

import numpy as np                                                                           # noqa: E402
from matplotlib.lines import Line2D                                                          # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
OUT.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
BASE, QBASE = str(OUT / "S2b_gain_multiples"), str(QA / "S2b_gain_multiples")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 57.5

pts = c.gain_points(n=9)
models = c.ordered({p[0] for p in pts})
W, H = width_mm, height_mm
L, R, G, B, T = 14.0, 2.0, 4.0, 10.5, 12.5            # mm: left, right, gutter, bottom, top
PW = (W - L - R - 3 * G) / 4
xs_all = np.array([p[3] for p in pts])
lo = min(p[4][1] for p in pts); hi = max(p[4][2] for p in pts)
XL = (10 * np.floor((xs_all.min() - 4) / 10), 100)
YT = np.arange(5 * np.floor((lo - 1) / 5), 5 * np.ceil((hi + 4) / 5) + 0.1, 5)   # y ticks
YL = (YT[0], YT[-1] + 2.5)                            # headroom keeps the top tick label off the panel title
XT = np.arange(20 * np.ceil((XL[0] + 1) / 20), XL[1] - 1, 20)   # no tick label on a shared panel edge

fig = plt.figure(figsize=(W * MM, H * MM))
axes = []
for k, a in enumerate(c.ARCHS):
    ax = fig.add_axes([(L + k * (PW + G)) / W, B / H, PW / W, (H - B - T) / H])
    axes.append(ax)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.axhline(0, color=PAL["neutral_mid"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    P = [p for p in pts if p[2] == a]
    if P:
        fit = c.ols_slope_ci(P)
        if fit:
            slope, icpt, s_lo, s_hi = fit
            x = np.array([p[3] for p in P])
            xx = np.array([x.min() - 2, x.max() + 2])
            ax.plot(xx, icpt + slope * xx, color=PAL["neutral_dark"], lw=1.0, zorder=2)
            ax.text(0.04, 0.97, f"slope {c.fmt(slope)}\n[{c.fmt(s_lo)}, {c.fmt(s_hi)}]", transform=ax.transAxes,
                    ha="left", va="top", fontsize=ANNOT_PT, color=PAL["neutral_dark"])
        for m, s, _, x, g in P:
            st = c.style(m)
            ax.plot([x, x], [g[1], g[2]], color=st["color"], lw=0.6, alpha=0.5, solid_capstyle="butt", zorder=3)
            ax.plot([x], [g[0]], ls="none", marker=st["marker"], ms=ms(st["marker"], 4.0), mfc=st["color"],
                    mec="white", mew=0.4, zorder=4)
    else:
        ax.text(0.5, 0.5, "running", transform=ax.transAxes, ha="center", va="center", fontsize=ANNOT_PT,
                color=PAL["neutral_mid"], style="italic")
    ax.set_xticks(XT)
    ax.set_yticks(YT, [signed(v) for v in YT] if k == 0 else [])
    ax.tick_params(labelsize=TICK_PT)
    if k == 0:
        ax.set_ylabel("Multi-agent gain (pp)", fontsize=BASE_PT)
fig.text((L + (W - L - R) / 2) / W, 1.2 / H, "SAS accuracy (%)", ha="center", va="bottom", fontsize=BASE_PT)
for k, (ax, a) in enumerate(zip(axes, c.ARCHS)):
    header(ax, "abcd"[k], f"{a}, N = 9")

h_mod = [Line2D([], [], ls="none", marker=c.style(m)["marker"], ms=ms(c.style(m)["marker"], 4.0),
                mfc=c.style(m)["color"], mec="white", mew=0.4, label=c.style(m)["label"]) for m in models]
fig.legend(handles=h_mod, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.8 / H), ncol=len(models), frameon=False,
           fontsize=LEGEND_PT, handlelength=1.0, handletextpad=0.4, columnspacing=1.4, borderaxespad=0)

require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=["a", "b", "c", "d"],
                                   row_groups=[{"id": "architectures", "panels": ["a", "b", "c", "d"]}],
                                   json_out=QBASE + ".alignment.json", overlay_svg=QBASE + ".alignment.svg",
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE, "| points", len(pts), "| models", models)
