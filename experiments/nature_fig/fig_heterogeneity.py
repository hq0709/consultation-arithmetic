"""fig4_heterogeneity — candidate A: weakening any member lowers accuracy; specialist capability matters
about four times more than the orchestrator's.

Nature-figure skill, Python backend. Data and estimator reproduce experiments/fig_heterogeneity.py
(same rows, same status filter, same accuracy and relative change vs the all-strong panel).
"""
import sys
import pathlib

sys.dont_write_bytecode = True   # the repository is read-only: nothing may write __pycache__ there

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
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
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
})

NF = pathlib.Path(__file__).resolve().parent                     # experiments/nature_fig
REPO = NF.parents[1]
for p in (NF, REPO):
    sys.path.insert(0, str(p))

from nf_style import (MM, TEXT_W_MM, PAL, add_panel_label, add_panel_title,        # noqa: E402
                      text_height_data, clear_of_lines, minus)
from audit_panel_alignment import require_matplotlib_panel_alignment               # noqa: E402

import json                                                                         # noqa: E402
import collections                                                                  # noqa: E402
import numpy as np                                                                  # noqa: E402

OUT = REPO / "paper" / "figures"
QA = NF / "qa"
QA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig4_heterogeneity")
QBASE = str(QA / "fig4_heterogeneity")
width_mm = 160
height_mm = 60

# ── data (read-only): the heterogeneity arm, same rows and filter as the current figure ──
rows = [json.loads(l) for l in (REPO / "results/H_heterogeneity.jsonl").open() if l.strip()]
n_before = len(rows)
rows = [r for r in rows if r.get("status") == "ok"]
n_after = len(rows)
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["arch"], r["hetero"])].append(r)

# reference, the two single-role placements, the mixed panel, the all-weak floor
ORDER = {"centralized": ["homog-high", "lo-orch/hi-sub", "hi-orch/lo-sub", "mixed-panel", "homog-low"],
         "discussion": ["homog-high", "mixed-panel", "homog-low"]}   # no orchestrator to place
TICK = {"homog-high": "All strong", "lo-orch/hi-sub": "Strong\nspecialists",
        "hi-orch/lo-sub": "Strong\norchestrator", "homog-low": "All weak", "mixed-panel": "Mixed\npanel"}
DECISIVE = {"lo-orch/hi-sub", "hi-orch/lo-sub"}
MIX_GREY = "#B2B2B2"     # midpoint of PAL neutral_mid and neutral_light: partial capability
FILL = {"homog-high": PAL["neutral_mid"], "lo-orch/hi-sub": PAL["cool_dark"],
        "hi-orch/lo-sub": PAL["cool_dark"], "mixed-panel": MIX_GREY, "homog-low": PAL["neutral_light"]}
TITLE = {"centralized": "Centralized (orchestrator + 3 specialists)",
         "discussion": "Decentralized (3 peers, no orchestrator)"}

panels = []
for arch in ("centralized", "discussion"):
    names = ORDER[arch]
    assert all((arch, n) in cells for n in names)
    acc = np.array([100 * np.mean([x["correct"] for x in cells[(arch, n)]]) for n in names])
    ns = [len(cells[(arch, n)]) for n in names]
    panels.append((arch, names, acc, ns))
bases = [acc[names.index("homog-high")] for _, names, acc, _ in panels]
top = max(float(acc.max()) for _, _, acc, _ in panels)

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[5, 3])
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
for ax in axes:
    ax.set_ylim(0, 56)
th = text_height_data(axes[0], 6)            # one 6-pt text line in data units (same scale both panels)
pad = 0.35 * th
row_y = clear_of_lines(top + pad, th, bases, 0.6 * th)  # one change row, clear above every bar and line
for ax in axes:
    ax.set_ylim(0, row_y + th + pad)
for k, (ax, (arch, names, acc, ns)) in enumerate(zip(axes, panels)):
    x = np.arange(len(names))
    ax.bar(x, acc, 0.62, color=[FILL[n] for n in names], edgecolor="white", linewidth=0.6, zorder=3)
    base = acc[names.index("homog-high")]
    ax.axhline(base, ls=(0, (3, 2)), lw=0.7, color=PAL["neutral_dark"], zorder=4)
    for i, (n, a) in enumerate(zip(names, acc)):
        dark = n in DECISIVE
        ax.text(i, a - pad, f"{a:.1f}", ha="center", va="top", fontsize=6, zorder=5,
                color="white" if dark else PAL["ink"])
        if n == "homog-high":
            ax.text(i, row_y, "ref.", ha="center", va="bottom", fontsize=6, color=PAL["neutral_dark"])
        else:
            rel = (a - base) / base * 100
            ax.text(i, row_y, minus(f"{rel:+.0f}%"), ha="center", va="bottom", fontsize=6,
                    color=PAL["delta_down"] if dark else PAL["neutral_dark"],
                    fontweight="bold" if dark else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([TICK[n] for n in names], fontsize=6)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_yticks([0, 20, 40])
    add_panel_title(ax, TITLE[arch])
    if k == 0:
        ax.set_ylabel("Accuracy (%)")
    add_panel_label(ax, "ab"[k])

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
print("rows n_before", n_before, "ok n_after", n_after,
      {a: dict(zip(nm, [round(float(v), 1) for v in ac])) for a, nm, ac, _ in panels},
      "items per cell", sorted({n for *_, ns in panels for n in ns}))
