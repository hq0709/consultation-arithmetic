"""Figure: a panel's accuracy follows from two numbers, its readers' mean accuracy and their error
correlation.

Every 9-reader panel on yes/no items (VQA-RAD, SLAKE, SLAKE-target, SLAKE-pairs) in every condition:
original and degraded images, specialty and generic roles, renderings, reference cases, reports.
a  majority-vote accuracy against the beta-binomial prediction from the readers' mean accuracy and
   their error correlation phi (intraclass correlation)
b  gain of the majority over its own readers' mean accuracy, against phi; curves, the beta-binomial
   gain for readers of mean accuracy 60%, 75% and 90%
Data: results/imaging_v2_A.json, key panel_arithmetic (experiments/imaging_v2_analysis.py A).
"""
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import betabinom

from rf_style import (MM, PAL, TEXT_W_MM, ANNOT_PT, LEGEND_PT, MODELS, MODEL, RESULTS, setup, save, add_panel_label,
                      minus)

setup()
P = json.loads((RESULTS / "imaging_v2_A.json").read_text())["panel_arithmetic"]
cells = P["cells"]

fig = plt.figure(figsize=(TEXT_W_MM * MM, 62 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], left=0.095, right=0.985, bottom=0.17, top=0.84, wspace=0.34)
ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

for c in cells:
    st = MODEL[c["model"]]
    kw = dict(marker=st["marker"], ms=3.2, color=st["color"], mec="white", mew=0.3, lw=0, alpha=0.9, zorder=3)
    ax_a.plot([c["predicted"]], [c["majority"]], **kw)
    ax_b.plot([c["phi"]], [c["majority"] - c["reader_mean"]], **kw)

lo, hi = 30, 100
ax_a.plot([lo, hi], [lo, hi], color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax_a.set_xlim(lo, hi); ax_a.set_ylim(lo, hi); ax_a.set_aspect("equal")
ax_a.set_xlabel("Predicted majority accuracy (%)")
ax_a.set_ylabel("Observed majority accuracy (%)")
ax_a.text(0.96, 0.05, f"{P['n_cells']} panels\nobserved − predicted\n{minus(f'{P['observed_minus_predicted_mean']:+.1f}')} pp, SD {P['observed_minus_predicted_sd']:.1f}",
          transform=ax_a.transAxes, ha="right", va="bottom", fontsize=ANNOT_PT, color=PAL["neutral_dark"])

ph = np.linspace(0.005, 0.995, 200)
for pbar, ls in ((0.6, (0, (1, 1.5))), (0.75, (0, (4, 1.5))), (0.9, "-")):
    a_, b_ = pbar * (1 - ph) / ph, (1 - pbar) * (1 - ph) / ph
    g = (1 - betabinom.cdf(4, 9, a_, b_) - pbar) * 100
    ax_b.plot(ph, g, color=PAL["neutral_mid"], lw=0.9, ls=ls, zorder=2)
    ax_b.text(0.018, g[0] + 0.3, f"{pbar:.0%}", fontsize=ANNOT_PT, color=PAL["neutral_dark"], ha="left", va="bottom")
ax_b.axhline(0, color=PAL["neutral_light"], lw=0.6, zorder=0)
ax_b.set_xlim(0, 1); ax_b.set_ylim(-4, 23)
ax_b.set_xlabel("Error correlation of the readers, $\\phi$")
ax_b.set_ylabel("Majority minus mean reader (pp)")
ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: minus(f"{v:+.0f}") if v else "0"))
ax_b.text(0.2, 21.5, "Curves: mean reader accuracy", fontsize=ANNOT_PT, color=PAL["neutral_dark"], ha="left", va="top")

fig.legend(handles=[Line2D([], [], marker=MODEL[m]["marker"], ms=3.6, lw=0, color=MODEL[m]["color"], label=MODEL[m]["label"])
                    for m in MODELS], loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0), handletextpad=0.2,
           columnspacing=1.2, fontsize=LEGEND_PT)
add_panel_label(ax_a, "d", x_offset_pt=-34, y_offset_pt=6)        # Fig. 2d-e, under fig2_panels (a-c)
add_panel_label(ax_b, "e", x_offset_pt=-38, y_offset_pt=6)
print(save(fig, "fig_arithmetic"))
