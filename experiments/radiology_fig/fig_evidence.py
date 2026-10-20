"""Figure 3: less image evidence lowers accuracy toward the question-text prior and, for most models,
raises the error correlation of the readers.

Rows: a single-reader accuracy; b error correlation phi of the independent panel of 9.
Columns: data sets. x: image at its original size (longest side 1024 px), 256 px, 128 px, uniform gray.
Data: results/imaging_v2_A.json (experiments/imaging_v2_analysis.py A).
"""
import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from rf_style import (MM, PAL, TEXT_W_MM, ANNOT_PT, LEGEND_PT, MODELS, MODEL, SETS, SET_LABEL, RESULTS,
                      setup, save, add_panel_label)

setup()
A = json.loads((RESULTS / "imaging_v2_A.json").read_text())
LEVELS = ["real", "res256", "res128", "blank"]
TICKS = ["≤1024", "256", "128", "Blank"]
CHANCE = {"vqarad": 50, "slake": 50, "mxqa_mm_rad": 20}
DX = 0.07

fig, axes = plt.subplots(2, 3, figsize=(TEXT_W_MM * MM, 86 * MM), sharex=True)
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.885, wspace=0.28, hspace=0.16)
for ci, s in enumerate(SETS):
    for ri, key in enumerate(("single", "phi")):
        ax = axes[ri, ci]
        for mi, m in enumerate(MODELS):
            d = A.get(f"{m}|{s}") or {}
            pts = [(li + (mi - 1.5) * DX, (d.get(lv) or {}).get(key)) for li, lv in enumerate(LEVELS)]
            st = MODEL[m]
            for (x0, v0), (x1, v1) in zip(pts, pts[1:]):        # join adjacent levels only
                if v0 and v1:
                    ax.plot([x0, x1], [v0[0], v1[0]], color=st["color"], lw=1.0, zorder=2)
            pts = [(x, v) for x, v in pts if v]
            for x, v in pts:
                ax.plot([x, x], [v[1], v[2]], color=st["color"], lw=0.8, zorder=2)
                ax.plot([x], [v[0]], marker=st["marker"], ms=3.4, color=st["color"], lw=0, zorder=3)
        if key == "single":
            ax.axhline(CHANCE[s], color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
            ax.text(-0.33, CHANCE[s] - 1.5, "Chance", ha="left", va="top", fontsize=ANNOT_PT, color=PAL["neutral_dark"])
            ax.set_ylim(10 if s == "mxqa_mm_rad" else 30, 100)
            ax.set_title(SET_LABEL[s], fontsize=9, pad=4)
        else:
            ax.set_ylim(0.3, 1.0)
            ax.set_xticks(range(4), TICKS)
        ax.set_xlim(-0.4, 3.4)
axes[0, 0].set_ylabel("Accuracy (%)")
axes[1, 0].set_ylabel("Error correlation, $\\phi$")
fig.text(0.537, 0.02, "Image (longest side, px)", ha="center", va="bottom", fontsize=9)
add_panel_label(axes[0, 0], "a", x_offset_pt=-22)
add_panel_label(axes[1, 0], "b", x_offset_pt=-22)
fig.legend(handles=[Line2D([], [], marker=MODEL[m]["marker"], ms=4, lw=1.0, color=MODEL[m]["color"], label=MODEL[m]["label"])
                    for m in MODELS], loc="upper center", ncol=4, bbox_to_anchor=(0.537, 1.0), handletextpad=0.3,
           columnspacing=1.2, fontsize=LEGEND_PT, handlelength=1.6)
print(save(fig, "fig3_evidence"))
