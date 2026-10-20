"""Figure 7: what makes two AI readers err together.

a  error correlation of two readers that share everything (same model, image, rendering, role and
   reference cases; generic reading panel), item bootstrap 95% CI
b  change in error correlation when two readers of the same model do not share one component:
   matched within-model contrasts of mean pairwise phi (docs/IMAGING_PROTOCOL.md section 12).
   Colored markers, each model (gray, each model pair for "reasoner" and "model"); black, mean over
   models with item bootstrap 95% CI.
Data: results/imaging_v2_C.json (experiments/imaging_v2_analysis.py C).
"""
import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from rf_style import (MM, PAL, TEXT_W_MM, LEGEND_PT, TICK_PT, BASE_PT, MODELS, MODEL, SETS, SET_LABEL, RESULTS,
                      setup, save, add_panel_label, minus)

setup()
C = json.loads((RESULTS / "imaging_v2_C.json").read_text())
SET_MARK = {"vqarad": "o", "slake": "s", "mxqa_mm_rad": "^"}
SET_DY = {"vqarad": 0.22, "slake": 0.0, "mxqa_mm_rad": -0.22}
ROWS = [("role", "Role"), ("rendering", "Contrast adjustment"), ("reference cases", "Reference cases"),
        ("report", "Report"), ("perceiver", "Perception model"), ("reasoner", "Interpretation model"),
        ("model", "Model")]

fig = plt.figure(figsize=(TEXT_W_MM * MM, 112 * MM))
outer = fig.add_gridspec(2, 1, height_ratios=[4, 7.6], left=0.24, right=0.972, bottom=0.18, top=0.935, hspace=0.5)
ax_a = fig.add_subplot(outer[0])
gb = outer[1].subgridspec(1, 3, wspace=0.1)
axb = [fig.add_subplot(gb[0, 0])]
axb += [fig.add_subplot(gb[0, i], sharey=axb[0], sharex=axb[0]) for i in (1, 2)]


def hbar(ax, y, ci, color, marker, ms=3.4, lw=0.9):
    est, lo, hi = ci
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=3)
    ax.plot([est], [y], marker=marker, ms=ms, color=color, lw=0, zorder=4)


# a: sampling-only baseline
for mi, m in enumerate(MODELS):
    for s in SETS:
        v = (((C.get(s) or {}).get("contrasts") or {}).get("sampling only") or {}).get(m)
        if v:
            hbar(ax_a, -mi + SET_DY[s], v, MODEL[m]["color"], SET_MARK[s])
ax_a.set_yticks([-i for i in range(len(MODELS))], [MODEL[m]["label"] for m in MODELS], fontsize=TICK_PT)
ax_a.set_ylim(-len(MODELS) + 0.5, 0.5); ax_a.set_xlim(0.3, 1.0)
ax_a.set_xlabel("Error correlation, $\\phi$")
ax_a.tick_params(axis="y", length=0, pad=3); ax_a.spines["left"].set_visible(False)
ax_a.legend(handles=[Line2D([], [], marker=SET_MARK[s], ms=3.6, lw=0, color=PAL["ink"], label=SET_LABEL[s]) for s in SETS],
            loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, handletextpad=0.1, columnspacing=1.2,
            borderaxespad=0.1, fontsize=LEGEND_PT)
fig.text(0.045, ax_a.get_position().y1 + 0.012, "Self-consistency readers", ha="left", va="bottom", fontsize=BASE_PT)

# b: matched within-model contrasts, one panel per data set
for si, (s, ax) in enumerate(zip(SETS, axb)):
    con = (C.get(s) or {}).get("contrasts") or {}
    for ri, (key, _) in enumerate(ROWS):
        d = con.get(key) or {}
        y = -ri
        items = sorted((k, v) for k, v in d.items() if k != "mean")
        for j, (k, v) in enumerate(items):
            dy = 0.3 - 0.6 * (j + 0.5) / max(1, len(items))
            st = MODEL.get(k)
            ax.plot([v[0]], [y + dy], marker=st["marker"] if st else "o", ms=2.8 if st else 2.4,
                    color=st["color"] if st else PAL["neutral_mid"], lw=0, zorder=3)
        if d.get("mean"):
            est, lo, hi = d["mean"]
            ax.plot([lo, hi], [y, y], color=PAL["ink"], lw=1.3, solid_capstyle="butt", zorder=5)
            ax.plot([est], [y], marker="|", ms=7, mew=1.6, color=PAL["ink"], lw=0, zorder=6)
    ax.axvline(0, color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-0.75, 0.15)
    ax.set_title(SET_LABEL[s], fontsize=BASE_PT, pad=3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: minus(f"{v:+.1f}") if abs(v) > 1e-9 else "0"))
    ax.set_xticks([-0.6, -0.4, -0.2, 0])
    ax.tick_params(axis="y", length=0, pad=3); ax.spines["left"].set_visible(False)
    if si:
        plt.setp(ax.get_yticklabels(), visible=False)
axb[0].set_yticks([-i for i in range(len(ROWS))], [lab for _, lab in ROWS], fontsize=TICK_PT)
axb[0].set_ylim(-len(ROWS) + 0.5, 0.6)
axb[1].set_xlabel("Change in error correlation when the two readers do not share it, $\\Delta\\phi$")
fig.text(0.045, axb[0].get_position().y1 + 0.045, "Two readers of one model that do not share", ha="left", va="bottom",
         fontsize=BASE_PT)
fig.legend(handles=[Line2D([], [], marker=MODEL[m]["marker"], ms=3.2, lw=0, color=MODEL[m]["color"], label=MODEL[m]["label"])
                    for m in MODELS] + [Line2D([], [], marker="o", ms=2.6, lw=0, color=PAL["neutral_mid"], label="Model pair"),
                                        Line2D([], [], marker="|", ms=7, mew=1.6, lw=1.3, color=PAL["ink"], label="Mean, 95% CI")],
           loc="lower center", bbox_to_anchor=(0.55, -0.005), ncol=3, handletextpad=0.1, columnspacing=1.2,
           labelspacing=0.3, fontsize=LEGEND_PT)
for ax, lab, dy in ((ax_a, "a", 0.012), (axb[0], "b", 0.045)):
    fig.text(0.008, ax.get_position().y1 + dy, lab, ha="left", va="bottom", fontsize=10, fontweight="bold")
print(save(fig, "fig7_budget"))
