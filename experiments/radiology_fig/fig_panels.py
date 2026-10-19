"""Figure 2: reading panels of nine add little accuracy because their readers err together.

Forest layout, one row per model and data set, shared by the three panels:
a  accuracy of the independent panel of 9 and of the discussion panel of 9 minus the single reader
   (percentage points, paired item bootstrap 95% CI)
b  effective number of independent readers among the 9, N_eff = 9 / (1 + 8 phi) (95% CI from phi)
c  error correlation phi in round 0 and after discussion
Data: results/imaging_v2_A.json (experiments/imaging_v2_analysis.py A).
"""
import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

from rf_style import (MM, PAL, TEXT_W_MM, ANNOT_PT, LEGEND_PT, TICK_PT, BASE_PT, MODELS, MODEL, SETS, SET_LABEL,
                      RESULTS, setup, save, add_panel_label, minus)

setup()
A = json.loads((RESULTS / "imaging_v2_A.json").read_text())
get = lambda m, s: A.get(f"{m}|{s}") or {}

GAP = 1.6                                                   # blank rows between data sets
ypos = {(s, m): -(si * (len(MODELS) + GAP) + mi) for si, s in enumerate(SETS) for mi, m in enumerate(MODELS)}
y_head = {s: -(si * (len(MODELS) + GAP)) + 0.95 for si, s in enumerate(SETS)}

fig = plt.figure(figsize=(TEXT_W_MM * MM, 74 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.72, 1.08], left=0.205, right=0.985, bottom=0.155, top=0.94, wspace=0.16)
axes = [fig.add_subplot(gs[0, 0])]
axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
ax_a, ax_b, ax_c = axes


def hbar(ax, y, ci, st, filled=True, size=3.4):
    est, lo, hi = ci
    ax.plot([lo, hi], [y, y], color=st["color"], lw=0.9, solid_capstyle="butt", zorder=3)
    ax.plot([est], [y], marker=st["marker"], ms=size, color=st["color"], mfc=st["color"] if filled else "white",
            mec=st["color"], mew=0.9, lw=0, zorder=4)


for (s, m), y in ypos.items():
    st, d = MODEL[m], get(m, s)
    pd_ = d.get("paired") or {}
    if pd_.get("panel9_minus_single"):
        hbar(ax_a, y + 0.19, pd_["panel9_minus_single"], st)
    if pd_.get("disc9_minus_single"):
        hbar(ax_a, y - 0.19, pd_["disc9_minus_single"], st, filled=False)
    ph = (d.get("real") or {}).get("phi")
    if ph:
        n = lambda p: 9 / (1 + 8 * p)
        hbar(ax_b, y, [n(ph[0]), n(ph[2]), n(ph[1])], st)
    phf = (d.get("discussion") or {}).get("phi_last")
    if ph and phf:
        ax_c.plot([ph[0], phf[0]], [y, y], color=st["color"], lw=0.9, zorder=2)
        ax_c.plot([ph[0]], [y], marker=st["marker"], ms=3.4, color=st["color"], lw=0, zorder=3)
        ax_c.plot([phf[0]], [y], marker=st["marker"], ms=3.4, color=st["color"], mfc="white", mew=0.9, lw=0, zorder=3)

# shared rows: model names as tick labels, data set names as group headers
ax_a.set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.012, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=BASE_PT, fontweight="bold", color=PAL["ink"])
for ax in (ax_b, ax_c):
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", length=0)
lo_y = min(ypos.values()) - 0.8
for ax in axes:
    ax.set_ylim(lo_y, max(y_head.values()) + 0.6)
    ax.spines["left"].set_visible(False)
    for s in SETS[1:]:                                       # light rule between data sets
        ax.axhline(y_head[s] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)

ax_a.plot([0, 0], [lo_y, y_head["vqarad"] - 0.55], color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax_a.set_xlabel("Gain over single reader (pp)")
ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: minus(f"{v:+.0f}") if v else "0"))
ax_a.legend(handles=[Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], label="Independent"),
                     Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], mfc="white", label="Discussion")],
            loc="center left", bbox_to_anchor=(-0.02, y_head["vqarad"]), bbox_transform=ax_a.get_yaxis_transform(),
            ncol=2, handletextpad=0.0, columnspacing=0.5, borderaxespad=0.0, fontsize=LEGEND_PT)

ax_b.set_xscale("log")
ax_b.set_xlim(0.92, 10.5)
ax_b.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 9]))
ax_b.xaxis.set_minor_locator(NullLocator())
ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_b.plot([9, 9], [lo_y, y_head["vqarad"] - 0.55], color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax_b.set_xlabel("Effective readers, $N_{\\mathrm{eff}}$")

ax_c.set_xlim(0, 1.05)
ax_c.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
ax_c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_c.set_xlabel("Error correlation, $\\phi$")
ax_c.legend(handles=[Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], label="Round 0"),
                     Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], mfc="white", label="Final")],
            loc="center left", bbox_to_anchor=(-0.02, y_head["vqarad"]), bbox_transform=ax_c.get_yaxis_transform(),
            ncol=2, handletextpad=0.0, columnspacing=0.5, borderaxespad=0.0, fontsize=LEGEND_PT)

for ax, lab in zip(axes, "abc"):
    add_panel_label(ax, lab, x=0, y=1, x_offset_pt=-2, y_offset_pt=1)
print(save(fig, "fig2_panels"))
