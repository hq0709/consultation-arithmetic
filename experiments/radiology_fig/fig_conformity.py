"""Figure: discussion overturns correct answers that came from the image.

Readers of the discussion panel of 9 who were correct in round 0, split by whether their round-0
answer differed from the same reader's answer on the gray image (came from the image) or matched it
(same as the prior). Ratios of reader counts, item bootstrap 95% CI.
a  share of those correct answers abandoned by the final round
b  the same, within the round-0 majority and within the minority
c  share of those correct answers that were in the round-0 minority
Data: results/imaging_v2_A.json (experiments/imaging_v2_analysis.py A).
"""
import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from rf_style import (MM, PAL, TEXT_W_MM, LEGEND_PT, TICK_PT, BASE_PT, MODELS, MODEL, SETS, SET_LABEL, RESULTS,
                      setup, save, add_panel_label)

setup()
A = json.loads((RESULTS / "imaging_v2_A.json").read_text())
GAP = 1.6
ypos = {(s, m): -(si * (len(MODELS) + GAP) + mi) for si, s in enumerate(SETS) for mi, m in enumerate(MODELS)}
y_head = {s: -(si * (len(MODELS) + GAP)) + 0.95 for si, s in enumerate(SETS)}

fig = plt.figure(figsize=(TEXT_W_MM * MM, 80 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 0.85], left=0.205, right=0.975, bottom=0.12, top=0.885, wspace=0.14)
axes = [fig.add_subplot(gs[0, 0])]
axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
ax_a, ax_b, ax_c = axes


def hbar(ax, y, ci, st, filled=True, dy=0.0, marker=None):
    est, lo, hi = ci
    ax.plot([lo, hi], [y + dy, y + dy], color=st["color"], lw=0.9, solid_capstyle="butt", zorder=3)
    ax.plot([est], [y + dy], marker=marker or st["marker"], ms=3.4, color=st["color"],
            mfc=st["color"] if filled else "white", mec=st["color"], mew=0.9, lw=0, zorder=4)


for (s, m), y in ypos.items():
    cf = ((A.get(f"{m}|{s}") or {}).get("discussion") or {}).get("conformity") or {}
    st = MODEL[m]
    if cf.get("right_to_wrong_image_dependent"):
        hbar(ax_a, y, cf["right_to_wrong_image_dependent"], st, dy=0.19)
    if cf.get("right_to_wrong_prior_consistent"):
        hbar(ax_a, y, cf["right_to_wrong_prior_consistent"], st, filled=False, dy=-0.19)
    for side, dy, mk in (("minority", 0.19, "o"), ("majority", -0.19, "s")):
        v = cf.get(f"right_to_wrong_{side}")
        if v:
            hbar(ax_b, y, v, st, dy=dy, marker=mk)
    if cf.get("image_dependent_correct_in_minority_pct"):
        hbar(ax_c, y, cf["image_dependent_correct_in_minority_pct"], st, dy=0.19)
    if cf.get("prior_consistent_correct_in_minority_pct"):
        hbar(ax_c, y, cf["prior_consistent_correct_in_minority_pct"], st, filled=False, dy=-0.19)

ax_a.set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.012, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=BASE_PT, fontweight="bold", color=PAL["ink"])
lo_y = min(ypos.values()) - 0.8
for ax in axes:
    if ax is not ax_a:
        plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(axis="y", length=0)
    ax.set_ylim(lo_y, max(y_head.values()) + 0.6)
    ax.spines["left"].set_visible(False)
    for s in SETS[1:]:
        ax.axhline(y_head[s] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)

ax_a.set_xlim(0, 40); ax_a.set_xlabel("Abandoned (%)")
ax_b.set_xlim(0, 105); ax_b.set_xlabel("Abandoned (%)")
ax_c.set_xlim(0, 30); ax_c.set_xlabel("In the minority (%)")
hdr = dict(loc="center left", ncol=2, handletextpad=0.0, columnspacing=0.5, borderaxespad=0.0, fontsize=LEGEND_PT)
fig.legend(handles=[Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], label="Differs from gray-image answer"),
                    Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], mfc="white", label="Same as gray-image answer")],
           loc="upper left", bbox_to_anchor=(0.2, 1.0), ncol=2, handletextpad=0.1, columnspacing=1.0, fontsize=LEGEND_PT)
ax_b.legend(handles=[Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], label="In minority"),
                     Line2D([], [], marker="s", ms=3.4, lw=0, color=PAL["ink"], label="In majority")],
            bbox_to_anchor=(-0.02, y_head["vqarad"]), bbox_transform=ax_b.get_yaxis_transform(), **hdr)
for ax, lab in zip(axes, "abc"):
    add_panel_label(ax, lab, x_offset_pt=-2, y_offset_pt=1)
print(save(fig, "fig_conformity"))
