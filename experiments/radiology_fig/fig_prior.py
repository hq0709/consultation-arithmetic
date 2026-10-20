"""Figure 4: most answers, right or wrong, are the answers the readers give without the image.

a  every reader answer of the independent panel of 9, cross-tabulated by its correctness with the
   image and with a uniform gray image (same reader slot, same prompt). Right of zero: correct with
   the image; left: wrong with the image. Saturated segments, next to zero, are the answers the image
   changed; light outer segments are the answers that are the same with or without it.
b  consensus errors (at least 7 of 9 readers wrong) whose majority answer is also the gray-image
   majority answer (item bootstrap 95% CI); n, number of consensus errors.
Data: results/imaging_v2_A.json (experiments/imaging_v2_analysis.py A).
"""
import json

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from rf_style import (MM, PAL, TEXT_W_MM, ANNOT_PT, LEGEND_PT, TICK_PT, BASE_PT, MODELS, MODEL, SETS, SET_LABEL,
                      RESULTS, setup, save, add_panel_label)

setup()
A = json.loads((RESULTS / "imaging_v2_A.json").read_text())
# diverging pair (dataviz reference: blue <-> red), saturated = changed by the image, light = unchanged
C_RIGHT_IMG, C_RIGHT_BOTH = "#256abf", "#b7d3f6"
C_WRONG_IMG, C_WRONG_BOTH = "#e34948", "#f4c2b8"

GAP = 1.6
ypos = {(s, m): -(si * (len(MODELS) + GAP) + mi) for si, s in enumerate(SETS) for mi, m in enumerate(MODELS)}
y_head = {s: -(si * (len(MODELS) + GAP)) + 0.95 for si, s in enumerate(SETS)}
H = 0.62                                                     # bar height in row units

fig = plt.figure(figsize=(TEXT_W_MM * MM, 88 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], left=0.205, right=0.975, bottom=0.27, top=0.945, wspace=0.08)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

for (s, m), y in ypos.items():
    pdq = (A.get(f"{m}|{s}") or {}).get("prior_decomposition")
    if not pdq:
        continue
    rr, wr = pdq["prior_right_stays_right"], pdq["image_corrects_prior"]
    rw, ww = pdq["image_misleads_prior"], pdq["prior_wrong_stays_wrong"]
    ax_a.barh(y, wr, left=0, height=H, color=C_RIGHT_IMG, edgecolor="white", lw=0.5)
    ax_a.barh(y, rr, left=wr, height=H, color=C_RIGHT_BOTH, edgecolor="white", lw=0.5)
    ax_a.barh(y, -rw, left=0, height=H, color=C_WRONG_IMG, edgecolor="white", lw=0.5)
    ax_a.barh(y, -ww, left=-rw, height=H, color=C_WRONG_BOTH, edgecolor="white", lw=0.5)
    ax_a.text(wr + rr + 1.2, y, f"{wr + rr:.0f}%", ha="left", va="center", fontsize=ANNOT_PT, color=PAL["neutral_dark"])
    ce = pdq.get("consensus_errors_matching_blank_majority")
    if ce:
        est, lo, hi = ce
        ax_b.plot([lo, hi], [y, y], color=MODEL[m]["color"], lw=0.9, solid_capstyle="butt", zorder=3)
        ax_b.plot([est], [y], marker=MODEL[m]["marker"], ms=3.4, color=MODEL[m]["color"], lw=0, zorder=4)
        ax_b.text(103, y, f"{pdq['consensus_errors']}", ha="left", va="center", fontsize=ANNOT_PT, color=PAL["neutral_dark"])

ax_a.set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.012, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=BASE_PT, fontweight="bold", color=PAL["ink"])
plt.setp(ax_b.get_yticklabels(), visible=False)
ax_b.tick_params(axis="y", length=0)
lo_y = min(ypos.values()) - 0.8
for ax in (ax_a, ax_b):
    ax.set_ylim(lo_y, max(y_head.values()) + 0.6)
    ax.spines["left"].set_visible(False)
    for s in SETS[1:]:
        ax.axhline(y_head[s] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)

ax_a.axvline(0, color=PAL["ink"], lw=0.8, zorder=5)
ax_a.set_xlim(-102, 112)
ax_a.set_xticks([-100, -50, 0, 50, 100], ["100", "50", "0", "50", "100"])
ax_a.set_xlabel("Reader answers (%)")
ax_a.text(-50, y_head["vqarad"], "Wrong with original", ha="center", va="center", fontsize=LEGEND_PT, color=PAL["ink"])
ax_a.text(50, y_head["vqarad"], "Right with original", ha="center", va="center", fontsize=LEGEND_PT, color=PAL["ink"])
fig.legend(handles=[Patch(color=C_WRONG_BOTH, label="Wrong with original and blank images"),
                    Patch(color=C_WRONG_IMG, label="Wrong only with original image"),
                    Patch(color=C_RIGHT_IMG, label="Right only with original image"),
                    Patch(color=C_RIGHT_BOTH, label="Right with original and blank images")],
           loc="lower left", bbox_to_anchor=(0.205, 0.0), ncol=2, handlelength=1.1, handleheight=0.8,
           handletextpad=0.4, columnspacing=1.2, labelspacing=0.3, fontsize=LEGEND_PT)

ax_b.set_xlim(0, 123)
ax_b.set_xticks([0, 50, 100])
ax_b.set_xlabel("Matching blank-image\nmajority (%)")
ax_b.text(103, y_head["vqarad"], "n", ha="left", va="center", fontsize=ANNOT_PT, color=PAL["neutral_dark"])
ax_b.axvline(50, color=PAL["neutral_light"], lw=0.6, zorder=0)

add_panel_label(ax_a, "a", x_offset_pt=-2, y_offset_pt=1)
add_panel_label(ax_b, "b", x_offset_pt=-2, y_offset_pt=1)
print(save(fig, "fig4_prior"))
