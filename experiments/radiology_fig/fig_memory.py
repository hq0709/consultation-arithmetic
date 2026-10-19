"""Figure: reference-case memory. Each reader receives three retrieved reference cases (closed
question-answer pairs from training images), with their images or as text only; shared, every reader
receives the same three; private, reader i receives set i.

a  single-reader accuracy: no references, shared references with images, shared references as text
b  generic panel of 9, majority-vote accuracy
c  error correlation phi of the 9 readers
Item bootstrap 95% CI. Data: results/imaging_v2_M.json (experiments/imaging_v2_analysis.py M).
"""
import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from rf_style import (MM, PAL, TEXT_W_MM, LEGEND_PT, TICK_PT, BASE_PT, MODEL, SET_LABEL, RESULTS,
                      setup, save, add_panel_label)

setup()
M = json.loads((RESULTS / "imaging_v2_M.json").read_text())
TRIO = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001"]
SETS2 = ["vqarad", "slake"]
# condition -> (marker, filled, vertical offset)
COND = {"none": ("o", "gray", 0.32), "shared-image": ("o", True, 0.16), "private-image": ("^", True, 0.0),
        "shared-text": ("o", False, -0.16), "private-text": ("^", False, -0.32)}
LAB = {"none": "No references", "shared-image": "Shared, with images", "private-image": "Private, with images",
       "shared-text": "Shared, text only", "private-text": "Private, text only"}

GAP = 1.6
ypos = {(s, m): -(si * (len(TRIO) + GAP) + mi) for si, s in enumerate(SETS2) for mi, m in enumerate(TRIO)}
y_head = {s: -(si * (len(TRIO) + GAP)) + 0.95 for si, s in enumerate(SETS2)}

fig = plt.figure(figsize=(TEXT_W_MM * MM, 70 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], left=0.205, right=0.975, bottom=0.27, top=0.93, wspace=0.14)
axes = [fig.add_subplot(gs[0, 0])]
axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]


def point(ax, y, ci, color, cond):
    mk, filled, dy = COND[cond]
    c = PAL["neutral_dark"] if filled == "gray" else color
    est, lo, hi = ci
    ax.plot([lo, hi], [y + dy, y + dy], color=c, lw=0.8, solid_capstyle="butt", zorder=3)
    ax.plot([est], [y + dy], marker=mk, ms=3.2, color=c, mfc=c if filled in (True, "gray") else "white", mec=c,
            mew=0.9, lw=0, zorder=4)


for (s, m), y in ypos.items():
    d = M.get(f"{m}|{s}") or {}
    col = MODEL[m]["color"]
    for cond, key in (("none", "single_none"), ("shared-image", "single_shared-image"), ("shared-text", "single_shared-text")):
        if d.get(key):
            point(axes[0], y, d[key], col, cond)
    for cond in COND:
        k1, k2 = ("panel_none", "phi_none") if cond == "none" else (f"panel_{cond}", f"phi_{cond}")
        if d.get(k1):
            point(axes[1], y, d[k1], col, cond)
        if d.get(k2):
            point(axes[2], y, d[k2], col, cond)

axes[0].set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
axes[0].tick_params(axis="y", length=0, pad=3)
for s in SETS2:
    axes[0].text(-0.012, y_head[s], SET_LABEL[s], transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                 fontsize=BASE_PT, fontweight="bold", color=PAL["ink"])
lo_y = min(ypos.values()) - 0.8
for i, ax in enumerate(axes):
    if i:
        plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(axis="y", length=0)
    ax.set_ylim(lo_y, max(y_head.values()) + 0.6)
    ax.spines["left"].set_visible(False)
    ax.axhline(y_head[SETS2[1]] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)
axes[0].set_xlim(50, 95); axes[0].set_xlabel("Single-reader accuracy (%)")
axes[1].set_xlim(50, 95); axes[1].set_xlabel("Panel accuracy (%)")
axes[2].set_xlim(0.5, 1.0); axes[2].set_xlabel("Error correlation, $\\phi$")
fig.legend(handles=[Line2D([], [], marker=COND[c][0], ms=3.4, lw=0, color=PAL["neutral_dark"] if c == "none" else PAL["ink"],
                           mfc="white" if COND[c][1] is False else (PAL["neutral_dark"] if c == "none" else PAL["ink"]),
                           label=LAB[c]) for c in COND],
           loc="lower center", bbox_to_anchor=(0.58, 0.0), ncol=3, handletextpad=0.1, columnspacing=1.2,
           labelspacing=0.3, fontsize=LEGEND_PT)
for ax, lab in zip(axes, "abc"):
    add_panel_label(ax, lab, x_offset_pt=-2, y_offset_pt=1)
print(save(fig, "fig_memory"))
