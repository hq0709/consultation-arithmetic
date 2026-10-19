"""Figure 6: seeing and judging. Readers answer from findings reports written by a perceiver model
that did not see the question, without the image.

Rows: the reasoner model (the reader that chooses the answer) within each data set.
a  mean reader accuracy; columns: the perceiver that wrote the reports (each of the 9 readers reads its
   own report), and the reasoner reading the image itself
b  error correlation phi of the 9 readers: reading the image itself, each reading its own report from
   the same model, and all reading one shared report (item bootstrap 95% CI)
Data: results/imaging_v2_B.json (experiments/imaging_v2_analysis.py B).
"""
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from rf_style import (MM, PAL, TEXT_W_MM, ANNOT_PT, LEGEND_PT, TICK_PT, BASE_PT, MODEL, SETS, SET_LABEL, RESULTS,
                      setup, save, add_panel_label)

setup()
B = json.loads((RESULTS / "imaging_v2_B.json").read_text())
TRIO = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001"]
# sequential blue (dataviz reference ramp, steps 100-600)
CMAP = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#184f95"])

GAP = 1.6
ypos = {(s, r): -(si * (len(TRIO) + GAP) + ri) for si, s in enumerate(SETS) for ri, r in enumerate(TRIO)}
y_head = {s: -(si * (len(TRIO) + GAP)) + 0.95 for si, s in enumerate(SETS)}
COLS = TRIO + ["image"]
XC = {c: i + (0.25 if c == "image" else 0) for i, c in enumerate(COLS)}         # gap before the image column

fig = plt.figure(figsize=(TEXT_W_MM * MM, 86 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=0.205, right=0.985, bottom=0.12, top=0.79, wspace=0.08)
ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

for s in SETS:
    mat = B.get(s) or {}
    vals = {}
    for r in TRIO:
        row = mat.get(r) or {}
        for c in COLS:
            e = row.get("direct") if c == "image" else row.get(f"{c}|own")
            if e and e.get("reader_acc"):
                vals[(r, c)] = e["reader_acc"][0]
    lo, hi = (min(vals.values()), max(vals.values())) if vals else (0, 1)
    for r in TRIO:
        y = ypos[(s, r)]
        for c in COLS:
            x = XC[c]
            v = vals.get((r, c))
            if v is None:
                ax_a.add_patch(plt.Rectangle((x - 0.47, y - 0.45), 0.94, 0.9, facecolor=PAL["neutral_light"], lw=0))
                ax_a.text(x, y, "–", ha="center", va="center", fontsize=ANNOT_PT, color=PAL["neutral_dark"])
                continue
            t = (v - lo) / (hi - lo) if hi > lo else 0.5
            ax_a.add_patch(plt.Rectangle((x - 0.47, y - 0.45), 0.94, 0.9, facecolor=CMAP(0.1 + 0.8 * t), lw=0))
            ax_a.text(x, y, f"{v:.1f}", ha="center", va="center", fontsize=ANNOT_PT,
                      color="white" if t > 0.55 else PAL["ink"], fontweight="bold" if c == r else "normal")
        ent = (mat.get(r) or {})
        for key, marker, filled, dy in (("direct", "s", True, 0.22), (f"{r}|own", "o", True, 0.0), (f"{r}|shared", "o", False, -0.22)):
            e = ent.get(key)
            if e and e.get("phi"):
                est, l_, h_ = e["phi"]
                col = MODEL[r]["color"]
                ax_b.plot([l_, h_], [y + dy, y + dy], color=col, lw=0.9, solid_capstyle="butt", zorder=3)
                ax_b.plot([est], [y + dy], marker=marker, ms=3.4, color=col, mfc=col if filled else "white", mec=col,
                          mew=0.9, lw=0, zorder=4)

ax_a.set_xlim(-0.55, XC["image"] + 0.55)
ax_a.set_xticks([XC[c] for c in COLS], ["GPT-5\nmini", "Gemini\n3.7 Flash", "Claude\nHaiku 4.5", "Image"], fontsize=TICK_PT)
ax_a.xaxis.tick_top(); ax_a.tick_params(axis="x", length=0, pad=2)
ax_a.text(1.0, 1.21, "Reports written by", transform=ax_a.get_xaxis_transform(), ha="center", va="bottom",
          fontsize=LEGEND_PT, color=PAL["neutral_dark"])
ax_a.plot([-0.45, 2.45], [1.195, 1.195], transform=ax_a.get_xaxis_transform(), color=PAL["neutral_mid"], lw=0.6,
          clip_on=False)
ax_a.set_yticks([ypos[k] for k in ypos], [MODEL[k[1]]["label"] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.012, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=BASE_PT, fontweight="bold", color=PAL["ink"])
for sp in ("left", "bottom", "top", "right"):
    ax_a.spines[sp].set_visible(False)
ax_a.set_xlabel("Reader accuracy (%)")
plt.setp(ax_b.get_yticklabels(), visible=False); ax_b.tick_params(axis="y", length=0)
ax_b.spines["left"].set_visible(False)
lo_y = min(ypos.values()) - 0.8
ax_a.set_ylim(lo_y, max(y_head.values()) + 0.6)
for s in SETS[1:]:
    ax_b.axhline(y_head[s] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)
ax_b.set_xlim(0.4, 1.02)
ax_b.set_xlabel("Error correlation, $\\phi$")
ax_b.legend(handles=[Line2D([], [], marker="s", ms=3.4, lw=0, color=PAL["ink"], label="Image"),
                     Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], label="Own report"),
                     Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], mfc="white", label="Shared report")],
            loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, handletextpad=0.1, labelspacing=0.25,
            borderaxespad=0.2, fontsize=LEGEND_PT)
add_panel_label(ax_a, "a", x=-0.36, y=1, x_offset_pt=0, y_offset_pt=34)
add_panel_label(ax_b, "b", x_offset_pt=-2, y_offset_pt=34)
print(save(fig, "fig6_report"))
