"""Figure 5: readers use the queried region unevenly, and half of their answers do not follow the image.

Rows: API models (readers of the independent panel of 9; reader-level rates averaged within item)
and open-weight models (one forward pass per item).
a  SLAKE-target: share of answers that change when a mid-gray box covers the queried organ or lesion
   (filled) or an equal-size box elsewhere in the body (open)
b  region dependence = target flip rate minus control flip rate (paired item bootstrap 95% CI)
c  SLAKE-pairs (the same question on two images with opposite reference answers): both answers
   correct (filled) and the two answers differ (open). A reader that ignores the image scores 0 on both.
Data: results/imaging_v2_A.json, results/imaging_v2_D.json (experiments/imaging_v2_analysis.py A, D).
"""
import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from rf_style import (MM, PAL, TEXT_W_MM, LEGEND_PT, TICK_PT, BASE_PT, MODEL, RESULTS, setup, save,
                      add_panel_label, minus)

setup()
A = json.loads((RESULTS / "imaging_v2_A.json").read_text())
D = json.loads((RESULTS / "imaging_v2_D.json").read_text())
API = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001"]
OW = ["medgemma-4b", "qwen3-vl-8b", "qwen2-vl-7b", "huatuogpt-vision-7b", "llava-1.5-7b"]
OW_LABEL = {"medgemma-4b": "MedGemma 4B", "qwen3-vl-8b": "Qwen3-VL 8B", "qwen2-vl-7b": "Qwen2-VL 7B",
            "huatuogpt-vision-7b": "HuatuoGPT-Vision 7B", "llava-1.5-7b": "LLaVA-1.5 7B"}
OW_STYLE = {"color": PAL["neutral_dark"], "marker": "o"}
GROUPS = [("API models", API), ("Open-weight models", OW)]
GAP = 1.6
ypos, y_head, start = {}, {}, 0.0
for g, ms in GROUPS:
    y_head[g] = start + 0.95
    for i, m in enumerate(ms):
        ypos[m] = start - i
    start -= len(ms) + GAP


def rates(m):
    """-> (target flip, control flip, region dependence, both correct, answers differ), each [est, lo, hi]."""
    if m in API:
        t = A.get(f"{m}|slake_target") or {}; p = A.get(f"{m}|slake_pairs") or {}
        return (t.get("panel9_occ_target_reader_flip"), t.get("panel9_occ_control_reader_flip"),
                t.get("panel9_region_dependence"), p.get("single_both_correct"), p.get("single_answers_differ"))
    t = D.get(f"{m}|slake_target") or {}; p = D.get(f"{m}|slake_pairs") or {}
    return (t.get("flip_target"), t.get("flip_control"), t.get("region_dependence_flip"),
            p.get("both_correct"), p.get("answers_differ"))


fig = plt.figure(figsize=(TEXT_W_MM * MM, 68 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.78, 1.12], left=0.25, right=0.965, bottom=0.2, top=0.915, wspace=0.2)
axes = [fig.add_subplot(gs[0, 0])]
axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
ax_a, ax_b, ax_c = axes


def hbar(ax, yy, ci, st, filled=True, dy=0.0):
    est, lo, hi = ci
    ax.plot([lo, hi], [yy + dy, yy + dy], color=st["color"], lw=0.9, solid_capstyle="butt", zorder=3)
    ax.plot([est], [yy + dy], marker=st["marker"], ms=3.4, color=st["color"], mfc=st["color"] if filled else "white",
            mec=st["color"], mew=0.9, lw=0, zorder=4)


for m, yy in ypos.items():
    st = MODEL.get(m) or OW_STYLE
    tf, cf, rd, bc, ad = rates(m)
    if tf and cf:
        ax_a.plot([cf[0], tf[0]], [yy, yy], color=st["color"], lw=0.9, alpha=0.45, zorder=2)
        hbar(ax_a, yy, tf, st, dy=0.17)
        hbar(ax_a, yy, cf, st, filled=False, dy=-0.17)
    if rd:
        hbar(ax_b, yy, rd, st)
    if bc:
        hbar(ax_c, yy, bc, st, dy=0.17)
    if ad:
        hbar(ax_c, yy, ad, st, filled=False, dy=-0.17)

labels = {m: (MODEL[m]["label"] if m in MODEL else OW_LABEL[m]) for m in ypos}
ax_a.set_yticks(list(ypos.values()), [labels[m] for m in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for g, _ in GROUPS:
    ax_a.text(-0.012, y_head[g], g, transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=BASE_PT, fontweight="bold", color=PAL["ink"])
lo_y = min(ypos.values()) - 0.8
for ax in axes:
    if ax is not ax_a:
        plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(axis="y", length=0)
    ax.set_ylim(lo_y, max(y_head.values()) + 0.6)
    ax.spines["left"].set_visible(False)
    ax.axhline(y_head["Open-weight models"] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)

top = y_head["API models"] - 0.55
ax_a.set_xlim(0, 52); ax_a.set_xlabel("Answers changed\nby occlusion (%)")
ax_b.plot([0, 0], [lo_y, top], color=PAL["neutral_mid"], lw=0.7, ls=(0, (3, 2)), zorder=1)
ax_b.set_xlim(-6, 32); ax_b.set_xlabel("Region dependence,\ntarget − control (pp)")
ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: minus(f"{v:+.0f}") if v else "0"))
ax_c.set_xlim(0, 100); ax_c.set_xlabel("Same question,\ndifferent image (%)")
for ax, h in ((ax_a, [("Target", True), ("Control", False)]),
              (ax_c, [("Both correct", True), ("Differ", False)])):
    ax.legend(handles=[Line2D([], [], marker="o", ms=3.4, lw=0, color=PAL["ink"], mfc=PAL["ink"] if f else "white",
                              label=l) for l, f in h],
              loc="center left", bbox_to_anchor=(-0.02, y_head["API models"]), bbox_transform=ax.get_yaxis_transform(),
              ncol=2, handletextpad=0.0, columnspacing=0.5, borderaxespad=0.0, fontsize=LEGEND_PT)
for ax, lab in zip(axes, "abc"):
    add_panel_label(ax, lab, x_offset_pt=-2, y_offset_pt=1)
print(save(fig, "fig5_region"))
