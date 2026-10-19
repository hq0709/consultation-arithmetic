"""Figure: open-weight models, one forward pass per item.

a  accuracy with the image (filled) and with a uniform gray image (open), item bootstrap 95% CI
b  visual reliance: |change in log-odds of "yes"| between image and gray image on yes/no items,
   KL divergence on multiple-option items (MedXpertQA-MM); mean and 95% CI, log scale
c  error correlation between two models, divided by its maximum for their error rates
   (phi / phi_max); the pair that shares the CLIP ViT-L/14-336 encoder is marked
Data: results/imaging_v2_D.json (experiments/imaging_v2_analysis.py D).
"""
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

from rf_style import (MM, PAL, TEXT_W_MM, LEGEND_PT, TICK_PT, BASE_PT, ANNOT_PT, SETS, SET_LABEL, RESULTS,
                      setup, save, add_panel_label)

setup()
D = json.loads((RESULTS / "imaging_v2_D.json").read_text())
OW = ["medgemma-4b", "qwen3-vl-8b", "qwen2-vl-7b", "huatuogpt-vision-7b", "llava-1.5-7b"]
OW_LABEL = {"medgemma-4b": "MedGemma 4B", "qwen3-vl-8b": "Qwen3-VL 8B", "qwen2-vl-7b": "Qwen2-VL 7B",
            "huatuogpt-vision-7b": "HuatuoGPT-Vision 7B", "llava-1.5-7b": "LLaVA-1.5 7B"}
INK, GRAYC = PAL["neutral_dark"], PAL["neutral_mid"]
CLIP_C = PAL["rose"]                                          # the shared-encoder pair
CHANCE = {"vqarad": 50, "slake": 50, "mxqa_mm_rad": 20}

GAP = 1.6
ypos = {(s, m): -(si * (len(OW) + GAP) + mi) for si, s in enumerate(SETS) for mi, m in enumerate(OW)}
y_head = {s: -(si * (len(OW) + GAP)) + 0.95 for si, s in enumerate(SETS)}

fig = plt.figure(figsize=(TEXT_W_MM * MM, 112 * MM))
gs = fig.add_gridspec(2, 2, height_ratios=[4.9, 1], left=0.235, right=0.975, bottom=0.085, top=0.955, wspace=0.12,
                      hspace=0.36)
ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a); ax_c = fig.add_subplot(gs[1, :])


def hbar(ax, y, ci, color, filled=True, dy=0.0, marker="o"):
    est, lo, hi = ci
    ax.plot([lo, hi], [y + dy, y + dy], color=color, lw=0.9, solid_capstyle="butt", zorder=3)
    ax.plot([est], [y + dy], marker=marker, ms=3.4, color=color, mfc=color if filled else "white", mec=color,
            mew=0.9, lw=0, zorder=4)


for (s, m), y in ypos.items():
    d = D.get(f"{m}|{s}") or {}
    if d.get("acc_real") and d.get("acc_blank"):
        ax_a.plot([d["acc_blank"][0], d["acc_real"][0]], [y, y], color=INK, lw=0.9, alpha=0.4, zorder=2)
        hbar(ax_a, y, d["acc_real"], INK, dy=0.17)
        hbar(ax_a, y, d["acc_blank"], INK, filled=False, dy=-0.17)
    if d.get("visual_reliance"):
        hbar(ax_b, y, d["visual_reliance"], INK)
for s in SETS:
    ys = [ypos[(s, m)] for m in OW]
    ax_a.plot([CHANCE[s]] * 2, [min(ys) - 0.5, max(ys) + 0.5], color=GRAYC, lw=0.7, ls=(0, (3, 2)), zorder=1)

ax_a.set_yticks([ypos[k] for k in ypos], [OW_LABEL[k[1]] for k in ypos], fontsize=TICK_PT)
ax_a.tick_params(axis="y", length=0, pad=3)
for s in SETS:
    ax_a.text(-0.012, y_head[s], SET_LABEL[s], transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=BASE_PT, fontweight="bold", color=PAL["ink"])
plt.setp(ax_b.get_yticklabels(), visible=False); ax_b.tick_params(axis="y", length=0)
lo_y = min(ypos.values()) - 0.8
for ax in (ax_a, ax_b):
    ax.set_ylim(lo_y, max(y_head.values()) + 0.6)
    ax.spines["left"].set_visible(False)
    for s in SETS[1:]:
        ax.axhline(y_head[s] + 0.75, color=PAL["neutral_light"], lw=0.6, zorder=0)
ax_a.set_xlim(10, 100); ax_a.set_xticks([20, 40, 60, 80]); ax_a.set_xlabel("Accuracy (%)")
ax_a.legend(handles=[Line2D([], [], marker="o", ms=3.4, lw=0, color=INK, label="Image"),
                     Line2D([], [], marker="o", ms=3.4, lw=0, color=INK, mfc="white", label="Gray image")],
            loc="center left", bbox_to_anchor=(-0.02, y_head["vqarad"]), bbox_transform=ax_a.get_yaxis_transform(),
            ncol=2, handletextpad=0.0, columnspacing=0.6, borderaxespad=0.0, fontsize=LEGEND_PT)
ax_b.set_xscale("log"); ax_b.set_xlim(0.01, 30)
ax_b.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
ax_b.xaxis.set_minor_locator(NullLocator())
ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_b.set_xlabel("Visual reliance")

# c: cross-model error correlation, one row per data set
rng = np.random.default_rng(3)
for si, s in enumerate(SETS):
    pts = [(v["phi_over_max"], v["shared_encoder"]) for k, v in D["cross_model_pairs"].items()
           if k.startswith(s + "|") and v.get("phi_over_max") is not None]
    for val, shared in pts:
        y = -si + (0 if shared else rng.uniform(-0.2, 0.2))
        ax_c.plot([val], [y], marker="D" if shared else "o", ms=4.2 if shared else 3.2,
                  color=CLIP_C if shared else GRAYC, mec=CLIP_C if shared else "white", mew=0.4, lw=0,
                  zorder=4 if shared else 3)
ax_c.set_yticks([0, -1, -2], [SET_LABEL[s] for s in SETS], fontsize=TICK_PT)
ax_c.tick_params(axis="y", length=0, pad=3)
ax_c.spines["left"].set_visible(False)
ax_c.set_ylim(-2.6, 0.6); ax_c.set_xlim(0, 0.8)
ax_c.set_xlabel("Error correlation between two models, $\\phi/\\phi_{\\max}$")
ax_c.legend(handles=[Line2D([], [], marker="D", ms=4.2, lw=0, color=CLIP_C, label="Shared CLIP encoder"),
                     Line2D([], [], marker="o", ms=3.2, lw=0, color=GRAYC, label="Different encoders")],
            loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.1, columnspacing=1.0, borderaxespad=0.1,
            fontsize=LEGEND_PT)

add_panel_label(ax_a, "a", x_offset_pt=-2, y_offset_pt=1)
add_panel_label(ax_b, "b", x_offset_pt=-2, y_offset_pt=1)
add_panel_label(ax_c, "c", x_offset_pt=-2, y_offset_pt=1)
print(save(fig, "fig_openweight"))
