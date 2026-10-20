"""S1a_scaling_accuracy — candidate for the agent-scaling display (Sec. 4.3), full width (174.6 mm).

Claim: independent agents of the same accuracy would gain 8 to 22 pp from a majority of nine; agents
of one VLM gain close to nothing, on every data set.
Encoding: one panel per data set (VQA-RAD, SLAKE, MedXpertQA-MM; 250 items each). x, number of agents
k in the vote (the first k agents of the Independent MAS of 9, original images); y, majority accuracy.
Solid lines with markers, observed; dashed lines, binomial prediction for k independent agents with
the same mean accuracy (yes/no sets only; MedXpertQA-MM has five options and a plurality vote).
Colours and markers identify the model.
Data: results/cvpr_nmi.json ("scaling"). Nature-figure skill, Python backend.
"""
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── MANDATORY (nature-figure api.md): sans-serif stack + editable SVG/PDF text ──
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": False,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans",
    "mathtext.cal": "Liberation Sans",
    "mathtext.tt": "Liberation Sans",
    "axes.labelcolor": "#1F1F1F",
    "xtick.color": "#1F1F1F",
    "ytick.color": "#1F1F1F",
    "text.color": "#1F1F1F",
    "axes.edgecolor": "#1F1F1F",
})

ALT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ALT.parent))
sys.path.insert(0, str(pathlib.Path.home() / ".claude" / "skills" / "nature-figure" / "scripts"))
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, ms, header)   # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment             # noqa: E402
sys.path.insert(0, str(ALT))
from s1_common import SETS, SET_LABEL, YESNO, style, models, cell, ROOT          # noqa: E402

import math                                                                     # noqa: E402
from matplotlib.lines import Line2D                                             # noqa: E402

OUT = ROOT / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
OUT.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
BASE, QBASE = str(OUT / "S1a_scaling_accuracy"), str(QA / "S1a_scaling_accuracy")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 68.5
W, H = width_mm, height_mm
L, R, GUT = 14.0, 3.0, 9.0                      # left margin, right margin, gutter (mm)
PW = (W - L - R - 2 * GUT) / 3                  # panel width
BOT, TOP = 12.5, 19.0                           # bottom margin; top margin holds two legend rows and titles
PH = H - BOT - TOP
MS_ = models()

fig = plt.figure(figsize=(W * MM, H * MM))
axes = []
for j, s in enumerate(SETS):
    ax = fig.add_axes([(L + j * (PW + GUT)) / W, BOT / H, PW / W, PH / H])
    axes.append(ax)
    vals = []
    for m in MS_:
        c = cell(m, s)
        if not c:
            continue
        st = style(m)
        if c["binomial"]:
            ax.plot(c["k"], c["binomial"], color=st["color"], lw=1.0, ls=(0, (3.2, 1.6)), alpha=0.7, zorder=2,
                    solid_capstyle="butt")
            vals += c["binomial"]
        ax.plot(c["k"], c["observed"], color=st["color"], lw=1.3, zorder=3, marker=st["marker"],
                ms=ms(st["marker"], 3.8), mfc=st["color"], mec="white", mew=0.5)
        vals += c["observed"]
    lo = math.floor((min(vals) - 3) / 10) * 10
    hi = min(100, math.ceil((max(vals) + 3) / 10) * 10)
    ax.set_ylim(lo, hi)
    ax.set_xlim(0.4, 9.6)
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_xlabel("Number of agents", fontsize=BASE_PT)
    ax.tick_params(labelsize=TICK_PT)
    ax.grid(axis="y", color=PAL["neutral_faint"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if s not in YESNO:
        ax.text(0.03, 0.80, "Five options, plurality vote:\nno independent-agent reference",
                transform=ax.transAxes, ha="left", va="top", fontsize=ANNOT_PT, color=PAL["neutral_dark"], zorder=4)
axes[0].set_ylabel("Majority accuracy (%)", fontsize=BASE_PT)
for ax, letter, s in zip(axes, "abc", SETS):
    header(ax, letter, SET_LABEL[s], y_offset_pt=7)

handles = [Line2D([], [], color=style(m)["color"], lw=1.3, marker=style(m)["marker"], ms=ms(style(m)["marker"], 3.8),
                  mfc=style(m)["color"], mec="white", mew=0.5, label=style(m)["label"]) for m in MS_]
styles = [Line2D([], [], color=PAL["neutral_dark"], lw=1.3, label="Observed majority of the first $k$ agents"),
          Line2D([], [], color=PAL["neutral_dark"], lw=1.0, ls=(0, (3.2, 1.6)),
                 label="Prediction for $k$ independent agents of the same accuracy (binomial)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(handles), 5),
           fontsize=LEGEND_PT, frameon=False, handlelength=2.4, columnspacing=1.6, borderaxespad=0.2)
fig.legend(handles=styles, loc="upper center", bbox_to_anchor=(0.5, 1 - 4.6 / H * (1 + (len(handles) - 1) // 5)),
           ncol=2, fontsize=LEGEND_PT, frameon=False, handlelength=2.4, columnspacing=2.0, borderaxespad=0.2)

require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=["a", "b", "c"],
                                   row_groups=[{"id": "sets", "panels": ["a", "b", "c"]}],
                                   json_out=QBASE + ".alignment.json", overlay_svg=QBASE + ".alignment.svg",
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg", dpi=600)
fig.savefig(BASE + ".pdf", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
