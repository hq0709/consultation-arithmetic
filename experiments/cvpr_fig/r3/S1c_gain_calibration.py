"""S1c_gain_calibration — candidate for the agent-scaling display (Sec. 4.3), full width (174.6 mm).

Claim: independence predicts that a majority of k agents gains 7 to 22 pp over one agent; the
beta-binomial with the observed error correlation predicts the gains agents of one VLM actually show.
Encoding: one point per system and vote size k = 3, 5, 7, 9 (the first k agents of the Independent MAS
of 9 with original images; yes/no sets, 250 items each). x, predicted gain over one agent; y, observed
gain over one agent (pp). (a) binomial prediction for k independent agents with the same mean accuracy;
(b) beta-binomial prediction with the observed phi. Thin lines join a system's points from k = 3 to 9;
the large marker is k = 9. Filled markers, VQA-RAD; open markers, SLAKE. Colour and marker, model.
Dashed grey line, observed = predicted (legend). MedXpertQA-MM (five options, plurality vote) has no prediction.
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
from cf_style import (MM, PAL, BASE_PT, TICK_PT, LEGEND_PT, ANNOT_PT, ms, header, signed)   # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment                       # noqa: E402
sys.path.insert(0, str(ALT))
from s1_common import YESNO, SET_LABEL, style, models, cell, ROOT                         # noqa: E402

from matplotlib.lines import Line2D                                                       # noqa: E402

OUT = ROOT / "paper" / "cvpr" / "figure_candidates" / "round3"
QA = ALT.parent / "qa" / "r3"
OUT.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
BASE, QBASE = str(OUT / "S1c_gain_calibration"), str(QA / "S1c_gain_calibration")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 76.0
W, H = width_mm, height_mm
L, GUT, PW, BOT, TOP = 14.0, 16.0, 70.0, 12.5, 19.5
PH = H - BOT - TOP
MS_ = models()

pts = {"binomial": [], "betabinomial": []}            # (model, set, k, predicted gain, observed gain)
for m in MS_:
    for s in YESNO:
        c = cell(m, s)
        if not c or not c["binomial"]:
            continue
        o1 = c["observed"][0]
        for kind in pts:
            for k, obs, pred in zip(c["k"], c["observed"], c[kind]):
                if k > 1:
                    pts[kind].append((m, s, k, pred - o1, obs - o1))
ys = [p[4] for p in pts["binomial"]]
YL = (min(-3.0, min(ys) - 0.8), max(5.5, max(ys) + 0.8))
XL = {"binomial": (min(-3.0, min(p[3] for p in pts["binomial"]) - 1), max(p[3] for p in pts["binomial"]) + 2),
      "betabinomial": YL}

fig = plt.figure(figsize=(W * MM, H * MM))
axes = []
for j, (kind, title) in enumerate((("binomial", "Independent agents (binomial)"),
                                   ("betabinomial", r"Beta-binomial with the observed $\phi$"))):
    ax = fig.add_axes([(L + j * (PW + GUT)) / W, BOT / H, PW / W, PH / H])
    axes.append(ax)
    lo, hi = XL[kind]
    ax.set_xlim(lo, hi); ax.set_ylim(*YL)
    d0, d1 = max(lo, YL[0]), min(hi, YL[1])
    ax.plot([d0, d1], [d0, d1], color=PAL["neutral_mid"], lw=1.0, ls=(0, (3.2, 1.6)), zorder=1)
    ax.axhline(0, color=PAL["neutral_light"], lw=0.7, zorder=0)
    ax.axvline(0, color=PAL["neutral_light"], lw=0.7, zorder=0)
    for m in MS_:
        st = style(m)
        for s in YESNO:
            P = [p for p in pts[kind] if p[0] == m and p[1] == s]
            if not P:
                continue
            ax.plot([p[3] for p in P], [p[4] for p in P], color=st["color"], lw=0.7, alpha=0.6, zorder=2)
            for (_, _, k, x, y) in P:
                ax.plot([x], [y], marker=st["marker"], ms=ms(st["marker"], 4.6 if k == 9 else 2.8), lw=0,
                        color=st["color"], mfc=st["color"] if s == "vqarad" else "white", mec=st["color"],
                        mew=0.8, zorder=3 if k == 9 else 2.5)
    ax.set_xlabel("Predicted gain over one agent (pp)", fontsize=BASE_PT)
    ax.tick_params(labelsize=TICK_PT)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: signed(v)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: signed(v)))
    header(ax, "ab"[j], title, y_offset_pt=5)
axes[0].set_ylabel("Observed gain over one agent (pp)", fontsize=BASE_PT)
h_models = [Line2D([], [], lw=0, marker=style(m)["marker"], ms=ms(style(m)["marker"], 4.2), color=style(m)["color"],
                   mfc=style(m)["color"], mec=style(m)["color"], label=style(m)["label"]) for m in MS_]
g = PAL["neutral_dark"]
h_enc = [Line2D([], [], lw=0, marker="o", ms=4.2, mfc=g, mec=g, label=SET_LABEL["vqarad"] + " (filled)"),
         Line2D([], [], lw=0, marker="o", ms=4.2, mfc="white", mec=g, mew=0.8, label=SET_LABEL["slake"] + " (open)"),
         Line2D([], [], lw=0.7, color=g, marker="o", ms=2.8, mfc=g, mec=g, label="$k$ = 3, 5, 7"),
         Line2D([], [], lw=0, marker="o", ms=4.6, mfc=g, mec=g, label="$k$ = 9"),
         Line2D([], [], lw=1.0, color=PAL["neutral_mid"], ls=(0, (3.2, 1.6)), label="observed = predicted")]
fig.legend(handles=h_models, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(h_models), 5),
           fontsize=LEGEND_PT, frameon=False, handletextpad=0.4, columnspacing=1.8, borderaxespad=0.2)
fig.legend(handles=h_enc, loc="upper center", bbox_to_anchor=(0.5, 1 - 4.6 / H * (1 + (len(h_models) - 1) // 5)),
           ncol=5, fontsize=LEGEND_PT, frameon=False, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.2)

require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=["a", "b"],
                                   row_groups=[{"id": "calibration", "panels": ["a", "b"]}],
                                   json_out=QBASE + ".alignment.json", overlay_svg=QBASE + ".alignment.svg",
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg", dpi=600)
fig.savefig(BASE + ".pdf", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
