"""A7_bump — candidate for Fig. 7, one column (83.34 mm): bump chart of the matched contrasts.

Claim: the ordering of shared components by how much they correlate two agents is stable across data
sets (model, then perception module, then report); only the reasoning module moves up on
MedXpertQA-MM. Rank 1 = largest reduction in error correlation; numbers, mean change over models.
In-context examples were run on VQA-RAD and SLAKE only.
Data: results/imaging_v2_C.json.
Nature-figure skill, Python backend.
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
from cf_style import (MM, PAL, SETS, SET_LABEL, TICK_PT, ANNOT_PT, LEGEND_PT, load)   # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment                # noqa: E402

import numpy as np                                                                     # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "A7_bump"), str(QA / "A7_bump")

width_mm = 83.34            # final printed width: CVPR \columnwidth
height_mm = 66

C = load("C")
COMP = [("model", "Model"), ("perceiver", "Perception module"), ("report", "Report"), ("reasoner", "Reasoning module"),
        ("reference cases", "In-context examples"), ("rendering", "Contrast adjustment"), ("role", "Role")]
COLOR = {"model": PAL["ink"], "reasoner": PAL["neg_dark"]}
HEAD = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM"}
rank, val = {}, {}
for si, s in enumerate(SETS):
    con = C[s]["contrasts"]
    present = [(k, con[k]["mean"][0]) for k, _ in COMP if k in con and con[k].get("mean")]
    for r, (k, v) in enumerate(sorted(present, key=lambda kv: kv[1]), 1):
        rank[(k, si)], val[(k, si)] = r, v

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
ax = fig.add_axes([0.37, 0.09, 0.56, 0.76])
ax.set_xlim(-0.3, 2.3)
ax.set_ylim(7.6, 0.4)
MS = 15.5                                     # circle diameter (points)
R_PX = MS / 2 / 72 * fig.dpi                  # circle radius in display pixels


def gap_segment(p0, p1):
    """Segment between two circle centres, cut back to the circle edges (display space)."""
    a, b = ax.transData.transform(p0), ax.transData.transform(p1)
    d = b - a
    n = np.hypot(*d)
    a2, b2 = a + d / n * R_PX, b - d / n * R_PX
    inv = ax.transData.inverted()
    return inv.transform(a2), inv.transform(b2)


for k, lab in COMP:
    pts = [(si, rank[(k, si)]) for si in range(3) if (k, si) in rank]
    col = COLOR.get(k, PAL["neutral_mid"])
    lw = 1.6 if k in COLOR else 1.0
    for p0, p1 in zip(pts[:-1], pts[1:]):
        (x0, y0), (x1, y1) = gap_segment(p0, p1)
        ax.plot([x0, x1], [y0, y1], color=col, lw=lw, zorder=2, solid_capstyle="butt")
    for si, r in pts:
        ax.plot([si], [r], ls="none", marker="o", ms=MS, mfc="white", mec=col, mew=lw, zorder=3)
        ax.text(si, r, f"{-val[(k, si)]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.5, color=PAL["ink"],
                zorder=4)
    ax.text(-0.33, pts[0][1], lab, ha="right", va="center", fontsize=TICK_PT,
            color=col if k in COLOR else PAL["ink"], fontweight="bold" if k in COLOR else "normal")
for si, s in enumerate(SETS):
    ax.text(si, 0.05, HEAD[s], ha="center", va="bottom", fontsize=ANNOT_PT)
ax.axis("off")
fig.text(0.5, 0.012, r"Circles: $|\Delta\phi|$, mean over models; rank 1 = largest reduction", ha="center", va="bottom",
         fontsize=ANNOT_PT, color=PAL["neutral_dark"])

require_matplotlib_panel_alignment(fig, axes=[ax], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
