"""A5_example — Fig. 5 of the CVPR paper (qualitative), full width (174.6 mm).

Claim: answers of Gemini 3.7 Flash follow the queried organ; Claude Haiku 4.5 gives the same answer
when the liver is covered and to two images with opposite reference answers.
Layout: a table. Rows are models (logo and name); columns are the images as sent to the models.
(a) SLAKE-target item slakeT/test-12078 (CT): the original image (dashed: target box, red; control
    box, blue), the target-occluded and the control-occluded image (mid-gray box); in each cell the
    9 agents of the model's Independent MAS, filled = answer Yes, open = answer No.
(b) SLAKE-pairs pair slakeP/005: the same question on two images with opposite reference answers;
    single-agent answers with correct / wrong marks.
Data: results/cvpr_items.json (every agent's answer, experiments/cvpr_fig/extract_items.py); images
rendered with common/llm.py::_render. Logos: experiments/cvpr_fig/logos.py (lobe-icons, MIT).
Font: Comic Sans MS 2.10 (experiments/cvpr_fig/fonts). Nature-figure skill, Python backend.
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
from cf_style import (MM, MODEL, BASE_PT, LEGEND_PT, ANNOT_PT, PANEL_LABEL_PT)   # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment           # noqa: E402
from example_kit import (STYLE as K, use_comic, runs, agents, check, cross, picture, tw, avatar, qtag,
                         hline)                                                    # noqa: E402

import json                                                                       # noqa: E402
import numpy as np                                                                # noqa: E402
from PIL import Image                                                             # noqa: E402
from matplotlib.patches import Rectangle                                          # noqa: E402

use_comic()                 # the qualitative figures are set in Comic Sans MS (author's choice)
REPO = ALT.parents[2]
sys.path.insert(0, str(REPO))
from common.llm import _render                                                    # noqa: E402

OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "A5_example"), str(QA / "A5_example")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 56.0

D = json.loads((REPO / "results" / "cvpr_items.json").read_text())
occ = D["occlusion"]["slakeT/test-12078"]
pair = D["pairs"]["slakeP/005"]
assert occ["stem"] == pair["stem"]
ORDER = ["gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4-5-20251001"]
W, H = width_mm, height_mm
LX = 1.5                                          # left margin; the label column runs to x = 28.5
S, G = 25.6, 2.2                                  # image side and gutter (mm)
XA = [30.0 + i * (S + G) for i in range(3)]       # panel a columns
XB = [XA[-1] + S + 5.6 + i * (S + G) for i in range(2)]   # panel b columns
assert XB[-1] + S <= W - 2.0
IMG_TOP = H - 9.6
IMG_BOT = IMG_TOP - S
TAB_TOP, PITCH = IMG_BOT - 2.2, 5.4               # table: top rule, row pitch
ROWS = [TAB_TOP - PITCH * (k + 0.5) for k in range(3)]
assert ROWS[-1] - PITCH / 2 > 0.8
assert tw("Claude Haiku 4.5", ANNOT_PT, "medium") + LX + 5.2 < XA[0] - 1.0
assert tw("Does the picture", LEGEND_PT, "medium") + LX + 4.6 < XA[0] - 1.0


def img(path, ops=()):
    with Image.open(REPO / path) as im:
        return np.asarray(_render(im.convert("RGB"), list(ops)))


fig = plt.figure(figsize=(W * MM, H * MM))
over = fig.add_axes([0, 0, 1, 1])          # millimetre layer for text, logos, rules and agent dots
over.set_xlim(0, W); over.set_ylim(0, H); over.axis("off")

# panel heads
for x0, letter, title, note in ((LX - 1.1, "a", "Region occlusion", "SLAKE-target · Independent MAS of 9"),
                                (XB[0] - 4.4, "b", "Paired images", "SLAKE-pairs · single agent")):
    over.text(x0, H - 2.7, letter, ha="left", va="center", fontsize=PANEL_LABEL_PT, fontweight="bold", color=K["ink"])
    end = runs(over, x0 + 4.0, H - 2.7, [(title, {"size": BASE_PT, "weight": "semibold", "color": K["ink"]}),
                                         ("   " + note, {"size": ANNOT_PT, "color": K["muted"]})])
    assert end < W - 1.0, end

# label column: the question, the reference answer and the dot key
qtag(over, LX + 1.7, IMG_TOP - 1.75)
for k, line in enumerate(("Does the picture", "contain liver?")):
    over.text(LX + 4.6, IMG_TOP - 1.75 - 3.5 * k, line, ha="left", va="center", fontsize=LEGEND_PT, fontweight="medium",
              color=K["ink"])
runs(over, LX + 4.6, IMG_TOP - 9.3, [("Reference  ", {"size": ANNOT_PT, "color": K["muted"]}),
                                     ("Yes", {"size": LEGEND_PT, "weight": "semibold", "color": K["ink"]})])
ky = IMG_BOT + 4.4
for x, f, lab in ((LX + 5.5, True, "Yes"), (LX + 14.6, False, "No")):
    agents(over, x, ky, [f], K["text2"], d=1.7)
    over.text(x + 1.9, ky, lab, ha="left", va="center", fontsize=ANNOT_PT, color=K["text2"])
over.text(LX + 4.6, ky - 3.2, "one dot per agent", ha="left", va="center", fontsize=ANNOT_PT, color=K["muted"])

# table rules and model rows
for x0, x1 in ((LX, XA[-1] + S), (XB[0], XB[-1] + S)):
    for k in range(4):
        hline(over, x0, x1, TAB_TOP - k * PITCH, color=K["rule_strong"] if k in (0, 3) else K["rule"])
for m, y in zip(ORDER, ROWS):
    avatar(over, m, LX + 2.1, y)
    over.text(LX + 5.2, y, MODEL[m]["label"], ha="left", va="center", fontsize=ANNOT_PT, fontweight="medium",
              color=K["ink"])

# panel a: the images as sent; detection-style boxes on the original
tb, cb = occ["target_box"], occ["control_box"]
occ_op = lambda b: [f"occ={','.join(f'{v:.0f}' for v in b)}"]
ims = [img(occ["image"]), img(occ["image"], occ_op(tb)), img(occ["image"], occ_op(cb))]
axes = []
for x, im, head in zip(XA, ims, ["Original", "Target occluded", "Control occluded"]):
    axes.append(picture(fig, W, H, x, IMG_BOT, S, im, edge="none", radius=0.042))
    over.text(x + S / 2, IMG_TOP + 2.2, head, ha="center", va="center", fontsize=ANNOT_PT, fontweight="medium",
              color=K["ink"])
px = ims[0].shape[1] / S                               # image pixels per printed mm
for key, bx in (("target", tb), ("control", cb)):
    col = K[key]
    axes[0].add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=col, lw=0.8, zorder=6))
    axes[0].add_patch(Rectangle((bx[0] - 0.2 * px, bx[1] - 2.8 * px), (tw(key, 6.8, "semibold") + 1.2) * px, 2.8 * px,
                                fc=col, ec="none", zorder=6))
    axes[0].text(bx[0] + 0.4 * px, bx[1] - 1.4 * px, key, ha="left", va="center", fontsize=6.8, fontweight="semibold",
                 color="white", zorder=7)
yes = occ["gold_letter"]
assert occ["options"][yes] == "Yes"
for m, y in zip(ORDER, ROWS):
    for x, cond in zip(XA, ("real", "occ_target", "occ_control")):
        agents(over, x + S / 2, y, [a == yes for a in occ["models"][m]["answers"][cond]], MODEL[m]["color"],
               pitch=2.3, d=1.7)

# panel b: the pair and one answer per model
pax = []
for x, key in zip(XB, ("a", "b")):
    pax.append(picture(fig, W, H, x, IMG_BOT, S, img(pair[key]["image"]), edge="none", radius=0.042))
    g = pair[key]["gold"]
    w = tw("Reference  ", ANNOT_PT) + tw(g, ANNOT_PT, "semibold")
    runs(over, x + S / 2 - w / 2, IMG_TOP + 2.2, [("Reference  ", {"size": ANNOT_PT, "color": K["muted"]}),
                                                  (g, {"size": ANNOT_PT, "weight": "semibold", "color": K["ink"]})])
    for m, y in zip(ORDER, ROWS):
        ans = pair["models"][m][key]
        ok = ans == g
        over.text(x + S / 2 + 0.3, y, ans, ha="right", va="center", fontsize=LEGEND_PT, fontweight="medium",
                  color=K["ink"] if ok else K["wrong"])
        (check if ok else cross)(over, x + S / 2 + 2.8, y, 0.85, K["right"] if ok else K["wrong"], lw=1.0)

require_matplotlib_panel_alignment(fig, axes=axes + pax, panel_ids=["a1", "a2", "a3", "b1", "b2"],
                                   row_groups=[{"id": "a-images", "panels": ["a1", "a2", "a3"]},
                                               {"id": "b-images", "panels": ["b1", "b2"]}],
                                   json_out=QBASE + ".alignment.json", overlay_svg=QBASE + ".alignment.svg",
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg", dpi=600)
fig.savefig(BASE + ".pdf", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
