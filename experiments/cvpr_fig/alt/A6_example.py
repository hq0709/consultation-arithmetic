"""A6_example — Fig. 8 of the CVPR paper (qualitative), full width (174.6 mm).

Claim: the perception module decides what the reasoning modules can know: with the Gemini 3.7 Flash
report every reasoning module answers correctly, with the other two reports none does.
Layout: a table. Columns are the inputs: the image (VQA-RAD vqarad/test-261, "is there biliary duct
dilation?", reference Yes; bottom caption strip of the MedPix file cropped) and the slot-0
query-agnostic report of each perception module (verbatim excerpts; the finding that answers the
question highlighted). Rows are the reasoning modules (logo and name); each cell shows the 9 agents,
filled = correct, open = wrong: self-consistency agents reading the image, or agents each reading its
own report from that perception module.
Data: results/cvpr_items.json (reports and every agent's answer). Logos: experiments/cvpr_fig/logos.py
(lobe-icons, MIT). Font: Comic Sans MS 2.10 (experiments/cvpr_fig/fonts). Nature-figure skill, Python backend.
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
from cf_style import (MM, MODEL, LEGEND_PT, ANNOT_PT)                            # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment           # noqa: E402
from example_kit import (STYLE as K, use_comic, runs, pill, agents, picture, paragraph, square, tw, avatar, qtag,
                         hline)                                                    # noqa: E402

import json                                                                       # noqa: E402
import numpy as np                                                                # noqa: E402
from PIL import Image                                                             # noqa: E402

use_comic()                 # the qualitative figures are set in Comic Sans MS (author's choice)
REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "A6_example"), str(QA / "A6_example")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 61.8

Q = "vqarad/test-261"
v = json.loads((REPO / "results" / "cvpr_items.json").read_text())["reports"][Q]
REASON = ["gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4-5-20251001"]
PERC = ["gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4-5-20251001"]
# verbatim excerpts of the slot-0 reports (ellipses mark omitted text); True marks the finding that answers the question
EXC = {"gemini-3.7-flash": [("… Findings: … diffuse, multifocal, saccular and cystic", False),
                            ("dilatation of the intrahepatic bile ducts", True),
                            ("throughout the hepatic parenchyma. …", False)],
       "gpt-5-mini": [("… Findings: Large gas- and fluid-filled gastric and adjacent colonic loops in the left upper "
                       "quadrant. Both kidneys demonstrate symmetric cortical enhancement …", False)],
       "claude-haiku-4-5-20251001": [("… Large low-attenuation collection in the right upper quadrant, predominantly "
                                      "anterior to the right kidney … consistent with ascites or loculated fluid …", False)]}
for p in PERC:                               # every excerpt must be a sequence of verbatim pieces of the report
    flat = " ".join(v["report"][p].replace("*", "").replace("#", "").split())
    for piece in " ".join(s for s, _ in EXC[p]).replace("…", "|").split("|"):
        piece = " ".join(piece.split()).strip(" .")
        assert not piece or piece in flat, (p, piece)
gold = v["gold_letter"]
W, H = width_mm, height_mm
LX = 1.5                                             # left margin; the label column runs to x = 28.5
SI, CW, CG = 31.0, 35.0, 2.2                         # image side, card width, gutter (mm)
XI = 30.0
XC = [XI + SI + CG + i * (CW + CG) for i in range(3)]
assert XC[-1] + CW <= W - 1.5
CENTRES = [XI + SI / 2] + [x + CW / 2 for x in XC]
IMG_TOP = H - 9.6
IMG_BOT = IMG_TOP - SI
TAB_TOP, PITCH = IMG_BOT - 4.0, 5.4
ROWS = [TAB_TOP - PITCH * (k + 0.5) for k in range(3)]
assert ROWS[-1] - PITCH / 2 > 0.8

fig = plt.figure(figsize=(W * MM, H * MM))
over = fig.add_axes([0, 0, 1, 1])
over.set_xlim(0, W); over.set_ylim(0, H); over.axis("off")

# column heads: the image, then the three reports under one head
L, R = XC[0], XC[-1] + CW
over.text((L + R) / 2, H - 2.6, "Query-agnostic report written by each perception module", ha="center", va="center",
          fontsize=ANNOT_PT, fontweight="medium", color=K["text2"])
over.plot([L + 0.3, L + 0.3, R - 0.3, R - 0.3], [H - 5.2, H - 4.5, H - 4.5, H - 5.2], color=K["rule_strong"], lw=0.5,
          solid_joinstyle="miter")
over.text(XI + SI / 2, IMG_TOP + 2.2, "Image", ha="center", va="center", fontsize=ANNOT_PT, fontweight="medium",
          color=K["ink"])
for x, p in zip(XC, PERC):
    w = 3.6 + 1.3 + tw(MODEL[p]["label"], ANNOT_PT, "medium")
    x0 = x + CW / 2 - w / 2
    avatar(over, p, x0 + 1.8, IMG_TOP + 2.2, d=3.6)
    over.text(x0 + 4.9, IMG_TOP + 2.2, MODEL[p]["label"], ha="left", va="center", fontsize=ANNOT_PT, fontweight="medium",
              color=K["ink"])

# label column: question, reference answer, data set, dot key and row head
qtag(over, LX + 1.7, IMG_TOP - 1.75)
for k, line in enumerate(("Is there biliary", "duct dilation?")):
    over.text(LX + 4.6, IMG_TOP - 1.75 - 3.5 * k, line, ha="left", va="center", fontsize=LEGEND_PT, fontweight="medium",
              color=K["ink"])
runs(over, LX + 4.6, IMG_TOP - 9.3, [("Reference  ", {"size": ANNOT_PT, "color": K["muted"]}),
                                     ("Yes", {"size": LEGEND_PT, "weight": "semibold", "color": K["ink"]})])
over.text(LX + 4.6, IMG_TOP - 12.6, "VQA-RAD · CT", ha="left", va="center", fontsize=ANNOT_PT, color=K["muted"])
over.text(LX, IMG_BOT - 2.1, "Reasoning module", ha="left", va="center", fontsize=ANNOT_PT, fontweight="medium",
          color=K["text2"])
# dot key, right-aligned on the same head line as the row head
note = "one dot per agent"
x = XC[-1] + CW - tw(note, ANNOT_PT)
over.text(x, IMG_BOT - 2.1, note, ha="left", va="center", fontsize=ANNOT_PT, color=K["muted"])
for f, lab in ((False, "wrong"), (True, "correct")):
    x -= 3.4 + tw(lab, ANNOT_PT)
    agents(over, x, IMG_BOT - 2.1, [f], K["text2"], d=1.7)
    over.text(x + 1.9, IMG_BOT - 2.1, lab, ha="left", va="center", fontsize=ANNOT_PT, color=K["text2"])
    x -= 1.6

# the image and the three report cards
with Image.open(REPO / v["image"]) as im:
    arr = np.asarray(im.convert("RGB"))
arr, _ = square(arr[: int(arr.shape[0] * 0.955)])           # crop the uploader caption strip, pad to a square
ax_img = picture(fig, W, H, XI, IMG_BOT, SI, arr, edge="none", radius=0.042)
for x, p in zip(XC, PERC):
    pill(over, x, IMG_BOT, CW, SI, fc=K["card"], r=1.3, zorder=2)
    over.plot([x + 1.3, x + CW - 1.3], [IMG_TOP - 0.25, IMG_TOP - 0.25], color=MODEL[p]["color"], lw=1.6,
              solid_capstyle="round", zorder=3)
    last = paragraph(over, x + 1.8, IMG_TOP - 2.4, CW - 3.6, EXC[p], size=7.3, color="#1F2937", hl_color=K["hl"])
    assert last > IMG_BOT + 1.4, (p, last)

# reasoning modules: one row each, 9 agents per cell
for k in range(4):
    hline(over, LX, XC[-1] + CW, TAB_TOP - k * PITCH, color=K["rule_strong"] if k in (0, 3) else K["rule"])
for r, y in zip(REASON, ROWS):
    avatar(over, r, LX + 2.1, y)
    over.text(LX + 5.2, y, MODEL[r]["label"], ha="left", va="center", fontsize=ANNOT_PT, fontweight="medium",
              color=K["ink"])
    for cx, key in zip(CENTRES, ["image"] + PERC):
        agents(over, cx, y, [a == gold for a in v["answers"][f"{r}|{key}"]], MODEL[r]["color"], pitch=2.3, d=1.7)

require_matplotlib_panel_alignment(fig, axes=[ax_img], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg", dpi=600)
fig.savefig(BASE + ".pdf", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
