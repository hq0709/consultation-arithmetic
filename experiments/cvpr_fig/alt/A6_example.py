"""A6_example — candidate for Fig. 6 (qualitative), full width (174.6 mm).

Claim: the perception module decides what the reasoning modules can know: with the Gemini 3.7 Flash
report every reasoning module answers correctly, with the other two reports none does.
Item vqarad/test-261 ("is there biliary duct dilation?", reference Yes). Left: the image (bottom caption strip of the MedPix file cropped); below it,
agents of each model reading the image (self-consistency, 9 agents) who answered correctly. Right:
slot-0 query-agnostic reports of the three perception modules (verbatim excerpts) and, for each, the
agents of each reasoning module (9 per model, own-report condition) who answered correctly.
Data: results/cvpr_items.json (reports and per-item accuracy).
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
from cf_style import (MM, PAL, MODEL, BASE_PT, LEGEND_PT, ANNOT_PT, ms)    # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment       # noqa: E402

import json                                                                   # noqa: E402
import textwrap                                                               # noqa: E402
import numpy as np                                                            # noqa: E402
from PIL import Image                                                         # noqa: E402
from matplotlib.patches import FancyBboxPatch                                 # noqa: E402

REPO = ALT.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "alt"
QA = ALT.parent / "qa" / "alt"
BASE, QBASE = str(OUT / "A6_example"), str(QA / "A6_example")

width_mm = 174.6            # final printed width: CVPR \textwidth
height_mm = 66

Q = "vqarad/test-261"
v = json.loads((REPO / "results" / "cvpr_items.json").read_text())["reports"][Q]
REASON = ["gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4-5-20251001"]
PERC = ["gemini-3.7-flash", "gpt-5-mini", "claude-haiku-4-5-20251001"]
# verbatim excerpts of the slot-0 reports (ellipses mark omitted text)
EXC = {"gemini-3.7-flash": ("Axial contrast-enhanced computed tomography (CT) of the abdomen. Findings: Evaluation of the upper abdomen "
                            "reveals diffuse, multifocal, saccular and cystic dilatation of the intrahepatic bile ducts "
                            "throughout the hepatic parenchyma. …"),
       "gpt-5-mini": ("Contrast-enhanced CT abdomen, axial image. … Findings: Large gas- and fluid-filled gastric and "
                      "adjacent colonic loops in the left upper quadrant. Both kidneys demonstrate symmetric cortical "
                      "enhancement without hydronephrosis. …"),
       "claude-haiku-4-5-20251001": ("Contrast-enhanced CT abdomen, axial plane. … Large low-attenuation collection in the "
                                     "right upper quadrant, predominantly anterior to the right kidney … consistent with "
                                     "ascites or loculated fluid …")}
for p in PERC:                               # every excerpt must be a sequence of verbatim pieces of the report
    flat = " ".join(v["report"][p].replace("*", "").replace("#", "").split())
    for piece in EXC[p].replace("…", "|").split("|"):
        piece = " ".join(piece.split()).strip(" .")
        assert not piece or piece in flat, (p, piece)
W, H = width_mm, height_mm
fig = plt.figure(figsize=(W * MM, H * MM))
over = fig.add_axes([0, 0, 1, 1])
over.set_xlim(0, W); over.set_ylim(0, H); over.axis("off")


def marker(x, y, m, size=3.4):
    st = MODEL[m]
    over.plot([x], [y], ls="none", marker=st["marker"], ms=ms(st["marker"], size), mfc=st["color"], mec="white",
              mew=0.4, zorder=5)


def counts(x, y, key):
    for r in REASON:
        n = round(v["acc"][f"{r}|{key}"] * 9)
        marker(x, y, r)
        over.text(x + 1.8, y, f"{n}/9", ha="left", va="center", fontsize=ANNOT_PT,
                  color=PAL["ink"] if n >= 5 else PAL["neg_dark"])
        x += 11.5


SIDE, IX, IY = 40.0, 3.0, 13.0
ax_img = fig.add_axes([IX / W, IY / H, SIDE / W, SIDE / H])
with Image.open(REPO / v["image"]) as im:
    arr = np.asarray(im.convert("RGB"))
    ax_img.imshow(arr[: int(arr.shape[0] * 0.955)], interpolation="lanczos")   # crop the uploader caption strip
ax_img.set_xticks([]); ax_img.set_yticks([])
for sp in ax_img.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color(PAL["neutral_mid"])
over.text(IX, H - 1.3, "“Is there biliary duct dilation?”", ha="left", va="top", fontsize=BASE_PT, style="italic")
over.text(IX, H - 5.4, "Reference: Yes  (VQA-RAD, CT)", ha="left", va="top", fontsize=LEGEND_PT, color=PAL["neutral_dark"])
over.text(IX, IY - 3.2, "Agents reading the image, correct:", ha="left", va="center", fontsize=ANNOT_PT)
counts(IX + 1.2, IY - 7.4, "image")

BX, BW, BH, BG = IX + SIDE + 6.0, W - (IX + SIDE + 6.0) - 2.0, 17.0, 3.0
for k, p in enumerate(PERC):
    y1 = H - 2.0 - k * (BH + BG)
    y0 = y1 - BH
    over.add_patch(FancyBboxPatch((BX, y0), BW, BH, boxstyle="round,pad=0,rounding_size=1.2", fc="#FAFAFA",
                                  ec=MODEL[p]["color"], lw=0.8, zorder=1))
    marker(BX + 2.0, y1 - 2.6, p, size=3.8)
    over.text(BX + 4.0, y1 - 2.6, f"Report by {MODEL[p]['label']}", ha="left", va="center", fontsize=LEGEND_PT,
              fontweight="bold", color=PAL["ink"])
    over.text(BX + BW - 39.0, y1 - 2.6, "Reasoning modules correct:", ha="right", va="center", fontsize=ANNOT_PT,
              color=PAL["neutral_dark"])
    counts(BX + BW - 36.5, y1 - 2.6, p)
    over.text(BX + 2.0, y1 - 5.6, textwrap.fill(EXC[p], 96), ha="left", va="top", fontsize=6.8, color=PAL["ink"],
              linespacing=1.25)
over.text(BX, 1.5, "Markers: Gemini 3.7 Flash (square), GPT-5 mini (circle), Claude Haiku 4.5 (diamond); 9 agents each.",
          ha="left", va="bottom", fontsize=ANNOT_PT, color=PAL["neutral_dark"])

require_matplotlib_panel_alignment(fig, axes=[ax_img], json_out=QBASE + ".alignment.json",
                                   overlay_svg=QBASE + ".alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=False, strict=False)
fig.savefig(BASE + ".svg", dpi=600)
fig.savefig(BASE + ".pdf", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
fig.savefig(QBASE + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("written", BASE)
