"""fig0_architectures — candidate A: the five coordination structures, and the visual key.

Nature-figure skill, Python backend. Redraw of experiments/fig_arch.py on one canvas in
millimetre coordinates. Every architecture is drawn in its ARCH colour and marker, the
encoding every later figure reuses. Hybrid follows panels/architectures.py::arch_tiered:
a generalist answers first; below the confidence threshold θ the case is referred, one way,
to the specialist panel, which exchanges messages only when its vote has no majority.
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
})

NF = pathlib.Path(__file__).resolve().parent                     # experiments/nature_fig
REPO = NF.parents[1]
for p in (NF, REPO):
    sys.path.insert(0, str(p))

from nf_style import MM, PAL, ARCH, BASE_PT, ANNOT_PT                              # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment               # noqa: E402

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch                   # noqa: E402

OUT = REPO / "paper" / "figures"
QA = NF / "qa"
QA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
BASE = str(OUT / "fig0_architectures")
QBASE = str(QA / "fig0_architectures")

width_mm = 160          # final printed width: \textwidth
height_mm = 40

# Hybrid's structure label. "star + peer" (the current figure) names the NMI Hybrid, which
# the paper disclaims (systems.tex l.113-118) and the code does not implement; the paper's
# own term is used instead. Set to "star + peer" to restore the old text.
HYBRID_C = "gated escalation"

# ── geometry (mm) ──
COL = [16.0, 48.0, 80.0, 112.0, 144.0]     # column centres
Y_TITLE, Y_ANN, Y_KEY = 36.2, 8.7, 3.0
Y_TOP, Y_BOT = 28.0, 17.6
Y_MID = (Y_TOP + Y_BOT) / 2
LEAD_MS, SPEC_MS = 8.8, 8.2                # marker sizes (pt): lead role, specialist (circle)
MS_SCALE = {"o": 1.0, "s": 0.88, "^": 1.15, "D": 0.86}
EDGE_C, EDGE_LW = PAL["neutral_dark"], 0.8
COND_C, COND_LS = PAL["neutral_mid"], (0, (2.2, 1.6))
LEAD_C = ARCH["cot"]["color"]
INK, SOFT = PAL["ink"], PAL["neutral_dark"]

fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, width_mm)
ax.set_ylim(0, height_mm)
ax.set_axis_off()


def r_mm(ms_pt):
    return ms_pt / 2 / 72 * 25.4


def node(x, y, arch=None, lead=False, size=None):
    if lead:
        ax.plot([x], [y], ls="none", marker="o", ms=size or LEAD_MS, mfc=LEAD_C, mec="white",
                mew=0.7, zorder=5)
        return
    mk = ARCH[arch]["marker"]
    ax.plot([x], [y], ls="none", marker=mk, ms=(size or SPEC_MS) * MS_SCALE[mk],
            mfc=ARCH[arch]["color"], mec="white", mew=0.7, zorder=5)


def edge(p, q, conditional=False):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=COND_C if conditional else EDGE_C,
            lw=EDGE_LW, ls=COND_LS if conditional else "-", solid_capstyle="butt",
            dash_capstyle="butt", zorder=3)


def role(x, y, text):
    ax.text(x, y + r_mm(LEAD_MS) + 0.9, text, ha="center", va="bottom", fontsize=ANNOT_PT,
            color=INK)


# ── titles: the ARCH glyph, then the condition name, centred as one group ──
order = ["cot", "independent", "centralized", "discussion", "tiered"]
renderer = fig.canvas.get_renderer()
for cx, a in zip(COL, order):
    t = ax.text(0, Y_TITLE, ARCH[a]["label"], ha="left", va="center", fontsize=BASE_PT,
                color=INK)
    w = t.get_window_extent(renderer).width / fig.dpi * 25.4
    glyph_w, gap = 2.6, 1.5
    x0 = cx - (glyph_w + gap + w) / 2
    t.set_x(x0 + glyph_w + gap)
    mk = ARCH[a]["marker"]
    ax.plot([x0 + glyph_w / 2], [Y_TITLE], ls="none", marker=mk, ms=7.0 * MS_SCALE[mk],
            mfc=ARCH[a]["color"], mec="white", mew=0.4, zorder=5)

# ── a: single doctor ──
node(COL[0], Y_MID, lead=True)

# ── b: Independent, no channel ──
for dx in (-7.0, 0.0, 7.0):
    node(COL[1] + dx, Y_MID, "independent")

# ── c: Centralized, star through the attending ──
cx = COL[2]
for dx in (-8.0, 0.0, 8.0):
    edge((cx, Y_TOP), (cx + dx, Y_BOT))
    node(cx + dx, Y_BOT, "centralized")
node(cx, Y_TOP, lead=True)
role(cx, Y_TOP, "Attending")

# ── d: Decentralized, complete graph ──
cx = COL[3]
tri = [(cx, Y_TOP), (cx - 8.0, Y_BOT), (cx + 8.0, Y_BOT)]
for i in range(3):
    for j in range(i + 1, 3):
        edge(tri[i], tri[j])
for x, y in tri:
    node(x, y, "discussion")

# ── e: Hybrid, gated one-way referral to a panel with conditional peer channels ──
cx = COL[4]
y_gen = Y_TOP                                     # same height as the attending
box = (cx - 10.5, 12.6, 21.0, 9.4)                 # x, y, w, h of the specialist panel
ax.add_patch(FancyBboxPatch(box[:2], box[2], box[3], boxstyle="round,pad=0,rounding_size=1.6",
                            fc=PAL["neutral_light"], ec="none", alpha=0.45, zorder=1))
inv = [(cx - 6.0, 19.3), (cx + 6.0, 19.3), (cx, 15.0)]
for i in range(3):
    for j in range(i + 1, 3):
        edge(inv[i], inv[j], conditional=True)
for x, y in inv:
    node(x, y, "tiered")
node(cx, y_gen, lead=True)
role(cx, y_gen, "Generalist")
ax.add_patch(FancyArrowPatch((cx, y_gen - r_mm(LEAD_MS) - 0.5), (cx, box[1] + box[3] + 0.15),
                             arrowstyle="-|>,head_length=1.9,head_width=1.15", color=EDGE_C,
                             lw=EDGE_LW, shrinkA=0, shrinkB=0, zorder=3,
                             mutation_scale=1.0))
ax.text(cx + 1.4, (y_gen - r_mm(LEAD_MS) + box[1] + box[3]) / 2, r"conf. < $\theta$",
        ha="left", va="center", fontsize=ANNOT_PT, color=SOFT)

# ── structure annotations ──
for cx, s in zip(COL, [r"$\{a\}$", r"$C=\varnothing$", "star", "complete", HYBRID_C]):
    ax.text(cx, Y_ANN, s, ha="center", va="center", fontsize=ANNOT_PT, color=SOFT)

# ── key strip: roles and edge types, frameless and quiet ──
items = []
x = 0.0


def key_text(x, s):
    t = ax.text(x, Y_KEY, s, ha="left", va="center", fontsize=ANNOT_PT, color=SOFT)
    return t, t.get_window_extent(renderer).width / fig.dpi * 25.4


entries = []
# lead role
entries.append(("lead", "Attending / generalist"))
entries.append(("spec", "Specialists"))
entries.append(("line", "Message channel"))
entries.append(("dash", "Only if the panel has no majority"))
entries.append(("arrow", "Referral"))
widths = {"lead": 2.6, "spec": 10.4, "line": 4.6, "dash": 4.6, "arrow": 4.6}
gap_glyph, gap_item = 1.2, 3.0
texts = []
for kind, s in entries:
    t, w = key_text(0, s)
    texts.append((kind, t, w))
total = sum(widths[k] + gap_glyph + w for k, _, w in texts) + gap_item * (len(texts) - 1)
x = (width_mm - total) / 2
for kind, t, w in texts:
    g = widths[kind]
    if kind == "lead":
        ax.plot([x + g / 2], [Y_KEY], ls="none", marker="o", ms=6.4, mfc=LEAD_C, mec="white",
                mew=0.4)
    elif kind == "spec":
        for k, a in enumerate(["independent", "centralized", "discussion", "tiered"]):
            mk = ARCH[a]["marker"]
            ax.plot([x + 1.2 + k * 2.65], [Y_KEY], ls="none", marker=mk, ms=5.6 * MS_SCALE[mk],
                    mfc=ARCH[a]["color"], mec="white", mew=0.4)
    elif kind == "line":
        ax.plot([x, x + g], [Y_KEY, Y_KEY], color=EDGE_C, lw=EDGE_LW, solid_capstyle="butt")
    elif kind == "dash":
        ax.plot([x, x + g], [Y_KEY, Y_KEY], color=COND_C, lw=EDGE_LW, ls=COND_LS,
                dash_capstyle="butt")
    else:
        ax.add_patch(FancyArrowPatch((x, Y_KEY), (x + g, Y_KEY),
                                     arrowstyle="-|>,head_length=1.6,head_width=1.0",
                                     color=EDGE_C, lw=EDGE_LW, shrinkA=0, shrinkB=0,
                                     mutation_scale=1.0))
    t.set_x(x + g + gap_glyph)
    x += g + gap_glyph + w + gap_item

# ── render-time alignment gate (single canvas: NOT APPLICABLE), then exports ──
require_matplotlib_panel_alignment(
    fig,
    json_out=QBASE + ".alignment.json",
    overlay_svg=QBASE + ".alignment.svg",
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    require_panel_labels=True,
    strict=True,
)
fig.savefig(BASE + ".svg")
fig.savefig(BASE + ".pdf")
fig.savefig(QBASE + ".tiff", dpi=600)
fig.savefig(BASE + ".png", dpi=300)
plt.close(fig)
print("key strip width (mm):", round(total, 1))
