"""Style board (round 4): the same two figures in three design directions, for the author to choose one
direction for every data figure of the CVPR paper. Point estimates only (the paper shows no intervals).

  python experiments/cvpr_fig/r4/style_board.py

Figures: (1) agent scaling, majority accuracy of the first k agents of the Independent MAS against k,
observed vs the prediction for k independent agents (binomial), VQA-RAD and SLAKE; (2) capability vs
multi-agent gain, one point per model x data set: SAS accuracy against the mean gain of the MAS
architectures available at N = 9.
Directions: A brand-minimal (family brand hues, model logos as direct labels and markers, light grid);
B pastel-soft (NMI pastel family, tinted panels, white grid, no spines); C editorial (individual systems in
grey, one accent for the key comparison, large in-plot statements).
Data: results/cvpr_nmi.json (experiments/cvpr_fig/extract_nmi.py). Outputs paper/cvpr/figure_candidates/
round4/board_<fig><dir>.{pdf,png,svg}; QA in experiments/cvpr_fig/qa/r4/. Nature-figure skill, Python.
"""
import sys
import json
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
mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
                     "legend.frameon": False})

R4 = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(R4.parent))
sys.path.insert(0, str(pathlib.Path.home() / ".claude" / "skills" / "nature-figure" / "scripts"))
from cf_style import MM, MODEL                                                  # noqa: E402  (Times New Roman)
from audit_panel_alignment import require_matplotlib_panel_alignment            # noqa: E402
import logos                                                                     # noqa: E402
from example_kit import tw                                                       # noqa: E402

import numpy as np                                                               # noqa: E402

REPO = R4.parents[2]
OUT = REPO / "paper" / "cvpr" / "figure_candidates" / "round4"
QA = R4.parent / "qa" / "r4"
OUT.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
D = json.loads((REPO / "results" / "cvpr_nmi.json").read_text())
CORE = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001", "qwen3.8-flash"]
SETS = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM"}
COL_W, TEXT_W = 83.34, 174.6

# design tokens per direction
BRAND = {"gpt-5-mini": "#10A37F", "gemini-3.7-flash": "#1A73E8", "claude-haiku-4-5-20251001": "#D97757",
         "qwen3.8-flash": "#7A5AF0"}
PASTEL = {"gpt-5-mini": "#5B6AAE", "gemini-3.7-flash": "#5FAE9F", "claude-haiku-4-5-20251001": "#DDA15E",
          "qwen3.8-flash": "#CF7790"}
INK, GREY, LIGHT = "#1F1F1F", "#6B6B6B", "#C9C9C9"
ACCENT = "#E4572E"
SHORT = {"gpt-5-mini": "GPT-5 mini", "gemini-3.7-flash": "Gemini 3.7 Flash", "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
         "qwen3.8-flash": "Qwen 3.8 Flash"}


def axes_style(ax, d):
    if d == "A":
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        for k in ("left", "bottom"):
            ax.spines[k].set_color("#9E9E9E"); ax.spines[k].set_linewidth(0.6)
        ax.tick_params(colors="#555555", width=0.6, length=2.5)
        ax.grid(axis="y", color="#ECECEC", lw=0.6); ax.set_axisbelow(True)
    elif d == "B":
        ax.set_facecolor("#F3F4F9")
        for k in ax.spines:
            ax.spines[k].set_visible(False)
        ax.tick_params(length=0, colors="#555555", pad=3)
        ax.grid(color="white", lw=0.9); ax.set_axisbelow(True)
    else:
        for k in ("top", "right", "left"):
            ax.spines[k].set_visible(False)
        ax.spines["bottom"].set_color("#444444"); ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(axis="y", length=0, colors="#555555"); ax.tick_params(axis="x", colors="#444444", width=0.7)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)


def title(ax, d, text):
    ax.set_title(text, loc="left", fontsize=9, fontweight="bold" if d in ("A", "C") else "normal",
                 color=INK, pad=6)


def overlay(fig, W, H):
    o = fig.add_axes([0, 0, 1, 1]); o.set_xlim(0, W); o.set_ylim(0, H); o.axis("off")
    return o


def to_mm(fig, ax, x, y, W, H):
    """Data point of ax -> millimetre coordinates of the figure overlay."""
    X, Y = ax.transData.transform((x, y))
    fx, fy = fig.transFigure.inverted().transform((X, Y))
    return fx * W, fy * H


def save(fig, name, axes=None, ids=None):
    base, qbase = str(OUT / name), str(QA / name)
    if axes and len(axes) > 1:
        require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=ids, row_groups=[{"id": "row", "panels": ids}],
                                           json_out=qbase + ".alignment.json",
                                           overlay_svg=qbase + ".alignment.svg", tolerance_pt=1.5,
                                           gutter_tolerance_pt=1.5, require_panel_labels=False, strict=False)
    fig.savefig(base + ".svg", dpi=600)
    fig.savefig(base + ".pdf", dpi=600)
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(qbase + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def repel(ys, gap):
    """Spread label heights so that neighbours are at least `gap` apart (order kept)."""
    order = np.argsort(ys); out = np.array(ys, float)
    for _ in range(50):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if out[b] - out[a] < gap:
                mid = (out[a] + out[b]) / 2; out[a], out[b] = mid - gap / 2, mid + gap / 2; moved = True
        if not moved:
            break
    return out


# ------------------------------------------------------------------------- figure 1: agent scaling
def scaling(d):
    W, H = TEXT_W, 64.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    # A: direct labels in the gutter; B: plain gap; C: a label zone right of every panel
    L, B, PH = 14.0, 11.0, 42.0
    PW, GAP = {"A": (60.0, 38.0), "B": (73.0, 12.0), "C": (54.0, 32.0)}[d]
    axes = [fig.add_axes([(L + i * (PW + GAP)) / W, B / H, PW / W, PH / H]) for i in range(2)]
    o = overlay(fig, W, H)
    pal = BRAND if d == "A" else PASTEL
    for ax, (s, lab) in zip(axes, list(SETS.items())[:2]):
        axes_style(ax, d)
        lo = 55 if s == "vqarad" else 65
        ends = []
        obs_all, ind_all = [], []
        for m in CORE:
            S = D["scaling"].get(f"{m}|{s}")
            if not S:
                continue
            k, obs, ind = np.array(S["k"]), np.array(S["observed"]), np.array(S["binomial"])
            obs_all.append(obs); ind_all.append(ind)
            if d == "A":
                ax.plot(k, ind, color=pal[m], lw=1.0, ls=(0, (3, 2)), alpha=0.7, zorder=2)
                ax.plot(k, obs, color=pal[m], lw=1.7, marker="o", ms=3.4, mfc="white", mew=1.0, zorder=4,
                        solid_capstyle="round")
                ends.append((m, obs[-1]))
            elif d == "B":
                ax.plot(k, ind, color=pal[m], lw=1.1, ls=(0, (1, 1.6)), zorder=2)
                ax.plot(k, obs, color=pal[m], lw=1.6, marker="o", ms=3.8, mec="white", mew=0.8, zorder=4)
            else:
                ax.plot(k, ind, color=LIGHT, lw=0.8, ls=(0, (3, 2)), zorder=2)
                ax.plot(k, obs, color=LIGHT, lw=0.9, zorder=2)
        if d == "C":
            mo, mi = np.mean(obs_all, 0), np.mean(ind_all, 0)
            ax.plot(k, mi, color=ACCENT, lw=2.0, ls=(0, (4, 2)), zorder=5)
            ax.plot(k, mo, color=INK, lw=2.2, zorder=6, solid_capstyle="round")
            gi, go = mi[-1] - mi[0], mo[-1] - mo[0]
            statements = ((mi[-1], f"{gi:+.0f} pp", "if independent", ACCENT),
                          (mo[-1], f"{go:+.1f} pp", "observed", INK))
        ax.set_xlim(0.6, 9.4); ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_ylim(lo, 101); ax.set_yticks(range(lo + 5, 101, 10) if lo % 10 == 5 else range(lo, 101, 10))
        ax.set_xlabel("Number of agents")
        title(ax, d, lab)
        if d == "C":          # statements at the line ends, in the label zone right of the panel (after limits)
            for yv, big, small, col in statements:
                X, Y = to_mm(fig, ax, 9.0, yv, W, H)
                o.text(X + 3.0, Y + 1.5, big, color=col, fontsize=8, fontweight="bold", va="center", ha="left")
                o.text(X + 3.0, Y - 1.6, small, color=col, fontsize=7.5, va="center", ha="left")
        if d == "A" and ax is axes[0]:                       # direct labels with logos, in the gutter
            ys = repel([v for _, v in ends], 3.6)
            for (m, v), yy in zip(ends, ys):
                X, Y = to_mm(fig, ax, 9.0, yy, W, H)
                logos.draw(o, m, X + 3.6, Y, 2.8)
                o.text(X + 5.6, Y, SHORT[m], fontsize=7.5, va="center", ha="left", color=INK)
    axes[0].set_ylabel("Majority accuracy (%)")
    # keys
    if d == "A":
        o.plot([L, L + 6], [H - 3.2, H - 3.2], color=GREY, lw=1.7); o.text(L + 7.5, H - 3.2, "observed", fontsize=8, va="center")
        o.plot([L + 26, L + 32], [H - 3.2, H - 3.2], color=GREY, lw=1.0, ls=(0, (3, 2)))
        o.text(L + 33.5, H - 3.2, "k independent agents of the same accuracy", fontsize=8, va="center")
    elif d == "B":
        x = L
        for m in CORE:
            o.plot([x, x + 5], [H - 3.2, H - 3.2], color=PASTEL[m], lw=1.6)
            o.text(x + 6.2, H - 3.2, SHORT[m], fontsize=8, va="center"); x += 8 + len(SHORT[m]) * 1.45
        o.plot([x + 2, x + 7], [H - 3.2, H - 3.2], color=GREY, lw=1.1, ls=(0, (1, 1.6)))
        o.text(x + 8.2, H - 3.2, "independent agents", fontsize=8, va="center")
    else:
        o.text(L, H - 3.2, "Grey: each model.  Black: mean observed.  Orange: mean expected for independent agents.",
               fontsize=8, va="center", color=GREY)
    save(fig, f"board_scaling{d}", axes, ["a", "b"])


# ------------------------------------------------------------------------- figure 2: capability vs gain
def capability(d):
    W, H = COL_W, {"A": 78.0, "B": 82.0, "C": 74.0}[d]
    fig = plt.figure(figsize=(W * MM, H * MM))
    L, B, PW, PH = 14.0, 11.0, 63.0, 52.0
    ax = fig.add_axes([L / W, B / H, PW / W, PH / H]); axes_style(ax, d)
    o = overlay(fig, W, H)
    pts = []
    for m in CORE:
        for s in SETS:
            A = D["arch"].get(f"{m}|{s}")
            if not A:
                continue
            g = [A[f"{a}|9|gain"] for a in ("Independent", "Centralized", "Decentralized", "Hybrid") if f"{a}|9|gain" in A]
            if g:
                pts.append((m, s, A["SAS"], float(np.mean(g))))
    x = np.array([p[2] for p in pts]); y = np.array([p[3] for p in pts])
    slope, icpt = np.polyfit(x, y, 1)
    xx = np.array([x.min() - 3, x.max() + 3])
    ax.axhline(0, color="#9A9A9A", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.plot(xx, icpt + slope * xx, color=GREY if d != "C" else INK, lw=1.0, zorder=2)
    marks = {"vqarad": "o", "slake": "s", "mxqa_mm_rad": "^"}
    for m, s, xv, yv in pts:
        if d == "A":
            continue                                                  # logo markers are placed after the limits
        elif d == "B":
            ax.plot([xv], [yv], ls="none", marker=marks[s], ms=6, mfc=PASTEL[m], mec="white", mew=0.8, zorder=4)
        else:
            hl = m == "qwen3.8-flash"
            ax.plot([xv], [yv], ls="none", marker="o", ms=5.6 if hl else 4.6, mfc=ACCENT if hl else LIGHT,
                    mec="white", mew=0.7, zorder=5 if hl else 3)
    ax.set_xlim(25, 100); ax.set_ylim(-6, 16)
    ax.set_xlabel("SAS accuracy (%)"); ax.set_ylabel("Mean multi-agent gain (pp)")
    if d == "A":
        from example_kit import avatar                                # noqa: E402
        for m, s, xv, yv in pts:
            X, Y = to_mm(fig, ax, xv, yv, W, H)
            avatar(o, m, X, Y, d=3.8)
    if d == "A":
        ax.text(97, 14.6, f"slope {slope:+.2f} pp per pp", ha="right", va="top", fontsize=7.5, color=GREY)
        for k, m in enumerate(CORE):
            X, Y = L + (k % 2) * 34.0, H - 3.0 - (k // 2) * 4.2
            logos.draw(o, m, X + 1.4, Y, 2.6)
            o.text(X + 3.2, Y, SHORT[m], fontsize=7.5, va="center")
    elif d == "B":
        o.text(L + PW, B + PH + 2.4, f"slope {slope:+.2f} pp per pp", ha="right", va="center", fontsize=7.5,
               color=GREY)
        for k, m in enumerate(CORE):
            X, Y = L + (k % 2) * 34.0, H - 3.0 - (k // 2) * 4.0
            o.plot([X + 1.2], [Y], ls="none", marker="o", ms=5, mfc=PASTEL[m], mec="white")
            o.text(X + 2.8, Y, SHORT[m], fontsize=7.5, va="center")
        x0 = L
        for s, lab in SETS.items():
            o.plot([x0 + 1.2], [H - 11.0], ls="none", marker=marks[s], ms=4.6, mfc="#9A9A9A", mec="white")
            o.text(x0 + 2.8, H - 11.0, lab, fontsize=7.5, va="center"); x0 += 6.0 + tw(lab, 7.5)
    else:
        q = [p for p in pts if p[0] == "qwen3.8-flash"]; r = [p for p in pts if p[0] != "qwen3.8-flash"]
        o.text(L, H - 2.8, f"Qwen 3.8 Flash: {min(p[3] for p in q):+.0f} to {max(p[3] for p in q):+.0f} pp",
               color=ACCENT, fontsize=8, fontweight="bold", va="center")
        o.text(L, H - 6.6, f"other models: {min(p[3] for p in r):+.0f} to {max(p[3] for p in r):+.0f} pp",
               color=GREY, fontsize=8, va="center")
    save(fig, f"board_capability{d}")


for direction in "ABC":
    scaling(direction)
    capability(direction)
print("written", OUT)
