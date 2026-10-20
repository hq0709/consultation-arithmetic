"""Drawing helpers shared by the two qualitative figures (A5_example, A6_example).

All helpers draw on a millimetre overlay axes (x and y in mm, equal units) that covers the figure.
Text widths are measured with matplotlib's TextToPath at the final point size, so runs of differently
styled text can be set side by side and paragraphs can be wrapped to a width in mm.
"""
import pathlib

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import text_to_path

PT_MM = 25.4 / 72
FONT_DIR = pathlib.Path(__file__).resolve().parent / "fonts"

# design tokens of the two qualitative figures
STYLE = {
    "ink": "#111827",          # primary text
    "text2": "#4B5563",        # secondary text
    "muted": "#6B7280",        # tertiary text, keys
    "rule": "#E5E7EB",         # hairlines between rows
    "rule_strong": "#D1D5DB",  # table top and bottom rules, avatar rings
    "card": "#F5F6F8",         # report cards
    "hl": "#FCE59A",           # highlighter
    "right": "#2F5D8C",        # correct (as in the answer cross-tabulation of Fig. 4)
    "wrong": "#B2453A",        # wrong
    "target": "#F25F4C",       # occlusion boxes, bright enough on a dark scan
    "control": "#4DA3FF",
}


def use_comic():
    """Register Comic Sans MS 2.10 (Microsoft core fonts, fonts/ComicSansMS-*.ttf; regular and bold, embeddable)
    and make it the sans-serif face. Requested weights resolve to the nearest face: medium to regular,
    semibold to bold."""
    from matplotlib import font_manager, rcParams
    for f in sorted(FONT_DIR.glob("ComicSansMS-*.ttf")):
        font_manager.fontManager.addfont(str(f))
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Comic Sans MS", "Arial", "Liberation Sans", "DejaVu Sans"]
    for k in ("rm", "sf", "it"):
        rcParams[f"mathtext.{k}"] = "Comic Sans MS"
    rcParams["mathtext.bf"] = "Comic Sans MS:bold"


def use_inter():
    """Register Inter 4.0 (SIL OFL 1.1, fonts/Inter-*.ttf) and make it the sans-serif face."""
    from matplotlib import font_manager, rcParams
    for f in sorted(FONT_DIR.glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(f))
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Inter", "Arial", "Liberation Sans", "DejaVu Sans"]
    for k in ("rm", "sf"):
        rcParams[f"mathtext.{k}"] = "Inter"
    rcParams["mathtext.it"] = "Inter:italic"
    rcParams["mathtext.bf"] = "Inter:bold"


def font(size, weight="normal", style="normal"):
    return FontProperties(family="sans-serif", size=size, weight=weight, style=style)


def tw(s, size, weight="normal", style="normal"):
    """Advance width of s in mm."""
    if not s:
        return 0.0
    w = text_to_path.get_text_width_height_descent(s.replace(" ", " "), font(size, weight, style), ismath=False)[0]
    return w * PT_MM


def runs(ax, x, y, parts, va="center", zorder=7):
    """Set (text, kwargs) parts one after another from x; kwargs may hold size, weight, style, color.
    Returns the x where the last part ends."""
    for s, kw in parts:
        size, weight, style = kw.get("size", 8), kw.get("weight", "normal"), kw.get("style", "normal")
        ax.text(x, y, s, ha="left", va=va, fontsize=size, fontweight=weight, fontstyle=style,
                color=kw.get("color", "#1F1F1F"), zorder=zorder)
        x += tw(s, size, weight, style)
    return x


def pill(ax, x, y, w, h, fc="#F1F3F5", ec="none", lw=0.0, r=None, zorder=2):
    """Rounded box with lower-left corner (x, y)."""
    r = h / 2 if r is None else r
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=zorder))


def agents(ax, cx, y, filled, color, pitch=2.65, d=2.0, zorder=6):
    """One dot per agent, centred on cx: filled dots for True, open dots for False."""
    n = len(filled)
    ms = d / PT_MM
    for i, f in enumerate(filled):
        x = cx + (i - (n - 1) / 2) * pitch
        ax.plot([x], [y], ls="none", marker="o", ms=ms, mfc=color if f else "white", mec=color,
                mew=0.4 if f else 0.6, zorder=zorder)


def check(ax, x, y, s, color, lw=1.1, zorder=7):
    ax.plot([x - s, x - 0.3 * s, x + s], [y, y - 0.7 * s, y + 0.9 * s], color=color, lw=lw,
            solid_capstyle="round", solid_joinstyle="round", zorder=zorder)


def cross(ax, x, y, s, color, lw=1.1, zorder=7):
    for sgn in (1, -1):
        ax.plot([x - s, x + s], [y - sgn * s, y + sgn * s], color=color, lw=lw, solid_capstyle="round", zorder=zorder)


def picture(fig, W, H, x, y, side, arr, edge="#8C8C8C", radius=0.05):
    """Image axes with rounded corners; (x, y) is the lower-left corner in mm. Returns the axes."""
    ax = fig.add_axes([x / W, y / H, side / W, side / H])
    im = ax.imshow(arr, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    clip = FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={radius}", transform=ax.transAxes,
                          fc="none", ec="none")
    im.set_clip_path(clip)
    if edge != "none":
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={radius}", transform=ax.transAxes,
                                    fill=False, ec=edge, lw=0.5, zorder=5, clip_on=False))
    return ax


def paragraph(ax, x, y_top, width, segments, size=7.5, leading=1.3, hl_color="#FCE59A", color="#1F2937", hl_weight="semibold", zorder=7):
    """Wrap (text, highlighted) segments to `width` mm from the top edge y_top; highlighted words are set
    bold on a highlighter band. Returns the y of the last baseline."""
    words = [(w, h) for s, h in segments for w in s.split()]
    sp = tw(" ", size)
    lines, cur, cw = [], [], 0.0
    for w, h in words:
        ww = tw(w, size, hl_weight if h else "normal")
        if cur and cw + sp + ww > width:
            lines.append(cur); cur, cw = [], 0.0
        cw += (sp if cur else 0) + ww
        cur.append((w, h, ww))
    if cur:
        lines.append(cur)
    lh = size * leading * PT_MM
    asc = 0.74 * size * PT_MM
    base = y_top - asc
    for line in lines:
        xx = x
        k = 0
        while k < len(line):                      # group consecutive words of the same style into one run
            j = k
            while j + 1 < len(line) and line[j + 1][1] == line[k][1]:
                j += 1
            text = " ".join(w for w, _, _ in line[k:j + 1])
            h = line[k][1]
            wrun = tw(text, size, hl_weight if h else "normal")
            if h:
                ax.add_patch(Rectangle((xx - 0.2, base - 0.3 * size * PT_MM), wrun + 0.4, 1.12 * size * PT_MM,
                                       fc=hl_color, ec="none", zorder=zorder - 1))
            ax.text(xx, base, text, ha="left", va="baseline", fontsize=size, fontweight=hl_weight if h else "normal",
                    color=color, zorder=zorder)
            xx += wrun + sp
            k = j + 1
        base -= lh
    return base + lh


def square(arr):
    """Pad an image array to a square with its corner colour (scans have black backgrounds)."""
    h, w = arr.shape[:2]
    if h == w:
        return arr, (0, 0)
    s = max(h, w)
    out = np.empty((s, s) + arr.shape[2:], arr.dtype)
    out[...] = arr[0, 0]
    oy, ox = (s - h) // 2, (s - w) // 2
    out[oy:oy + h, ox:ox + w] = arr
    return out, (ox, oy)


def avatar(ax, model, cx, cy, d=4.2, ring="#D1D5DB", zorder=6):
    """Model logo inside a white circle of diameter d (mm), so marks of different visual weight read alike."""
    import logos
    from matplotlib.patches import Circle
    ax.add_patch(Circle((cx, cy), d / 2, fc="white", ec=ring, lw=0.45, zorder=zorder))
    logos.draw(ax, model, cx, cy, d * 0.64, zorder=zorder + 1)


def qtag(ax, x, y, letter="Q", s=3.4, fc="#111827", zorder=6):
    """Small rounded square with a white letter; (x, y) is its centre."""
    ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle=f"round,pad=0,rounding_size={s * 0.22}",
                                fc=fc, ec="none", zorder=zorder))
    ax.text(x, y - 0.02, letter, ha="center", va="center", fontsize=7.0, fontweight="semibold", color="white",
            zorder=zorder + 1)


def hline(ax, x0, x1, y, color="#E5E7EB", lw=0.45, zorder=1):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
