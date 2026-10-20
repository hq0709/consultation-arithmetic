"""Shared vocabulary for the CVPR-format figure candidates (nature-figure skill, Python backend).

Every figure script still carries the skill's mandatory rcParams lines and its own explicit
SVG/PDF/TIFF exports, because validate_figure.py audits each source file on its own. Importing this
module then sets the face of the CVPR figures: Times New Roman, as the body text (author's choice,
2026-10-20); the two qualitative figures switch to Comic Sans MS in their own scripts. This
module holds only what the figures share: final widths, font sizes, palette, model encodings,
data loading, panel-letter placement and number formatting.

Widths: CVPR two-column layout, \\columnwidth = 237.14 pt (83.34 mm, measured), \\textwidth = 6.875 in
(174.6 mm). Every figure is drawn at its final printed width, so its text prints at the sizes
below (the author's scale: axis labels 9 pt, ticks 8.5, legends 8, in-plot values 7.5,
panel letters 10).
Model colours: the categorical set of the radiology figures, validated with the dataviz
palette validator (--pairs all, light surface): worst CVD dE 9.8, worst normal-vision dE 19.6.
Teal and amber sit below 3:1 contrast on white, so every model also has its own marker.
"""
import json
import pathlib

import numpy as np

MM = 1 / 25.4
COL_W_MM = 83.34                  # \columnwidth = 237.14 pt (measured with cvpr.sty)
TEXT_W_MM = 174.625                # \textwidth = 496.86 pt = 6.875 in

BASE_PT = 9.0
TICK_PT = 8.5
LEGEND_PT = 8.0
ANNOT_PT = 7.5
PANEL_LABEL_PT = 10.0

CF = pathlib.Path(__file__).resolve().parent
REPO = CF.parents[1]
RESULTS = REPO / "results"
OUT = REPO / "paper" / "cvpr" / "figure_candidates"
QA = CF / "qa"

PAL = {
    "ink": "#1F1F1F",
    "neutral_dark": "#606060",
    "neutral_mid": "#8C8C8C",
    "neutral_light": "#D8D8D8",
    "neutral_faint": "#EDEDED",
    "pos_dark": "#2F5D8C",       # correct answers (cross-tab), dark = changed by the image
    "pos_soft": "#A9C4DE",
    "neg_dark": "#B2453A",       # wrong answers
    "neg_soft": "#E6B3AC",
}

MODELS = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001", "qwen3.8-flash"]
API3 = MODELS[:3]
MODEL = {
    "gpt-5-mini":                {"label": "GPT-5 mini",       "color": "#454F91", "marker": "o"},
    "gemini-3.7-flash":          {"label": "Gemini 3.7 Flash", "color": "#3AAFA0", "marker": "s"},
    "claude-haiku-4-5-20251001": {"label": "Claude Haiku 4.5", "color": "#C98A1E", "marker": "D"},
    "qwen3.8-flash":             {"label": "Qwen 3.8 Flash",   "color": "#B83B5E", "marker": "^"},
}
OW = ["medgemma-4b", "qwen3-vl-8b", "qwen2-vl-7b", "huatuogpt-vision-7b", "llava-1.5-7b"]
OW_LABEL = {"medgemma-4b": "MedGemma 4B", "qwen3-vl-8b": "Qwen3-VL 8B", "qwen2-vl-7b": "Qwen2-VL 7B",
            "huatuogpt-vision-7b": "HuatuoGPT-Vision 7B", "llava-1.5-7b": "LLaVA-1.5 7B"}
OW_STYLE = {"color": PAL["neutral_dark"], "marker": "o"}
SETS = ["vqarad", "slake", "mxqa_mm_rad"]
SET_LABEL = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM"}
# markers marshal the matplotlib size so a diamond and a triangle read as large as a circle
MS_SCALE = {"o": 1.0, "s": 0.88, "D": 0.8, "^": 1.05, "v": 1.05}


FONT_DIR = CF / "fonts"


def use_times():
    """Times New Roman 2.82 (Microsoft core fonts; fonts/TimesNewRoman-*.ttf, fetched by fonts/fetch_fonts.sh),
    the face of the CVPR body text. Mathtext uses STIX, the matching Times-style math face."""
    from matplotlib import font_manager, rcParams
    files = sorted(FONT_DIR.glob("TimesNewRoman-*.ttf"))
    if not files:
        raise SystemExit("Times New Roman not found: run experiments/cvpr_fig/fonts/fetch_fonts.sh")
    for f in files:
        font_manager.fontManager.addfont(str(f))
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
    rcParams.update({"mathtext.fontset": "stix"})   # Times-style math; Greek letters italic as in the text


use_times()                        # applied on import, after each script's own rcParams block


def load(name):
    return json.loads((RESULTS / f"imaging_v2_{name}.json").read_text())


def ms(marker, size):
    return size * MS_SCALE.get(marker, 1.0)


def minus(s):
    """Typographic minus sign (U+2212) instead of a hyphen."""
    return str(s).replace("-", "−")


def signed(v, nd=0):
    """+3 / 0 / -3 tick labels with a typographic minus."""
    if abs(v) < 1e-9:
        return "0"
    return minus(f"{v:+.{nd}f}")


def add_panel_label(ax, label, x_offset_pt=-4, y_offset_pt=3, fontsize=PANEL_LABEL_PT):
    """Bold lowercase panel letter at a fixed physical offset from the axes' top-left corner."""
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + offset, fontsize=fontsize, fontweight="bold",
            color=PAL["ink"], ha="left", va="bottom")


def add_panel_title(ax, text, x_offset_pt=7, y_offset_pt=3, fontsize=BASE_PT):
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, text, transform=ax.transAxes + offset, fontsize=fontsize, color=PAL["ink"],
            ha="left", va="bottom")


def header(ax, letter, title=None, min_frac=-0.33, y_offset_pt=3):
    """Letter at the left edge of the axis decorations (inside the auditor's label window),
    title right after it."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = ax.get_window_extent(r)
    off = (ax.get_tightbbox(r).x0 - box.x0) * 72 / fig.dpi
    off = max(off, min_frac * box.width * 72 / fig.dpi)
    add_panel_label(ax, letter, x_offset_pt=off, y_offset_pt=y_offset_pt)
    if title:
        add_panel_title(ax, title, x_offset_pt=off + 10, y_offset_pt=y_offset_pt)


def at(ax, dx_pt=0.0, dy_pt=0.0):
    """Data transform shifted by a fixed physical offset."""
    from matplotlib.transforms import ScaledTranslation
    return ax.transData + ScaledTranslation(dx_pt / 72, dy_pt / 72, ax.figure.dpi_scale_trans)


def hci(ax, y, ci, color, marker, filled=True, size=3.6, lw=0.9, zorder=3, alpha=1.0):
    """Horizontal 95% CI with its point estimate."""
    est, lo, hi = ci
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder, alpha=alpha)
    ax.plot([est], [y], marker=marker, ms=ms(marker, size), color=color, mfc=color if filled else "white",
            mec=color, mew=0.9, lw=0, zorder=zorder + 1, alpha=alpha)


def beta_binomial_gain(pbar, phi, n=9):
    """Majority-of-n accuracy minus mean reader accuracy when the readers' correctness is
    beta-binomial with mean pbar and intraclass correlation phi (n odd)."""
    from scipy.stats import betabinom
    phi = np.asarray(phi, float)
    a, b = pbar * (1 - phi) / phi, (1 - pbar) * (1 - phi) / phi
    return 1 - betabinom.cdf((n - 1) // 2, n, a, b) - pbar


def neff(phi, n=9):
    return n / (1 + (n - 1) * np.asarray(phi, float))


def phi_from_neff(ne, n=9):
    return (n / np.asarray(ne, float) - 1) / (n - 1)
