"""Shared figure system for the candidate redesign (nature-figure skill, Python backend).

Every candidate script still carries the skill's mandatory rcParams lines and its own
explicit SVG/PDF/TIFF exports, because validate_figure.py audits each source file on
its own. This module holds only the shared vocabulary: final widths, palette,
architecture encoding, panel-label placement and number formatting.
"""
import numpy as np

MM = 1 / 25.4                    # inches per millimetre
TEXT_W_MM = 160.0                # ACL \textwidth  (6.30 in)
COL_W_MM = 80.0                  # ACL \columnwidth (3.15 in)

# Font sizes at final physical size (pt). Mathtext scripts shrink to ~0.7x, so any
# label that contains a sub/superscript uses LABEL_MATH_PT (0.7 x 7.5 = 5.25 pt).
BASE_PT = 7.0
TICK_PT = 6.0
LEGEND_PT = 6.0
LABEL_MATH_PT = 7.5
PANEL_LABEL_PT = 8.0
ANNOT_PT = 6.0

# NMI-pastel-derived unified family (nature-figure api.md), with two reserved
# directional colours. Lightness steps are separated for grey-scale print, and
# every architecture also has its own marker shape, so colour is never the only cue.
PAL = {
    "ink": "#1F1F1F",            # text / single-doctor reference
    "neutral_dark": "#606060",
    "neutral_mid": "#8C8C8C",
    "neutral_light": "#D8D8D8",
    "cool_dark": "#454F91",
    "cool_mid": "#7A86B8",
    "cool_soft": "#98AEE0",
    "rose": "#C0587A",
    "rose_soft": "#E4B4C4",
    "teal": "#42949E",           # accent for a highlighted contrast (e.g. cross-vendor)
    "delta_up": "#2E9E44",       # gains only
    "delta_down": "#E53935",     # drops only
}

# One encoding per condition, reused in every figure.
ARCH = {
    "cot":         {"label": "Single doctor",    "color": PAL["ink"],         "marker": "o"},
    "sc":          {"label": "Self-consistency", "color": PAL["neutral_mid"], "marker": "X"},
    "independent": {"label": "Independent",      "color": PAL["cool_mid"],    "marker": "o"},
    "centralized": {"label": "Centralized",      "color": PAL["cool_dark"],   "marker": "s"},
    "discussion":  {"label": "Decentralized",    "color": PAL["rose"],        "marker": "^"},
    "tiered":      {"label": "Hybrid",           "color": PAL["cool_soft"],   "marker": "D"},
}
MAS_ORDER = ["independent", "centralized", "discussion", "tiered"]

BENCH_ORDER = ["medxpertqa", "medagentsbench", "medqa"]
BENCH_LABEL = {"medxpertqa": "MedXpertQA", "medagentsbench": "MedAgentsBench-hard",
               "medqa": "MedQA (USMLE)"}

# Canonical capitalisation for display labels (never .title()).
MODEL_LABEL = {
    "gpt-4.1-nano": "GPT-4.1 nano", "gpt-5-nano": "GPT-5 nano", "gpt-5-mini": "GPT-5 mini",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gemini-3.7-flash": "Gemini 3.7 Flash",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5", "claude-haiku-4.5": "Claude Haiku 4.5",
    "claude-sonnet-5": "Claude Sonnet 5", "deepseek-v4-flash": "DeepSeek-V4 Flash",
}
VENDOR_LABEL = {"openai": "OpenAI", "google": "Google", "anthropic": "Anthropic",
                "deepseek": "DeepSeek"}


def add_panel_label(ax, label, x=0, y=1, x_offset_pt=-4, y_offset_pt=3,
                    fontsize=PANEL_LABEL_PT, color=PAL["ink"]):
    """Bold lowercase panel letter at a fixed physical offset (nature-figure api.md)."""
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + offset, fontsize=fontsize,
            fontweight="bold", color=color, ha="left", va="bottom")


def fmt_signed(v, nd=1):
    return f"{v:+.{nd}f}"


def oklab_delta_e(hex_a, hex_b):
    """OKLab ΔE×100, for the colour-separation note in the QA record."""
    def ok(h):
        c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]])
        m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                       [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]])
        return m2 @ np.cbrt(m1 @ c)
    return float(np.linalg.norm(ok(hex_a) - ok(hex_b)) * 100)


def text_height_data(ax, fontsize_pt=ANNOT_PT, factor=1.25):
    """Height of one text line in y-data units at final size (call after limits are set)."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    y0, y1 = ax.get_ylim()
    h_in = bbox.height / fig.dpi
    return (fontsize_pt * factor / 72) / h_in * (y1 - y0)


def clear_of_lines(y, height, lines, pad):
    """Lift a label whose band [y, y+height] would be crossed by a horizontal line."""
    for ly in sorted(lines):
        if y - pad <= ly <= y + height + pad:
            y = ly + pad
    return y


def add_panel_title(ax, text, x_offset_pt=7, y_offset_pt=3, fontsize=BASE_PT, color="#1F1F1F"):
    """Panel title that starts to the right of the panel letter, at fixed physical offsets."""
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, text, transform=ax.transAxes + offset, fontsize=fontsize, color=color,
            ha="left", va="bottom")


def minus(s):
    """Typographic minus sign (U+2212) instead of a hyphen in signed numbers."""
    return str(s).replace("-", "−")
