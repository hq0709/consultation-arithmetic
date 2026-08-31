"""统一绘图样式 —— 复刻用户指定的 CVPR/ICLR 视觉语言（token-pruning 那张三联图）。

逐像素对照目标图后确定的规则：
  1. **衬线字体**（STIXGeneral = Times 克隆），与正文 Times 一致。matplotlib 默认无衬线
     是"一眼看出是随手画的"的头号来源。
  2. **四边细边框**（不是去掉上右 spine）+ **极淡灰网格**，网格永远在数据下方。
  3. **颜色 = 序列身份（架构）**，**面板 = benchmark**。目标图里 VQA/MME 是面板标题，
     FastV/DivPrune/Ours 是颜色 —— 我们照搬：benchmark 作面板，架构作颜色。
  4. **基线全部细虚线 (lw 1.3, ms 5.2)；被强调的那条更粗更亮 (lw 2.3, ms 6.8)**。
  5. **参考线 = 灰虚线 + 行内灰标签**（目标图的 "LLaVA-1.5-7B"）。我们的对应物是
     单医生基线。
  6. **图例在面板内，带浅灰细框**，不是浮在图外。
  7. 图按**最终排版宽度**出图（单栏 3.15in / 双栏 6.3in），绝不出大图再让 LaTeX 缩，
     否则等效字号会掉到 6pt 以下。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- 排版实宽（acl.sty，letter 双栏）
COL_W, TEXT_W = 3.15, 6.30

# ---- 调色板：直接取自目标图（橙 / 玫瑰 / 紫 / 青），全部低饱和、明度接近、可区分
C_ORANGE = "#e8974a"
C_ROSE   = "#cf6a86"
C_PURPLE = "#7d6fbe"
C_CYAN   = "#4ec3de"
C_GRAY   = "#8c8c8c"

# 颜色 = 架构（与目标图 FastV/DivPrune/PruMerge/Ours 的角色一一对应）
ARCH_SOLID = {
    "cot":          C_GRAY,
    "independent":  C_ORANGE,
    "centralized":  C_ROSE,
    "discussion":   C_PURPLE,
    "tiered":       C_CYAN,
}

# 形状 = 架构（目标图：o / s / ^ / D）
ARCH_MARKER = {
    "cot":          dict(label="Single doctor",     marker="o", ms=5.2, ls="--"),
    "independent":  dict(label="MAS-Independent",   marker="o", ms=5.2, ls="--"),
    "centralized":  dict(label="MAS-Centralized",   marker="s", ms=5.0, ls="--"),
    "discussion":   dict(label="MAS-Decentralized", marker="^", ms=5.6, ls="--"),
    "tiered":       dict(label="MAS-Hybrid",        marker="D", ms=4.8, ls="--"),
}
ARCH_ORDER = ["cot", "centralized", "discussion", "independent", "tiered"]
MAS_ORDER = ["independent", "centralized", "discussion", "tiered"]

INK = "#1a1a1a"; MUTED = "#7a7a7a"; FAINT = "#e5e5e5"
LINE = "#b8b8b8"; LINE_SAS = "#9a9a9a"
GRID = "#e3e3e3"; FRAME = "#3a3a3a"
GAIN_POS = "#2e8b57"; GAIN_NEG = "#c0392b"

# 兼容旧接口：颜色不再由 benchmark 决定，但保留符号避免上游报错。
BENCH_RAMP = {b: [ARCH_SOLID[a] for a in ARCH_ORDER]
              for b in ("medxpertqa", "medagentsbench", "medqa")}
BENCH_COLOR = {"medxpertqa": C_ROSE, "medagentsbench": C_PURPLE, "medqa": C_ORANGE}


def arch_color(bench, arch):
    """颜色编码架构（目标图的做法）；benchmark 由面板标题承担。"""
    return ARCH_SOLID.get(arch, C_GRAY)


TIER_LABEL = {"gpt-4.1-nano": "GPT-4.1\nnano", "gpt-5-nano": "GPT-5\nnano",
              "gpt-5-mini": "GPT-5\nmini",
              "gemini-3.5-flash-lite": "Gemini 3.5\nFlash Lite",
              "gemini-3.7-flash": "Gemini 3.7\nFlash"}
TIER_ORDER = ["gpt-4.1-nano", "gpt-5-nano", "gpt-5-mini"]
BENCH_LABEL = {"medxpertqa": "MedXpertQA", "medqa": "MedQA (USMLE)",
               "medagentsbench": "MedAgentsBench-hard"}
BENCH_ORDER = ["medxpertqa", "medagentsbench", "medqa"]
CAPABILITY = {"gpt-4.1-nano": 34.0, "gpt-5-nano": 50.8, "gpt-5-mini": 59.2,
              "gemini-3.5-flash-lite": 54.2, "gemini-3.7-flash": 81.7}


def rcparams():
    plt.rcParams.update({
        # 衬线，与正文 Times 一致
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "normal",
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "legend.fontsize": 8.4,
        # 四边细边框
        "axes.edgecolor": FRAME, "axes.linewidth": 0.8,
        "xtick.color": FRAME, "ytick.color": FRAME,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.direction": "out", "ytick.direction": "out",
        # 极淡网格，永远在数据下方
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "grid.linestyle": "-", "axes.axisbelow": True,
        # 面板内浅框图例
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "#c4c4c4", "legend.fancybox": False,
        "legend.borderpad": 0.42, "legend.labelspacing": 0.34,
        "legend.handlelength": 1.9, "legend.handletextpad": 0.55,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def clean(ax, grid_axis="both"):
    """目标图保留四边框；grid_axis 控制网格方向（默认双向）。"""
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8); s.set_color(FRAME)
    ax.tick_params(direction="out", length=3.0, width=0.8, top=False, right=False)
    ax.grid(True, axis=grid_axis if grid_axis in ("x", "y", "both") else "both",
            color=GRID, lw=0.7, ls="-")
    ax.set_axisbelow(True)


def series(ax, x, y, arch, highlight=False, label=None, **kw):
    """目标图的线型分工：基线细虚线 + 小实心点；被强调的一条更粗更亮更大点。"""
    st = ARCH_MARKER[arch]; c = ARCH_SOLID[arch]
    kw.setdefault("color", c)
    return ax.plot(x, y, ls="--", marker=st["marker"],
                   lw=2.3 if highlight else 1.3,
                   ms=(st["ms"] + 1.6) if highlight else st["ms"],
                   mfc=c, mec=c, mew=0.0,
                   zorder=6 if highlight else 4,
                   label=label if label is not None else st["label"], **kw)


def baseline_line(ax, y, label, x=0.985, va="bottom", color="#7a7a7a", fontsize=8.0):
    """目标图的 'LLaVA-1.5-7B' 灰虚参考线 + 行内灰标签。"""
    ax.axhline(y, color=color, ls="--", lw=1.15, zorder=2)
    ax.text(x, y, label + " ", transform=ax.get_yaxis_transform(),
            ha="right", va=va, fontsize=fontsize, color=color)


def panel_legend(ax, loc="lower right", ncol=1, **kw):
    """面板内浅框图例（目标图的做法）。"""
    kw.setdefault("fontsize", 8.4)
    return ax.legend(loc=loc, ncol=ncol, frameon=True, framealpha=0.92,
                     edgecolor="#c4c4c4", fancybox=False, borderpad=0.42,
                     labelspacing=0.34, handlelength=1.9, handletextpad=0.55, **kw)


def _handles(keys, colored=True):
    return [Line2D([], [], color=ARCH_SOLID[k] if colored else "#8f8f8f",
                   marker=ARCH_MARKER[k]["marker"], ms=ARCH_MARKER[k]["ms"],
                   ls="--", lw=1.3, mew=0.0, label=ARCH_MARKER[k]["label"])
            for k in keys]


def shape_legend(fig, ncol=5, y=0.005, include=None):
    """图级图例（多面板共享时用）。目标图风格：带浅框、水平一条。"""
    fig.legend(handles=_handles(include or ARCH_ORDER), loc="lower center",
               bbox_to_anchor=(0.5, y), ncol=ncol, columnspacing=1.6,
               handletextpad=0.55, handlelength=1.9, frameon=True,
               framealpha=0.92, edgecolor="#c4c4c4", fancybox=False)


def solid_legend(fig, ncol=5, y=0.005, keys=None):
    fig.legend(handles=_handles(keys or ARCH_ORDER), loc="lower center",
               bbox_to_anchor=(0.5, y), ncol=ncol, columnspacing=1.6,
               handletextpad=0.55, handlelength=1.9, frameon=True,
               framealpha=0.92, edgecolor="#c4c4c4", fancybox=False)


def gain_arrow(ax, x, y0, y1, pct):
    col = GAIN_POS if pct >= 0 else GAIN_NEG
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=12))
    ax.text(x, (y0 + y1) / 2, f" {pct:+.1f}%", color=col, fontsize=8.8,
            fontweight="bold", va="center", ha="left")


# ---- 模型家族（列维度）。补 Gemini / Claude 实验时只需在此登记模型即可自动加列。
FAMILY = {
    "openai":    dict(label="OpenAI GPT",       logo="openai.png",
                      models=["gpt-4.1-nano", "gpt-5-nano", "gpt-5-mini"]),
    "google":    dict(label="Google Gemini",    logo="gemini.png",
                      models=["gemini-3.5-flash-lite", "gemini-3.7-flash"]),
    "anthropic": dict(label="Anthropic Claude", logo="claude.png", models=[]),
}
FAMILY_ORDER = ["openai", "google", "anthropic"]


def family_of(model):
    for f, d in FAMILY.items():
        if model in d["models"]:
            return f
    return None


# ---- 厂商 logo：不同来源的 PNG 画布尺寸与透明边距都不同（openai 512px 无边距、
# gemini 1024px 含 84px 边距、claude 1200px），统一 zoom 会渲染出一大一小。
# 按 alpha 通道裁到实际内容，再反算 zoom，使所有 logo 的**内容高度**都等于
# LOGO_PT 磅。以后加新厂商无需手工调参。
LOGO_PT = 13.0


def _logo_offsetimage(path, target_pt=LOGO_PT, dpi=72.0):
    """裁掉透明边距并归一化到统一显示高度的 OffsetImage。"""
    import numpy as np
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage
    img = mpimg.imread(str(path))
    if img.ndim == 3 and img.shape[2] == 4:                 # 有 alpha：裁到内容
        ys, xs = np.where(img[..., 3] > 0.04)
        if len(ys):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h_px = img.shape[0]
    return OffsetImage(img, zoom=target_pt / h_px * (dpi / 72.0))


def title_with_logo(ax, fam, text, y=1.035, zoom=0.028, fontsize=11.0, tail=None):
    """标题 + 厂商 logo 打包居中；logo 紧贴厂商名右侧，不挂在标题末尾。"""
    from matplotlib.offsetbox import (AnnotationBbox, TextArea, HPacker)
    import pathlib as _p
    f = _p.Path(__file__).resolve().parents[1] / "paper/figures/logos" / FAMILY[fam]["logo"]
    kids = [TextArea(text, textprops=dict(color=INK, fontsize=fontsize, family="serif"))]
    if f.exists():
        kids.append(_logo_offsetimage(f, target_pt=fontsize * 1.18))
    if tail:
        kids.append(TextArea(tail, textprops=dict(color=INK, fontsize=fontsize, family="serif")))
    box = HPacker(children=kids, align="center", pad=0, sep=6) if len(kids) > 1 else kids[0]
    ax.add_artist(AnnotationBbox(box, (0.5, y), xycoords="axes fraction",
                                 frameon=False, box_alignment=(0.5, 0.0)))


def fig_title_with_logo(fig, fam, text, y=0.985, zoom=0.026, fontsize=10.5):
    """厂商标识作为**整图**总标题出现一次，而不是每个面板挂一个。"""
    from matplotlib.offsetbox import (AnnotationBbox, TextArea, HPacker)
    import pathlib as _p
    f = _p.Path(__file__).resolve().parents[1] / "paper/figures/logos" / FAMILY[fam]["logo"]
    kids = [TextArea(text, textprops=dict(color=INK, fontsize=fontsize, family="serif"))]
    if f.exists():
        kids.append(_logo_offsetimage(f, target_pt=fontsize * 1.18))
    box = HPacker(children=kids, align="center", pad=0, sep=5) if len(kids) > 1 else kids[0]
    ab = AnnotationBbox(box, (0.5, y), xycoords="figure fraction",
                        frameon=False, box_alignment=(0.5, 1.0))
    fig.add_artist(ab)
    return ab
