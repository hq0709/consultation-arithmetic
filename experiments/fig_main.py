"""Fig 1 —— 复刻 NMI Figure 1：列 = 模型家族（标题带厂商 logo），行 = benchmark。
灰色细连线 + 彩色 marker，形状编码架构，同色系明度台阶，模型名粗体直标，绿/红增益箭头。
只有已登记模型的家族会出现，因此补 Gemini / Claude 实验后自动扩展为 3 列。"""
import sys, pathlib, glob, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib.pyplot as plt
from experiments.analyze import load
from experiments.grid_files import main_grid
from experiments.vizstyle import series as vs_series
from experiments.vizstyle import (_logo_offsetimage, rcparams, clean, shape_legend, GAIN_POS, GAIN_NEG, ARCH_MARKER,
                                  ARCH_ORDER, MAS_ORDER, arch_color, LINE, LINE_SAS,
                                  TIER_LABEL, BENCH_ORDER, BENCH_LABEL, CAPABILITY, INK, MUTED, TEXT_W,
                                  FAMILY, FAMILY_ORDER, title_with_logo, fig_title_with_logo)
FIG = ROOT / "paper/figures"
# x 刻度用短名，避免 I=50.8 与 59.2 两点的标签相撞
TICK_LABEL = {"gpt-4.1-nano": "4.1-nano", "gpt-5-nano": "5-nano",
              "gpt-5-mini": "5-mini",
              "gemini-3.5-flash-lite": "3.5-lite", "gemini-3.7-flash": "3.7-flash",
              "claude-haiku-4-5-20251001": "haiku-4.5", "claude-sonnet-5": "sonnet-5",
              "deepseek-v4-flash": "V4-flash", "deepseek-v4-pro": "V4-pro"}


SHORT_FAM = {"openai": "OpenAI", "google": "Google", "anthropic": "Anthropic",
             "deepseek": "DeepSeek"}


def vendor_side_label(ax, fam):
    """厂商名竖排在面板左外侧，图标叠在名字上方。

    厂商名放在标题行会占掉每行约 0.4in 的垂直空间，四个厂商就撑破版面；
    放到左侧则不占高度。"""
    from matplotlib.offsetbox import AnnotationBbox
    import pathlib as _p
    ax.annotate(SHORT_FAM.get(fam, FAMILY[fam]["label"]), xy=(-0.42, 0.5), xycoords="axes fraction",
                rotation=90, ha="center", va="center", fontsize=9.6, color=INK)
    lg = FAMILY[fam].get("logo")
    f = (_p.Path(__file__).resolve().parents[1] / "paper/figures/logos" / lg) if lg else None
    if f is not None and f.exists():
        ax.add_artist(AnnotationBbox(_logo_offsetimage(f, target_pt=11.0),
                                     (-0.42, 1.03), xycoords="axes fraction",
                                     frameon=False, box_alignment=(0.5, 0.0)))


def main():
    rcparams()
    # 与全文分析同一个网格定义：图里出现的厂商就是分析里用的厂商。
    # 各画各的会让图上多出没进任何分析的厂商，也把版面撑坏。
    rows = load(main_grid())

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    have = {k[0] for k in cells}
    fams = [f for f in FAMILY_ORDER if any(m in have for m in FAMILY[f]["models"])]
    benches = [b for b in BENCH_ORDER if any(k[1] == b for k in cells)]
    nc, nr = len(fams), len(benches)
    # 只有一个家族时（当前：仅 OpenAI），把 benchmark 摊成一行更好读；
    # 补齐 Gemini / Claude 后自动切回「列=家族、行=benchmark」的网格。
    single = (nc == 1)
    if single:
        # 三个 benchmark 的绝对刻度差得太远（15-60 / 20-47 / 66-94），只看上排
        # 无法比较架构之间的差异；下排画相对单医生的增益，共用一根刻度。
        fig, axes0 = plt.subplots(2, nr, figsize=(2.10 * nr, 3.55), squeeze=False,
                                  gridspec_kw={"height_ratios": [1.25, 1]})
        axes = [[axes0[0][i]] for i in range(nr)]
        gain_axes = [axes0[1][i] for i in range(nr)]
    else:
        # 行=厂商、列=benchmark。厂商名竖排在最左，省掉每行的标题空间，
        # 所以厂商数增加时不必压面板高度。宽度钉死在 \textwidth。
        nr, nc = len(fams), len(benches)
        fig, axes = plt.subplots(nr, nc, figsize=(TEXT_W, 1.78 * nr), squeeze=False,
                                 sharey=True)

    gy_all = []
    for ri, fam in enumerate(fams):
        for ci, b in enumerate(benches):
            ax = axes[ri][ci]
            models = [m for m in FAMILY[fam]["models"]
                      if (m, b, "cot", 1) in cells and m in CAPABILITY]
            if not models:
                ax.axis("off"); continue
            xs = [CAPABILITY[m] for m in models]
            series = {}
            for a in ARCH_ORDER:
                ys = []
                for m in models:
                    if a == "cot":
                        v = cells.get((m, b, "cot", 1), [])
                        ys.append(sum(x["correct"] for x in v) / len(v) * 100 if v else np.nan)
                    else:
                        Ns = [k[3] for k in cells if k[:3] == (m, b, a)]
                        ys.append(max(sum(x["correct"] for x in cells[(m, b, a, N)]) /
                                      len(cells[(m, b, a, N)]) * 100 for N in Ns) if Ns else np.nan)
                series[a] = ys
            # 画相对单医生的增益，不画绝对准确率：绝对量程 15--95% 会把架构之间
            # 仅 ±8pp 的差异压成一条线，五条曲线全部重叠，图就失去了意义。
            base = series["cot"]
            for a in ARCH_ORDER:
                series[a] = [v - b0 if not (np.isnan(v) or np.isnan(b0)) else np.nan
                             for v, b0 in zip(series[a], base)]
            xspan = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
            dodge = {a: (i - 2) * xspan * 0.012 for i, a in enumerate(ARCH_ORDER)}
            # 目标图的线型分工：连线与 marker 同色；单医生基线是灰虚参考线。
            for a in ARCH_ORDER:
                vs_series(ax, [x + dodge[a] for x in xs], series[a], a)
            ax.axhline(0, color=MUTED, lw=0.9, ls="-", zorder=1)
            gy_all.extend(v for ys in series.values() for v in ys if not np.isnan(v))
            ax.set_xlim(min(xs) - xspan * 0.13, max(xs) + xspan * 0.13)
            # 模型名下沉为 x 刻度（目标图：x 轴是干净的数值轴，没有浮在数据里的标注）
            ax.set_xticks(xs)
            ax.set_xticklabels([TICK_LABEL[m] for m in models], fontsize=7.4)
            xi = len(models) - 1
            cand = [series[a][xi] for a in MAS_ORDER if not np.isnan(series[a][xi])]
            if cand:
                y1 = max(cand)
                ax.text(0.035, 0.955, f"best panel {y1:+.1f} pp",
                        transform=ax.transAxes, ha="left", va="top",
                        # 负增益必须用负色：绿色的 "-2.0 pp" 会被读成好消息
                        fontsize=8.0, fontweight="bold",
                        color=GAIN_POS if y1 >= 0 else GAIN_NEG)
            clean(ax)
            if single:
                ax.set_title(BENCH_LABEL.get(b, b), fontsize=10.0, pad=5)
                if ri == 0:
                    ax.set_ylabel("Performance (%)")
                if not single:
                    ax.set_xlabel("GPT model  (capability index $I$)", fontsize=9.0)
            else:
                if ri == 0:
                    ax.set_title(BENCH_LABEL.get(b, b), fontsize=10.2, pad=6)
                if ci == 0:
                    vendor_side_label(ax, fam)
                # x 轴刻度就是模型名，自解释，不再另加轴标题
    if not single and gy_all:
        lo, hi = min(gy_all), max(gy_all); sp = (hi - lo) or 1.0
        for row in axes:
            for ax in row:
                ax.set_ylim(lo - sp * 0.12, hi + sp * 0.12)
    if single:
        gy = []
        for ri, b in enumerate(benches):
            gax = gain_axes[ri]
            models = [m for m in FAMILY[fams[0]]["models"] if (m, b, "cot", 1) in cells]
            xs = [CAPABILITY[m] for m in models]
            base = []
            for m in models:
                v = cells.get((m, b, "cot", 1), [])
                base.append(sum(x["correct"] for x in v) / len(v) * 100 if v else np.nan)
            xspan = (max(xs) - min(xs)) or 1.0
            dodge = {a: (i - 1.5) * xspan * 0.012 for i, a in enumerate(MAS_ORDER)}
            for a in MAS_ORDER:
                ys = []
                for mi, m in enumerate(models):
                    Ns = [k[3] for k in cells if k[:3] == (m, b, a)]
                    ys.append(max(sum(x["correct"] for x in cells[(m, b, a, N)]) /
                                  len(cells[(m, b, a, N)]) * 100 for N in Ns) - base[mi]
                              if Ns else np.nan)
                gy += [v for v in ys if not np.isnan(v)]
                vs_series(gax, [x + dodge[a] for x in xs], ys, a)
            gax.axhline(0, color="#4a4a4a", lw=1.0, zorder=3)
            gax.set_xticks(xs)
            gax.set_xticklabels([TICK_LABEL[m] for m in models], fontsize=7.4)
            gax.set_xlim(min(xs) - xspan * 0.16, max(xs) + xspan * 0.16)
            gax.set_xlabel("GPT model  (capability index $I$)", fontsize=9.0)
            if ri == 0:
                gax.set_ylabel("Gain over\nsingle doctor (pp)", fontsize=9.0, linespacing=1.3)
            clean(gax)
        lo, hi = min(gy), max(gy); pad = (hi - lo) * .12
        for gax in gain_axes:
            gax.set_ylim(lo - pad, hi + pad)          # 共享刻度：架构差异可直接横向比较
    shape_legend(fig, ncol=5, y=0.004)
    fig.tight_layout(rect=[0, 0.075 if single else 0.075, 1, 0.955 if single else 1])
    if single:
        # 厂商标识只出现一次，作为整图总标题（不是每个面板都挂一个）
        fig_title_with_logo(fig, fams[0], FAMILY[fams[0]]["label"], y=0.998)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_capability.{e}", dpi=200, bbox_inches="tight")
    print(f"fig1_capability ok  ({nc} 家族 x {nr} benchmark)")


if __name__ == "__main__":
    main()
