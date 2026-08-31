"""Fig 1 —— 复刻 NMI Figure 1：列 = 模型家族（标题带厂商 logo），行 = benchmark。
灰色细连线 + 彩色 marker，形状编码架构，同色系明度台阶，模型名粗体直标，绿/红增益箭头。
只有已登记模型的家族会出现，因此补 Gemini / Claude 实验后自动扩展为 3 列。"""
import sys, pathlib, glob, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib.pyplot as plt
from experiments.analyze import load
from experiments.vizstyle import series as vs_series
from experiments.vizstyle import (rcparams, clean, shape_legend, GAIN_POS, ARCH_MARKER,
                                  ARCH_ORDER, MAS_ORDER, arch_color, LINE, LINE_SAS,
                                  TIER_LABEL, BENCH_ORDER, BENCH_LABEL, CAPABILITY, INK, MUTED,
                                  FAMILY, FAMILY_ORDER, title_with_logo, fig_title_with_logo)
FIG = ROOT / "paper/figures"
# x 刻度用短名，避免 I=50.8 与 59.2 两点的标签相撞
TICK_LABEL = {"gpt-4.1-nano": "4.1-nano", "gpt-5-nano": "5-nano",
              "gpt-5-mini": "5-mini",
              "gemini-3.5-flash-lite": "3.5-lite", "gemini-3.7-flash": "3.7-flash"}


def main():
    rcparams()
    # Gemini 网格写的是 GEM_*.jsonl，与 OpenAI 的 G_*.jsonl 是两个独立网格。
    # 只有这张能力图把两者并列（列 = 厂商家族）；其余分析仍只用 OpenAI 网格，
    # 否则 "180 个配置"、窗口划分、phi=0.734 等主结果的口径会被静默改掉。
    rows = load(sorted(glob.glob(str(ROOT / "results/G_*.jsonl")))
                + sorted(glob.glob(str(ROOT / "results/GEM_*.jsonl"))))
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
        fig, axes = plt.subplots(nr, nc, figsize=(2.10 * nc + 0.3, 1.85 * nr), squeeze=False)

    for ri, b in enumerate(benches):
        for ci, fam in enumerate(fams):
            ax = axes[ri][ci]
            models = [m for m in FAMILY[fam]["models"] if (m, b, "cot", 1) in cells]
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
            xspan = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
            dodge = {a: (i - 2) * xspan * 0.012 for i, a in enumerate(ARCH_ORDER)}
            # 目标图的线型分工：连线与 marker 同色；单医生基线是灰虚参考线。
            for a in ARCH_ORDER:
                vs_series(ax, [x + dodge[a] for x in xs], series[a], a)
            allv = [v for ys in series.values() for v in ys if not np.isnan(v)]
            lo, hi = min(allv), max(allv); span = (hi - lo) or 1.0
            ax.set_ylim(lo - span * 0.18, hi + span * 0.40)
            ax.set_xlim(min(xs) - xspan * 0.13, max(xs) + xspan * 0.13)
            # 模型名下沉为 x 刻度（目标图：x 轴是干净的数值轴，没有浮在数据里的标注）
            ax.set_xticks(xs)
            ax.set_xticklabels([TICK_LABEL[m] for m in models], fontsize=7.4)
            xi = len(models) - 1
            s0 = series["cot"][xi]
            cand = [(series[a][xi], a) for a in MAS_ORDER if not np.isnan(series[a][xi])]
            if cand and not np.isnan(s0):
                y1, _ = max(cand)
                # 左上角是空的（数据自左下往右上走）；箭头挤在右侧会压住数据点，
                # 而且为它留的边距把三个模型压进面板左侧 70%，刻度因此相撞。
                ax.text(0.035, 0.965, f"best panel {y1 - s0:+.1f} pp",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=8.2, color=GAIN_POS, fontweight="bold")
            clean(ax)
            if single:
                ax.set_title(BENCH_LABEL.get(b, b), fontsize=10.0, pad=5)
                if ri == 0:
                    ax.set_ylabel("Performance (%)")
                if not single:
                    ax.set_xlabel("GPT model  (capability index $I$)", fontsize=9.0)
            else:
                if ri == 0:
                    title_with_logo(ax, fam, FAMILY[fam]["label"], y=1.045, zoom=0.030,
                                    fontsize=11.5)
                if ci == 0:
                    ax.set_ylabel(f"{BENCH_LABEL.get(b, b)}\nPerformance (%)", fontsize=9.2)
                if ri == nr - 1:
                    ax.set_xlabel("GPT model  (capability index $I$)", fontsize=9.0)
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
