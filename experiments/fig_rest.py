"""其余五图，统一按 NMI 的视觉语言重做。
编码方案按图分别选择（忠于原作：他们的 Fig 1 用颜色标家族，成本图用颜色标架构）。"""
import sys, pathlib, glob, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from experiments.analyze import load
from experiments.grid_files import main_grid
from experiments.vizstyle import (rcparams, clean, shape_legend, solid_legend, ARCH_MARKER,
                                  ARCH_ORDER, MAS_ORDER, arch_color, ARCH_SOLID, LINE, LINE_SAS,
                                  TIER_ORDER, TIER_LABEL, BENCH_ORDER, BENCH_LABEL, BENCH_COLOR, MODEL_ORDER, TEXT_W,
                                  INK, MUTED, FAINT, GAIN_POS, GAIN_NEG)
from mechanisms.nmi_metrics import config_metrics, fit_turn_powerlaw
from panels.architectures import NMI_CLASS
FIG = ROOT / "paper/figures"


def build():
    rows = load(main_grid())
    c = collections.defaultdict(list)
    for r in rows:
        c[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    return c


# 5 个类别挤进 2.1in 面板：只有三字母缩写能不旋转、不重叠（全称见 caption）
SHORT_ARCH = {"cot": "SAS", "centralized": "Cen", "discussion": "Dec",
              "independent": "Ind", "tiered": "Hyb"}


def acc(v):
    return sum(x["correct"] for x in v) / len(v) * 100 if v else np.nan


# ---------------------------------------------------------------- N-scaling
def fig_nscaling(cells):
    """行=模型（左侧带厂商图标）、列=benchmark，y 轴画相对单医生的增益。

    与图 1 同一套结构。旧版把模型挂在列头、只画 OpenAI 三个模型：模型数一多
    列头就撑破版面，而绝对准确率的 15--95% 量程又会把 ±8pp 的架构差异压平。
    """
    from experiments.vizstyle import vendor_side_label_for_model
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in cells)]
    nrow, ncol = len(models), len(BENCH_ORDER)
    # 按行共享 y，不是全图共享：4.1-nano 在 MedQA 上有一个 -13.6pp 的离群点，
    # 全图共享会把量程拉到 20pp，其余面板 ±5pp 的变化全被压平。
    # 本图回答的是「对这个模型加人管不管用」——同一模型跨 benchmark 比较。
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXT_W, 1.35 * nrow),
                             squeeze=False, sharey="row")
    gy_row = [[] for _ in models]
    for ri, m in enumerate(models):
        for ci, b in enumerate(BENCH_ORDER):
            ax = axes[ri][ci]
            base = acc(cells.get((m, b, "cot", 1), []))
            if np.isnan(base):
                ax.axis("off"); continue
            ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
            for a in MAS_ORDER:
                st = ARCH_MARKER[a]
                Ns = sorted(k[3] for k in cells if k[:3] == (m, b, a))
                if not Ns:
                    continue
                ys = [acc(cells[(m, b, a, N)]) - base for N in Ns]
                gy_row[ri].extend(y for y in ys if not np.isnan(y))
                col = arch_color(b, a)
                ax.plot(Ns, ys, ls="--", lw=1.3, color=col, marker=st["marker"],
                        ms=st["ms"], mfc=col, mec=col, mew=0.0, zorder=4)
            clean(ax)
            ax.set_xticks([1, 3, 5, 7, 9]); ax.set_xlim(0.3, 9.7)
            if ri == 0:
                ax.set_title(BENCH_LABEL.get(b, b), fontsize=9.8, pad=6)
            if ci == 0:
                vendor_side_label_for_model(ax, m)
            if ri == nrow - 1:
                ax.set_xlabel("Number of agents $n_a$", fontsize=9.0)
    for row, ys in zip(axes, gy_row):
        if not ys:
            continue
        lo, hi = min(ys), max(ys); sp = (hi - lo) or 1.0
        for ax in row:
            ax.set_ylim(lo - sp * 0.16, hi + sp * 0.16)
    shape_legend(fig, ncol=5, y=0.004)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig2_nscaling.{e}", dpi=200, bbox_inches="tight")
    print(f"fig2_nscaling ok ({nrow} 模型 x {ncol} benchmark)")


# ---------------------------------------------------------------- 成本-性能
def fig_cost(cells):
    benches = [b for b in BENCH_ORDER if any(k[1] == b for k in cells)]
    fig, axes = plt.subplots(1, len(benches), figsize=(2.10 * len(benches), 2.00), squeeze=False)
    for bi, b in enumerate(benches):
        ax = axes[0][bi]
        for a in ARCH_ORDER:
            st = ARCH_MARKER[a]
            xs, ys, xe, ye = [], [], [], []
            for k, v in cells.items():
                if k[1] != b or k[2] != a or not v:
                    continue
                acc_ = np.array([x["correct"] for x in v]) * 100
                usd = np.array([x["cost"]["usd"] for x in v]) * 1000
                xs.append(usd.mean()); ys.append(acc_.mean())
                xe.append(usd.std() / np.sqrt(len(usd))); ye.append(acc_.std() / np.sqrt(len(acc_)))
            if xs:
                ax.errorbar(xs, ys, xerr=xe, yerr=ye, ls="none", marker=st["marker"],
                            ms=st["ms"], mfc=ARCH_SOLID[a], mec="white", mew=.9,
                            ecolor=ARCH_SOLID[a], elinewidth=1.0, capsize=2.2,
                            alpha=.95, zorder=3)
        clean(ax, grid_axis="both")
        ax.set_title(BENCH_LABEL.get(b, b), pad=7)
        ax.set_xlabel("Cost / 1,000 questions ($)")
        if bi == 0:
            ax.set_ylabel("Performance (%)")
        ax.set_xlim(left=0)
    solid_legend(fig, ncol=5, y=0.005)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig4_cost.{e}", dpi=200, bbox_inches="tight")
    print("fig4_cost ok")


# ---------------------------------------------------------------- 分布箱线图
def fig_box(cells):
    benches = [b for b in BENCH_ORDER if any(k[1] == b for k in cells)]
    fig, axes = plt.subplots(1, len(benches), figsize=(2.10 * len(benches), 2.05), squeeze=False)
    for bi, b in enumerate(benches):
        ax = axes[0][bi]
        data, cols, labs = [], [], []
        for a in ARCH_ORDER:
            if a == "cot":
                vals = [acc(cells[(m, b, "cot", 1)]) for m in TIER_ORDER if (m, b, "cot", 1) in cells]
            else:
                vals = [acc(v) for k, v in cells.items() if k[1] == b and k[2] == a]
            if vals:
                data.append(vals); cols.append(arch_color(b, a))
                labs.append(SHORT_ARCH[a])
        base = np.mean(data[0]) if data else np.nan
        bp = ax.boxplot(data, patch_artist=True, widths=.58, showfliers=False,
                        medianprops=dict(color=INK, lw=1.4),
                        whiskerprops=dict(color="#9a9a9a", lw=1.0),
                        capprops=dict(color="#9a9a9a", lw=1.0))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(.32)
            patch.set_edgecolor(c); patch.set_linewidth(1.3)
        for i, (d, c) in enumerate(zip(data, cols), start=1):
            ax.plot(np.random.RandomState(i).normal(i, .05, len(d)), d, ".",
                    color=c, ms=3.6, alpha=.8, zorder=4)
        if not np.isnan(base) and base:
            rng = max(max(x) for x in data) - min(min(x) for x in data)
            for i, d in enumerate(data[1:], start=2):
                rel = (np.mean(d) - base) / base * 100
                col = GAIN_POS if rel >= 0 else GAIN_NEG
                ax.text(i, max(d) + rng * .05, f"{rel:+.1f}", ha="center",
                        fontsize=7.2, color=col, fontweight="bold")
        clean(ax)
        ax.set_xticklabels(labs, fontsize=7.8)
        ax.set_title(BENCH_LABEL.get(b, b), fontsize=9.8, pad=5)
        if bi == 0:
            ax.set_ylabel("Performance (%)")
        ax.margins(y=.20)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig5_distribution.{e}", dpi=200, bbox_inches="tight")
    print("fig5_distribution ok")


# ---------------------------------------------------------------- 协调动力学
def fig_coord(cells):
    md = []
    for (m, b, a, N), v in cells.items():
        if a in ("cot", "zeroshot"):
            continue
        base = cells.get((m, b, "cot", 1))
        if not base:
            continue
        r = config_metrics(v, base)
        if r:
            r.update(arch=a, cls=NMI_CLASS.get(a, a), model=m, bench=b)
            md.append(r)
    if len(md) < 5:
        print("fig6 数据不足"); return
    # 四面板：A_e 与 absorption 量纲差十倍，共用一根 y 轴会让 A_e 不可读，必须拆开
    fig, axes2 = plt.subplots(2, 2, figsize=(6.30, 4.35))
    axes = axes2.ravel()

    ax = axes[0]
    for a in MAS_ORDER:
        st = ARCH_MARKER[a]
        sub = [(r["n_agents"], r["turns"]) for r in md if r["arch"] == a]
        if sub:
            ax.plot([x[0] for x in sub], [x[1] for x in sub], ls="none", marker=st["marker"],
                    ms=st["ms"], mfc=ARCH_SOLID[a], mec="white", mew=.9, alpha=.9, zorder=4)
    pl = fit_turn_powerlaw([(r["n_agents"], r["turns"]) for r in md])
    xs = np.linspace(1, 10, 60)
    if pl:
        ax.plot(xs, pl["a"] * (xs + .5) ** pl["exponent"], "-", color=INK, lw=1.6, zorder=5)
        ax.plot(xs, 2.72 * (xs + .5) ** 1.724, "--", color="#a8a8a8", lw=1.4, zorder=3)
        ax.text(.04, .97, "ours: $T=%.2f(n{+}0.5)^{%.2f}$,  $R^2$=%.2f"
                % (pl["a"], pl["exponent"], pl["r2"]), transform=ax.transAxes,
                fontsize=7.6, va="top")
        ax.text(.04, .875, "general domain: $2.72(n{+}0.5)^{1.72}$,  $R^2$=0.97",
                transform=ax.transAxes, fontsize=7.6, color="#8a8a8a", va="top")
    ax.set_yscale("log"); clean(ax)
    ax.set_ylim(top=ax.get_ylim()[1] * 5.0)   # 给面板内两行拟合注释腾出顶部空间
    ax.set_xlabel("Number of agents $n_a$"); ax.set_ylabel("Reasoning turns $T$")
    ax.set_title("(a)  Turn-count scaling", loc="left", fontsize=9.6, pad=5)

    ax = axes[1]
    for a in MAS_ORDER:
        st = ARCH_MARKER[a]
        sub = [(r["msg_density"], r["accuracy"] * 100) for r in md if r["arch"] == a]
        if sub:
            ax.plot([x[0] for x in sub], [x[1] for x in sub], ls="none", marker=st["marker"],
                    ms=st["ms"], mfc=ARCH_SOLID[a], mec="white", mew=.9, alpha=.9, zorder=4)
    ax.axvline(.39, color=GAIN_NEG, ls="--", lw=1.3, zorder=3)
    clean(ax)
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + (y1 - y0) * .16)
    ax.text(.47, y1 + (y1 - y0) * .12, "plateau $c^*{=}0.39$", fontsize=7.6,
            color=GAIN_NEG, ha="left", va="top")
    ax.set_xlabel("Message density $c$"); ax.set_ylabel("Performance (%)")
    ax.set_title("(b)  Message density", loc="left", fontsize=9.6, pad=5)

    order = ["Independent", "Centralized", "Decentralized", "Hybrid"]
    key = {"Independent": "independent", "Centralized": "centralized",
           "Decentralized": "discussion", "Hybrid": "tiered"}
    cls = [c for c in order if any(r["cls"] == c for r in md)]
    cc = [ARCH_SOLID[key[c]] for c in cls]
    xp = np.arange(len(cls))
    short = [c.replace("Decentralized", "Decentr.").replace("Independent", "Independ.") for c in cls]

    ax = axes[2]
    ae = [np.mean([r["error_amp"] for r in md if r["cls"] == c]) for c in cls]
    ax.bar(xp, ae, .58, color=cc, zorder=3, edgecolor="white", lw=.7)
    for i2, v in enumerate(ae):
        ax.text(i2, v + max(ae) * .035, f"{v:.2f}", ha="center", fontsize=8.4, color=INK)
    clean(ax, grid_axis="y"); ax.set_xticks(xp); ax.set_xticklabels(short, fontsize=8.4)
    ax.set_ylabel("$A_e$  (information discarded)"); ax.margins(y=.18)
    ax.set_title("(c)  Error amplification", loc="left", fontsize=9.6, pad=5)

    ax = axes[3]
    ab = [np.mean([r["absorb"] for r in md if r["cls"] == c]) * 100 for c in cls]
    ax.bar(xp, ab, .58, color=cc, zorder=3, edgecolor="white", lw=.7, hatch="///")
    for i2, v in enumerate(ab):
        ax.text(i2, v + .5, f"{v:.1f}%", ha="center", fontsize=8.4, color=INK)
    ax.axhline(22.7, color=GAIN_NEG, ls="--", lw=1.2, zorder=4)
    ax.text(-.42, 23.6, "reported 22.7%", fontsize=7.8, color=GAIN_NEG, va="bottom")
    clean(ax, grid_axis="y"); ax.set_xticks(xp); ax.set_xticklabels(short, fontsize=8.4)
    ax.set_ylabel("Error absorption (%)"); ax.set_ylim(0, 27)
    ax.set_title("(d)  Error absorption", loc="left", fontsize=9.6, pad=5)

    shape_legend(fig, ncol=4, y=0.004, include=MAS_ORDER)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig6_coordination.{e}", dpi=200, bbox_inches="tight")
    print("fig6_coordination ok")


if __name__ == "__main__":
    rcparams()
    c = build()
    fig_nscaling(c); fig_cost(c); fig_box(c); fig_coord(c)
