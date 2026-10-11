"""附录的面板规模曲线（fig2_nscaling / fig2_nscalingb），以及 nature_fig/ 共用的数据入口 build、acc。
架构图、分布、协调、异质性与 φ 图由 experiments/nature_fig/ 生成；成本图已从论文删除。"""
import sys, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib.pyplot as plt
from experiments.grid_files import load_main
from experiments.vizstyle import (rcparams, clean, shape_legend, ARCH_MARKER, MAS_ORDER, arch_color,
                                  BENCH_ORDER, BENCH_LABEL, MODEL_ORDER, TEXT_W, MUTED)
FIG = ROOT / "paper/figures"


def build():
    rows = load_main()
    c = collections.defaultdict(list)
    for r in rows:
        c[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    return c


def acc(v):
    return sum(x["correct"] for x in v) / len(v) * 100 if v else np.nan


# ---------------------------------------------------------------- N-scaling
def fig_nscaling(cells):
    """行=模型（左侧带厂商图标）、列=benchmark，y 轴画相对单医生的增益。

    与图 1 同一套结构。旧版把模型挂在列头、只画 OpenAI 三个模型：模型数一多
    列头就撑破版面，而绝对准确率的 15--95% 量程又会把 ±8pp 的架构差异压平。
    """
    from experiments.vizstyle import vendor_side_label_for_model, FAMILY, FAMILY_ORDER
    allm = [m for m in MODEL_ORDER if any(k[0] == m for k in cells)]
    # 八个模型挤一页要把每行压到 0.92in，面板不到一英寸高，曲线全糊在一起。
    # 按厂商切成两张，每行保持 1.35in：OpenAI+Google 五行，Anthropic+DeepSeek 三行。
    groups = [[m for m in allm if any(m in FAMILY[f]["models"] for f in ("openai", "google"))],
              [m for m in allm if any(m in FAMILY[f]["models"] for f in ("anthropic", "deepseek"))]]
    for gi, models in enumerate(g for g in groups if g):
        _nscaling_panel(cells, models, gi, vendor_side_label_for_model)


def _nscaling_panel(cells, models, gi, vendor_side_label_for_model):
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
    suf = "" if gi == 0 else "b"
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig2_nscaling{suf}.{e}", dpi=200, bbox_inches="tight")
    print(f"fig2_nscaling{suf} ok ({nrow} 模型 x {ncol} benchmark)")


if __name__ == "__main__":
    rcparams()
    c = build()
    fig_nscaling(c)
