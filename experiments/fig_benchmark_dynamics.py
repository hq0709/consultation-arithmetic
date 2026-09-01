"""C1 —— 对标 NMI Figure 6: Benchmark-specific scaling dynamics。
各 benchmark 上「最佳 MAS 变体 vs 单智能体基线」随能力指数的演化，标注相对增益百分比。"""
import sys, pathlib, glob, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.analyze import load
from experiments.grid_files import main_grid
from experiments.vizstyle import (rcparams, clean, BENCH_COLOR, BENCH_LABEL, BENCH_ORDER,
                                  TIER_ORDER, TIER_LABEL, CAPABILITY, INK, MUTED,
                                  GAIN_POS, GAIN_NEG, LINE_SAS)
FIG = ROOT / "paper/figures"
MAS = ("independent", "centralized", "discussion", "tiered")


def main():
    rcparams()
    rows = load(main_grid())
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    benches = [b for b in BENCH_ORDER if any(k[1] == b for k in cells)]
    fig, axes = plt.subplots(1, len(benches), figsize=(2.10 * len(benches), 2.00), squeeze=False)
    for bi, b in enumerate(benches):
        ax = axes[0][bi]
        col = BENCH_COLOR[b]
        models = [m for m in TIER_ORDER if (m, b, "cot", 1) in cells]
        xs = [CAPABILITY[m] for m in models]
        sas, best, bestname = [], [], []
        for m in models:
            v = cells[(m, b, "cot", 1)]
            sas.append(sum(x["correct"] for x in v) / len(v) * 100)
            cand = []
            for a in MAS:
                for N in sorted(k[3] for k in cells if k[:3] == (m, b, a)):
                    vv = cells[(m, b, a, N)]
                    cand.append((sum(x["correct"] for x in vv) / len(vv) * 100, a, N))
            if cand:
                v_, a_, n_ = max(cand); best.append(v_); bestname.append(f"{a_[:5]}/{n_}")
            else:
                best.append(np.nan); bestname.append("")
        ax.fill_between(xs, sas, best, color=col, alpha=.17, zorder=1)
        ax.plot(xs, sas, "--o", color=LINE_SAS, ms=6.5, lw=1.2, mfc="white", mew=1.3, zorder=3)
        ax.plot(xs, best, "-s", color=col, ms=6.5, lw=1.6, mec="white", mew=1.1, zorder=4)
        for x, s_, b_ in zip(xs, sas, best):
            if np.isnan(b_):
                continue
            rel = (b_ - s_) / s_ * 100
            c = GAIN_POS if rel >= 0 else GAIN_NEG
            ax.annotate(f"{rel:+.1f}%", xy=(x, max(s_, b_)), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8.6,
                        color=c, fontweight="bold")
        lo = min([v for v in sas + best if not np.isnan(v)])
        for m, x in zip(models, xs):
            ax.annotate(TIER_LABEL[m], xy=(x, lo), xytext=(0, -13), textcoords="offset points",
                        ha="center", va="top", fontsize=8.4, color=INK, fontweight="bold")
        clean(ax)
        span = max(xs) - min(xs)
        ax.set_xlim(min(xs) - span * .20, max(xs) + span * .20)
        ax.margins(y=.26)
        ax.set_title(BENCH_LABEL.get(b, b), pad=8)
        ax.set_xlabel("Capability index $I$")
        if bi == 0:
            ax.set_ylabel("Performance (%)")
    fig.legend(handles=[
        Line2D([], [], color=LINE_SAS, marker="o", ms=6.5, ls="--", lw=1.2, mfc="white",
               mew=1.3, label="Single-agent baseline"),
        Line2D([], [], color="#6a6a6a", marker="s", ms=6.5, ls="-", lw=1.6,
               label="Best multi-agent variant")],
        loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=2, columnspacing=2.4)
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig6_benchmark_dynamics.{e}", dpi=200, bbox_inches="tight")
    print("fig6_benchmark_dynamics ok")


if __name__ == "__main__":
    main()
