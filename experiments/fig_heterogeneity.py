"""Fig 4 —— 对标 NMI Figure 4: Agent Heterogeneity Effects。
分组柱状：Centralized / Decentralized x 五种能力配置，标注相对同质高能力的变化。"""
import sys, pathlib, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from experiments.analyze import wilson
from experiments.vizstyle import (rcparams, clean, INK, MUTED, GAIN_POS, GAIN_NEG,
                                  ARCH_SOLID, C_PURPLE, C_CYAN)
FIG = ROOT / "paper/figures"

ORDER = ["homog-high", "lo-orch/hi-sub", "hi-orch/lo-sub", "homog-low", "mixed-panel"]
LABEL = {"homog-high": "all\nstrong", "homog-low": "all\nweak",
         "hi-orch/lo-sub": "strong\nattending", "lo-orch/hi-sub": "strong\nspecialists",
         "mixed-panel": "mixed\npanel"}
ARCHL = {"centralized": "Centralized (attending + specialists)",
         "discussion": "Decentralized (peer discussion)"}


def main():
    rcparams()
    f = ROOT / "results/H_heterogeneity.jsonl"
    rows = [json.loads(l) for l in f.open() if l.strip()]
    rows = [r for r in rows if r.get("status") == "ok"]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["arch"], r["hetero"])].append(r)
    arches = [a for a in ("centralized", "discussion") if any(k[0] == a for k in cells)]
    fig, axes = plt.subplots(1, len(arches), figsize=(3.10 * len(arches), 2.45), squeeze=False)
    for ai, arch in enumerate(arches):
        ax = axes[0][ai]
        names = [n for n in ORDER if (arch, n) in cells]
        # 去中心化没有 orchestrator，orchestrator 的能力操纵在结构上不适用：
        # 那两个条件会退化成同质高/同质低。画成独立柱子会误导读者。
        if arch == "discussion":
            names = [n for n in names if n not in ("hi-orch/lo-sub", "lo-orch/hi-sub")]
        accs, los, his, ns = [], [], [], []
        for n in names:
            v = cells[(arch, n)]
            p, lo, hi = wilson(sum(x["correct"] for x in v), len(v))
            accs.append(p * 100); los.append((p - lo) * 100); his.append((hi - p) * 100); ns.append(len(v))
        base = accs[names.index("homog-high")] if "homog-high" in names else np.nan
        cols = []
        for n in names:
            if n == "homog-high":
                cols.append(C_PURPLE)
            elif n == "homog-low":
                cols.append("#d9d2ee")
            else:
                cols.append(C_CYAN)
        xp = np.arange(len(names))
        ax.bar(xp, accs, .62, color=cols, yerr=[los, his], capsize=3,
               error_kw=dict(ecolor="#3a3a3a", lw=1.0), zorder=3, edgecolor="white", lw=.8)
        for i, (n, a_) in enumerate(zip(names, accs)):
            if n == "homog-high" or np.isnan(base) or not base:
                continue
            rel = (a_ - base) / base * 100
            c = GAIN_POS if rel >= 0 else GAIN_NEG
            # 抬高标注，避开同质高的参考虚线
            yy = a_ + his[i] + 1.2
            if abs(yy - base) < 2.5:
                yy = base + 3.0
            ax.text(i, yy, f"{rel:+.0f}%", ha="center", fontsize=8.0, color=c, fontweight="bold")
        if not np.isnan(base):
            ax.axhline(base, ls="--", lw=1.1, color=C_PURPLE, zorder=2, alpha=.85)
        clean(ax, grid_axis="y")
        ax.set_xticks(xp)
        ax.set_xticklabels([LABEL[n] for n in names], fontsize=8.0, linespacing=1.15)
        ax.set_title(ARCHL[arch], pad=5, fontsize=9.6)
        if ai == 0:
            ax.set_ylabel("Performance (%)")
        ax.margins(y=.20)
    fig.legend(handles=[Patch(facecolor=C_PURPLE, label="all agents strong"),
                        Patch(facecolor=C_CYAN, label="mixed capability"),
                        Patch(facecolor="#d9d2ee", label="all agents weak")],
               loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, columnspacing=2.0, fontsize=8.4)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig4_heterogeneity.{e}", dpi=200, bbox_inches="tight")
    print("fig4_heterogeneity ok")
    print("\n=== 能力放置的不对称 ===")
    for arch in arches:
        g = {n: sum(x["correct"] for x in cells[(arch, n)]) / len(cells[(arch, n)]) * 100
             for n in ORDER if (arch, n) in cells}
        if "homog-high" in g and "homog-low" in g:
            span = g["homog-high"] - g["homog-low"]
            print(f"\n{ARCHL[arch]}")
            print(f"  同质高 {g['homog-high']:.1f}%  同质低 {g['homog-low']:.1f}%  跨度 {span:.1f}pp")
            if "hi-orch/lo-sub" in g:
                print(f"  强主诊+弱专科 {g['hi-orch/lo-sub']:.1f}%  "
                      f"(相对同质低 {g['hi-orch/lo-sub']-g['homog-low']:+.1f}pp, "
                      f"挽回跨度 {(g['hi-orch/lo-sub']-g['homog-low'])/span*100:.0f}%)")
            if "lo-orch/hi-sub" in g:
                print(f"  弱主诊+强专科 {g['lo-orch/hi-sub']:.1f}%  "
                      f"(相对同质高 {g['lo-orch/hi-sub']-g['homog-high']:+.1f}pp, "
                      f"保留跨度 {(g['lo-orch/hi-sub']-g['homog-low'])/span*100:.0f}%)")


if __name__ == "__main__":
    main()
