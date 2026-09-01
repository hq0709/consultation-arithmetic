"""fig7 效应量毛虫图 · fig8 独立性算术 · fig9 校准。

三张图按目标视觉语言重做，关键改动：
  fig7  竖版 32 行→**横版双栏毛虫图**，按效应量排序，每行有名字；颜色只用两种
        （单医生 25–50% / 其余），显著性用实心/空心。原版 34% 的墨迹全是背景色带。
  fig8  砍掉"半个面板放两个数字"的斜率图，换成 φ 等值线族 + 120 个配置的 φ 分布。
  fig9  可靠性图的对角线跑出画布 → 改画**校准残差**（置信 − 准确率），零线永远可见。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.analyze import load, mcnemar
from experiments.grid_files import main_grid
from experiments.vizstyle import (rcparams, clean, panel_legend, ARCH_MARKER, MAS_ORDER,
                                  ARCH_SOLID, INK, MUTED, GRID, C_ORANGE, C_ROSE,
                                  C_PURPLE, C_CYAN)
FIG = ROOT / "paper/figures"
MAS = ("independent", "centralized", "discussion", "tiered")
AL = {"independent": "Independent", "centralized": "Centralized",
      "discussion": "Decentralized", "tiered": "Hybrid"}
SHORT_B = {"medxpertqa": "MedXpert", "medagentsbench": "MedAgents", "medqa": "MedQA"}
SHORT_M = {"gpt-4.1-nano": "4.1-nano", "gpt-5-nano": "5-nano", "gpt-5-mini": "5-mini",
           "claude-haiku-4-5-20251001": "haiku-4.5", "claude-sonnet-5": "sonnet-5",
           "gemini-3.5-flash-lite": "G3.5-lite", "gemini-3.7-flash": "G3.7-flash"}
SHORT_A = {"independent": "Ind", "centralized": "Cen",
           "discussion": "Dec", "tiered": "Hyb"}
SIG, NS = "#d9962f", "#9a9a9a"      # 只有两种颜色：显著 / 不显著


def collect():
    rows = load(main_grid())
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    recs = []
    for (m, b, a, N), v in cells.items():
        if a not in MAS:
            continue
        base = cells.get((m, b, "cot", 1))
        if not base:
            continue
        ids = {x["qid"] for x in v} & {x["qid"] for x in base}
        if len(ids) < 50:
            continue
        A = {x["qid"]: x["correct"] for x in v if x["qid"] in ids}
        B = {x["qid"]: x["correct"] for x in base if x["qid"] in ids}
        n = len(ids)
        d = [A[q] - B[q] for q in ids]
        g = float(np.mean(d)); se = float(np.std(d, ddof=1)) / np.sqrt(n)
        _, _, p = mcnemar(B, A)
        recs.append(dict(model=m, bench=b, arch=a, N=N, n=n,
                         psa=sum(B.values()) / n, gain=g * 100,
                         lo=(g - 1.96 * se) * 100, hi=(g + 1.96 * se) * 100, p=p))
    return recs, cells


# ------------------------------------------------------------------ fig7
def forest(recs):
    """横版双栏毛虫图：按效应量降序，左栏 rank 1..k，右栏 rank k+1..2k。
    读者一眼看到的是琥珀色全部落在正侧、灰色全部落在负侧 —— 这就是论文的主张本身。"""
    rcparams()
    best = {}
    for r in recs:
        k = (round(r["psa"] * 100), r["arch"])
        if k not in best or r["gain"] > best[k]["gain"]:
            best[k] = r
    sel = sorted(best.values(), key=lambda r: -r["gain"])
    half = (len(sel) + 1) // 2
    chunks = [sel[:half], sel[half:]]
    xlo = min(r["lo"] for r in sel); xhi = max(r["hi"] for r in sel)
    pad = (xhi - xlo) * 0.06

    fig, axes = plt.subplots(1, 2, figsize=(6.30, 0.145 * half + 0.95))
    for ci, (ax, chunk) in enumerate(zip(axes, chunks)):
        ys = np.arange(len(chunk))[::-1]
        for y, r in zip(ys, chunk):
            ins = 0.25 <= r["psa"] < 0.50
            c = SIG if ins else NS
            sig = r["p"] < 0.05
            ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=1.2,
                    solid_capstyle="round", zorder=3)
            ax.plot([r["gain"]], [y], marker="o", ms=4.6,
                    mfc=c if sig else "white", mec=c, mew=1.2, zorder=4)
        ax.axvline(0, color="#4a4a4a", lw=0.9, zorder=2)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{SHORT_B[r['bench']]} · {SHORT_M[r['model']]} · "
                            f"{SHORT_A[r['arch']]} $N{{=}}${r['N']}"
                            for r in chunk], fontsize=6.2)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.set_xlim(xlo - pad, xhi + pad)
        ax.set_ylim(-0.8, len(chunk) - 0.2)
        clean(ax, grid_axis="x")
        ax.set_xlabel("Collaboration gain (pp, 95% CI)", fontsize=8.6)
        ax.set_title(f"ranked by effect size: {ci * half + 1}\u2013"
                     f"{ci * half + len(chunk)} of {len(sel)}",
                     loc="left", fontsize=8.0, pad=4, color=MUTED)

    h = [Line2D([], [], color=SIG, lw=1.2, marker="o", ms=4.4, mfc=SIG, mec=SIG,
                label="single doctor 25–50%"),
         Line2D([], [], color=NS, lw=1.2, marker="o", ms=4.4, mfc=NS, mec=NS,
                label="outside that range"),
         Line2D([], [], ls="none", marker="o", ms=4.4, mfc=INK, mec=INK, label="$p<0.05$"),
         Line2D([], [], ls="none", marker="o", ms=4.4, mfc="white", mec=INK, mew=1.1,
                label="n.s.")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               columnspacing=1.5, handletextpad=0.5, fontsize=7.6, frameon=True,
               framealpha=0.92, edgecolor="#c4c4c4", fancybox=False)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig7_forest.{e}", dpi=200, bbox_inches="tight")
    print(f"fig7_forest ok ({len(sel)} 行, 双栏)")


# ------------------------------------------------------------------ fig8
def slope():
    """(a) N_eff(N, φ) 等值线族 + 实测曲线：把实测放进整个可能空间里看。
    (b) 120 个配置的 φ 分布 —— φ≈0.73 不是某一格的巧合，是全网格的常数。"""
    rcparams()
    ind = json.loads((ROOT / "results/independence.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.45),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) 等值线族
    ax = axes[0]
    Ns = np.linspace(1, 9, 120)
    for phi, lab in [(0.0, r"$\varphi=0$  (independent)"), (0.25, r"$0.25$"),
                     (0.50, r"$0.50$"), (0.75, r"$0.75$"), (0.95, r"$0.95$")]:
        ax.plot(Ns, Ns / (1 + (Ns - 1) * phi), ls=":", lw=1.0, color="#b0b0b0", zorder=2)
        yv = 9 / (1 + 8 * phi)
        ax.text(9.35, yv, lab if phi == 0.0 else f"$\\varphi={phi:g}$",
                fontsize=6.8, color=MUTED, va="center", ha="left")
    # 与正文同口径：只用每题都召集完整面板的三种架构（Hybrid 的门控只开 12.4%，
    # 它的 phi 测自更难的子集，混进均值会把曲线拉低，与正文的 0.734 对不上）
    FULL_PANEL = ("independent", "centralized", "discussion")
    by = collections.defaultdict(list)
    for r in ind:
        if r["arch"] in FULL_PANEL and r["N"] >= 3:
            by[r["N"]].append(r["phi0"])
    xs = sorted(by)
    ys = [n / (1 + (n - 1) * float(np.mean(by[n]))) for n in xs]
    phibar = float(np.mean([v for l in by.values() for v in l]))
    ax.plot(xs, ys, ls="--", lw=2.3, color=C_PURPLE, marker="^", ms=6.4,
            mfc=C_PURPLE, mec=C_PURPLE, zorder=6, label="measured panels")
    ax.annotate(f"measured\n$\\varphi={phibar:.2f}$", xy=(xs[0], ys[0]),
                xytext=(-2, 16), textcoords="offset points", fontsize=7.8,
                color=C_PURPLE, ha="left", va="bottom", fontweight="bold",
                linespacing=1.2)
    # 线性纵轴上 φ=0.5/0.75/0.95 三条线在 N=9 处只差 0.75，标签必然叠在一起；
    # 取对数后 1.06 / 1.29 / 1.8 / 3.0 / 9 均匀分开，且 N_eff 本就跨一个数量级。
    ax.axhline(1.0, color="#4a4a4a", lw=1.0, zorder=3)
    ax.text(0.30, 1.0, "one physician", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=7.2, color="#4a4a4a")
    ax.set_yscale("log")
    ax.set_xlim(0.6, 13.2); ax.set_ylim(0.90, 11.5)
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_yticks([1, 2, 5, 10]); ax.set_yticklabels(["1", "2", "5", "10"])
    ax.set_xlabel("Panel size $N$"); ax.set_ylabel("$N_{\\mathrm{eff}}$")
    ax.set_title("(a)  Effective independent opinions", loc="left", fontsize=9.4, pad=5)
    clean(ax)

    # (b) φ 的全网格分布
    ax = axes[1]
    rows = [(AL[a], [r["phi0"] for r in ind if r["arch"] == a], ARCH_SOLID[a])
            for a in MAS_ORDER]
    post = [r["phi_last"] for r in ind if r.get("phi_last") is not None]
    if post:
        rows.append(("after discussion", post, C_ROSE))
    for j, (lab, vals, c) in enumerate(rows):
        if not vals:
            continue
        jit = np.random.RandomState(7 + j).normal(0, .085, len(vals))
        ax.plot(vals, np.full(len(vals), j) + jit, "o", ms=3.0, mfc=c, mec=c,
                alpha=.42, zorder=3)
        med = float(np.median(vals))
        ax.plot([med], [j], "|", ms=15, color=INK, mew=1.8, zorder=5)
        ax.text(med, j + .34, f"{med:.2f}", ha="center", va="bottom", fontsize=7.4,
                color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.0)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("Error correlation $\\varphi$ between panel members")
    ax.set_title("(b)  $\\varphi$ across all 120 configurations", loc="left",
                 fontsize=9.4, pad=5)
    clean(ax, grid_axis="x")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig8_slope.{e}", dpi=200, bbox_inches="tight")
    print("fig8_slope ok")


# ------------------------------------------------------------------ fig9
def reliability(cells):
    """(a) 校准残差 = 置信 − 实测准确率。零线永远在画布内，正值即过度自信。
    原版画的是可靠性图，但 x∈[0.65,0.95]、y∈[0.30,0.65]，对角线整条跑出画布。
    (b) 哑铃图：错答置信 vs 对答置信，单位统一为 %。"""
    rcparams()
    series = {}
    for arch, lab in [("cot", "Single doctor"), ("independent", "Independent"),
                      ("centralized", "Centralized"), ("discussion", "Decentralized")]:
        pts = []
        for (m, b, a, N), v in cells.items():
            if a != arch or (arch != "cot" and N < 3):
                continue
            for ep in v:
                rs = [r for r in (ep.get("rounds") or []) if len(r) >= (1 if arch == "cot" else 2)]
                if not rs:
                    continue
                cf = [o.get("confidence", 50) for o in rs[-1] if o.get("answer")]
                if cf:
                    pts.append((float(np.mean(cf)), ep["correct"]))
        if len(pts) > 200:
            series[lab] = pts
    if not series:
        print("reliability 数据不足"); return
    COL = {"Single doctor": "#8c8c8c", "Independent": C_ORANGE,
           "Centralized": C_ROSE, "Decentralized": C_PURPLE}
    MK = {"Single doctor": "o", "Independent": "o", "Centralized": "s", "Decentralized": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.45),
                             gridspec_kw={"width_ratios": [1.12, 1]})

    # (a) 校准残差
    ax = axes[0]
    edges = np.array([50, 72, 80, 85.5, 89, 92.5, 100.1])
    for lab, pts in series.items():
        c = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts]) * 100
        xs, gaps = [], []
        for i in range(len(edges) - 1):
            msk = (c >= edges[i]) & (c < edges[i + 1])
            if msk.sum() >= 40:
                xs.append(c[msk].mean()); gaps.append(c[msk].mean() - y[msk].mean())
        if xs:
            ax.plot(xs, gaps, ls="--", lw=1.4, color=COL[lab], marker=MK[lab], ms=4.6,
                    mfc=COL[lab], mec=COL[lab], zorder=4, label=lab)
    ax.axhline(0, color="#4a4a4a", lw=1.1, zorder=3)
    ax.text(0.015, 0, " perfect calibration", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7.4, color="#4a4a4a")
    clean(ax)
    ax.set_xlabel("Stated confidence (%)")
    ax.set_ylabel("Overconfidence (pp)\nstated $-$ actual", fontsize=9.0, linespacing=1.3)
    ax.set_title("(a)  Every architecture overstates itself", loc="left", fontsize=9.4, pad=5)
    panel_legend(ax, loc="lower right", fontsize=7.4)

    # (b) 哑铃图
    ax = axes[1]
    rows = []
    for lab, pts in series.items():
        c = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        rows.append((lab, float(c[y == 0].mean()), float(c[y == 1].mean())))
    rows.sort(key=lambda r: r[2] - r[1])
    for j, (lab, w, r_) in enumerate(rows):
        ax.plot([w, r_], [j, j], color=COL[lab], lw=2.4, alpha=.55, zorder=2,
                solid_capstyle="round")
        ax.plot([w], [j], "o", ms=6.2, mfc="white", mec=COL[lab], mew=1.7, zorder=4)
        ax.plot([r_], [j], "o", ms=6.2, mfc=COL[lab], mec=COL[lab], zorder=4)
        ax.annotate(f"{r_ - w:.1f}", xy=((w + r_) / 2, j), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7.8,
                    color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    lo = min(r[1] for r in rows) - .9; hi = max(r[2] for r in rows) + .9
    ax.set_xlim(lo, hi); ax.set_ylim(-.65, len(rows) + .55)
    ax.set_xlabel("Mean stated confidence (%)")
    ax.set_title("(b)  Separation between wrong and right", loc="left", fontsize=9.4, pad=5)
    clean(ax, grid_axis="x")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=5.6, mfc="white",
                              mec=INK, mew=1.6, label="on wrong answers"),
                       Line2D([], [], ls="none", marker="o", ms=5.6, mfc=INK,
                              mec=INK, label="on correct answers")],
              loc="upper center", ncol=2, fontsize=7.2, frameon=True,
              framealpha=0.92, edgecolor="#c4c4c4", fancybox=False,
              columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig9_reliability.{e}", dpi=200, bbox_inches="tight")
    print("fig9_reliability ok")


if __name__ == "__main__":
    recs, cells = collect()
    forest(recs); slope(); reliability(cells)
