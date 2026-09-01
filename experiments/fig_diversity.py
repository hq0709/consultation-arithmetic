"""Fig 11 —— 多样性阶梯：把"换厂商买不到独立性"画出来。

(a) 四档多样性来源的归一化相关与九人面板的有效意见数：阶梯在跨家族处就已经走平，
    跨厂商这一步没有继续往下走。
(b) 全部 84 个模型对的散点：同厂商与跨厂商两团完全重叠 —— 这是 (a) 的原始证据，
    也说明 (a) 的均值不是被少数点带出来的。
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.vizstyle import (rcparams, clean, panel_legend, INK, MUTED,
                                  C_ORANGE, C_ROSE, C_PURPLE, C_CYAN, GAIN_NEG)
FIG = ROOT / "paper/figures"
SAME, CROSS = "#8c8c8c", C_CYAN          # 只用两色：同厂商 / 跨厂商


def main():
    rcparams()
    L = json.loads((ROOT / "results/diversity_ladder.json").read_text())
    recs = json.loads((ROOT / "results/phi_decomposition.json").read_text())
    lad = L["ladder"]

    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.50),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # ---------- (a) 阶梯
    ax = axes[0]
    SHORT = ["one model,\nrole prompts", "one family,\ndiff. checkpoints",
             "one vendor,\ndiff. families", "different\nvendors"]
    COL = [MUTED, C_ORANGE, C_PURPLE, CROSS]
    ys = np.arange(len(lad))[::-1]
    for y, r, c in zip(ys, lad, COL):
        ax.barh([y], [r["neff9"]], height=.58, color=c, zorder=3)
        ax.text(r["neff9"] + .12, y, f'{r["neff9"]:.2f}', va="center", ha="left",
                fontsize=8.0, color=INK, fontweight="bold")
    ax.axvline(9, color=GAIN_NEG, ls="--", lw=1.1, zorder=4)
    ax.text(8.75, ys[0] - .05, "9 = fully\nindependent", ha="right", va="center",
            fontsize=7.2, color=GAIN_NEG, linespacing=1.2)
    # 最后一步没有前进
    ax.annotate("", xy=(lad[3]["neff9"], ys[3]), xytext=(lad[2]["neff9"], ys[2]),
                arrowprops=dict(arrowstyle="-", color=CROSS, lw=1.2, ls=":"))
    ax.text(lad[3]["neff9"] + .95, (ys[2] + ys[3]) / 2,
            f'{lad[3]["neff9"] - lad[2]["neff9"]:+.2f}', va="center", ha="left",
            fontsize=8.0, color=CROSS, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(SHORT, fontsize=7.4, linespacing=1.2)
    ax.set_xlim(0, 9.9); ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_ylim(-.62, len(lad) - .25)
    ax.set_xlabel("Effective independent opinions in a panel of nine")
    ax.set_title("(a)  The ladder stops before the vendor boundary",
                 loc="left", fontsize=9.0, pad=5)
    # 左边框保留：(b) 面板是四边框，两图并排必须一致；而且横条从 x=0 起长，
    # 左边那条线就是条形的基线，去掉会让整排条悬空。
    clean(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)

    # ---------- (b) 84 个模型对
    ax = axes[1]
    means = {}
    for lab, sel, c, mk in [("same vendor", [r for r in recs if r["same_ecosystem"]], SAME, "o"),
                            ("cross vendor", [r for r in recs if not r["same_ecosystem"]], CROSS, "^")]:
        ax.plot([r["acc_gap"] * 100 for r in sel], [r["phi_norm"] for r in sel],
                ls="none", marker=mk, ms=4.2, mfc=c, mec=c, alpha=.70,
                zorder=4, label=f"{lab}  ($n={len(sel)}$)")
        m = float(np.mean([r["phi_norm"] for r in sel]))
        ax.axhline(m, color=c, ls="--", lw=1.3, zorder=3)
        means[lab] = m
    a, b = means["same vendor"], means["cross vendor"]
    ax.text(0.975, 0.955, f"means {a:.3f} vs {b:.3f}\ndiffer by {abs(b - a):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
            color=INK, fontweight="bold", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c4c4c4", lw=0.7))
    ax.set_xlabel("Error-rate gap between the two models (pp)", fontsize=9.5)
    ax.set_ylabel("$\\varphi/\\varphi_{\\max}$", fontsize=10)
    ax.set_title("(b)  The two clouds overlap", loc="left", fontsize=9.0, pad=5)
    clean(ax)
    panel_legend(ax, loc="lower right", fontsize=7.4)

    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig11_diversity.{e}", dpi=200, bbox_inches="tight")
    print("fig11_diversity ok")


if __name__ == "__main__":
    main()
