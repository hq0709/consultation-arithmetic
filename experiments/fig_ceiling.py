"""Fig 10 —— 核心论点的视觉锚点。

(a) 面板的「可用空间」从哪来：单医生 -> 同一模型采样 9 次 -> 九位专科医生。
    92% 来自采样，8% 来自专科名册；实测最佳架构落在哪里。
(b) 「认得出谁是对的」的刻度：随机挑一个成员 = 0，预言机 = 100，五种机制全在个位数。
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.vizstyle import (rcparams, clean, ARCH_SOLID, INK, MUTED,
                                  C_ORANGE, C_ROSE, C_PURPLE, C_CYAN, GAIN_NEG)
FIG = ROOT / "paper/figures"


def main():
    rcparams()
    cn = json.loads((ROOT / "results/ceiling_numbers.json").read_text())
    hd = json.loads((ROOT / "results/headroom_decomp.json").read_text())
    sd, sco, pao = hd["single"], hd["sc_oracle"], hd["panel_oracle"]
    best = cn["best_acc_n9"]

    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.35),
                             gridspec_kw={"width_ratios": [1.18, 1]})

    # ---------- (a) 四根并排的横条：让"两根一样高"这件事直接可见
    ax = axes[0]
    LAB = ["one doctor,\nasked once",
           "one generalist,\nasked 9 times",
           "nine specialists,\nasked once each",
           "best architecture,\nwhat it delivers"]
    VAL = [sd, sco, pao, best]
    COL = ["#c9c9c9", C_CYAN, C_ROSE, INK]
    ys = np.arange(4)[::-1]
    ax.barh(ys, VAL, height=.60, color=COL, zorder=3)
    for y, v in zip(ys, VAL):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", ha="left", fontsize=8.2,
                color=INK, fontweight="bold")
    # 两个预言机之间的等价：用一个括号标出来
    ax.annotate("", xy=(pao + 4.0, ys[1]), xytext=(pao + 4.0, ys[2]),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
    for yy in (ys[1], ys[2]):
        ax.plot([pao + 3.4, pao + 4.0], [yy, yy], color=MUTED, lw=1.0, zorder=4)
    ax.annotate(f"the specialty roster\nis worth {pao-sco:+.1f} pp",
                xy=(pao + 4.0, (ys[1] + ys[2]) / 2), xytext=(5, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=7.8, color=C_ROSE, fontweight="bold", linespacing=1.25)
    ax.set_yticks(ys); ax.set_yticklabels(LAB, fontsize=7.6, linespacing=1.25)
    ax.set_xlim(40, pao + 15.0)
    ax.set_xticks([40, 50, 60])
    ax.set_ylim(-0.65, 3.65)
    ax.set_xlabel("Accuracy: is a correct answer available? (%)")
    ax.set_title("(a)  Nine specialists $\\approx$ one generalist asked nine times",
                 loc="left", fontsize=9.0, pad=5)
    # 左边框保留：(b) 面板是四边框，两图并排必须一致（图 6 同因修过）
    clean(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)

    # ---------- (b) 捕获率刻度
    ax = axes[1]
    K = cn["kappa_by_arch"]
    order = ["Independent", "Self-consistency", "Centralized", "Decentralized", "Hybrid"]
    COL = {"Independent": C_ORANGE, "Centralized": C_ROSE, "Decentralized": C_PURPLE,
           "Hybrid": C_CYAN, "Self-consistency": "#8c8c8c"}
    MK = {"Independent": "o", "Centralized": "s", "Decentralized": "^",
          "Hybrid": "D", "Self-consistency": "o"}
    for j, a in enumerate(order):
        v = K[a]
        ax.plot([0, v], [j, j], color=COL[a], lw=1.6, alpha=.55, zorder=3,
                solid_capstyle="round")
        ax.plot([v], [j], marker=MK[a], ms=6.0, mfc=COL[a], mec=COL[a], zorder=5)
        ax.annotate(f"{v:.1f}", xy=(v, j), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.8, color=INK, fontweight="bold")
    ax.axvline(0, color="#4a4a4a", lw=1.0, zorder=4)
    ax.text(0, len(order) - .38, " random member", ha="left", va="bottom",
            fontsize=7.4, color="#4a4a4a")
    ax.axvline(100, color=GAIN_NEG, ls="--", lw=1.1, zorder=4)
    ax.text(100, len(order) - .38, "oracle ", ha="right", va="bottom",
            fontsize=7.4, color=GAIN_NEG)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.0)
    ax.set_ylim(-.6, len(order) - .05)
    ax.set_xlim(-6, 106)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Headroom recovered $\\kappa$ (%)")
    ax.set_title("(b)  How much of it any rule recovers", loc="left", fontsize=9.4, pad=5)
    clean(ax, grid_axis="x")

    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig10_ceiling.{e}", dpi=200, bbox_inches="tight")
    print("fig10_ceiling ok")


if __name__ == "__main__":
    main()
