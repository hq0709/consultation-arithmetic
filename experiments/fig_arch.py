"""Fig 0 —— 五架构示意图。窄面板里每个节点都挂标签必然糊成一团，
所以只标"被区分的那个角色"，专科医生的身份交给下方共享图例，临床解释交给 caption。"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from experiments.vizstyle import rcparams, INK, MUTED, C_CYAN
FIG = ROOT / "paper/figures"

LEAD = "#3d4f66"      # 主诊医师 / 全科医生（协调者）
SPEC = C_CYAN         # 专科医生
ARROW = "#2f2f2f"


def node(ax, x, y, color, r=0.17, label=None, fs=7.2, above=False):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", lw=1.2, zorder=4))
    if label:
        dy, va = ((r + .10), "bottom") if above else (-(r + .10), "top")
        ax.text(x, y + dy, label, ha="center", va=va, fontsize=fs, color=INK,
                fontweight="bold", zorder=5)


def arrow(ax, p, q, two=True, r=0.17):
    d = np.array(q) - np.array(p); L = np.hypot(*d)
    if L < 1e-9:
        return
    u = d / L
    ax.add_patch(FancyArrowPatch(np.array(p) + u * (r + .03), np.array(q) - u * (r + .03),
                                 arrowstyle="<|-|>" if two else "-|>", color=ARROW,
                                 lw=0.95, mutation_scale=7.5, zorder=3,
                                 shrinkA=0, shrinkB=0))


def main():
    rcparams()
    fig, axes = plt.subplots(1, 5, figsize=(6.30, 1.72))
    titles = ["Single-agent (SAS)", "Independent", "Centralized", "Decentralized", "Hybrid"]
    subs = [r"$\{a\}$", r"$C=\varnothing$", "star", "complete", "star + peer"]
    tri = [-.62, 0.0, .62]
    for i, ax in enumerate(axes):
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.02, 1.02); ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles[i], fontsize=8.8, pad=9)
        ax.text(0, -0.98, subs[i], ha="center", va="bottom", fontsize=7.8, color=MUTED)
        if i == 0:
            node(ax, 0, .12, LEAD, r=.22)
        elif i == 1:
            for x in tri:
                node(ax, x, .12, SPEC)
            ax.text(0, -.52, "no communication", ha="center", va="center",
                    fontsize=7.0, color=MUTED, style="italic")
        elif i == 2:
            node(ax, 0, .48, LEAD, label="Attending", above=True)
            for x in tri:
                node(ax, x, -.28, SPEC); arrow(ax, (0, .48), (x, -.28))
        elif i == 3:
            pts = [(-.62, -.28), (.62, -.28), (0, .55)]
            for x, y in pts:
                node(ax, x, y, SPEC)
            for a in range(3):
                for b_ in range(a + 1, 3):
                    arrow(ax, pts[a], pts[b_])
        else:
            node(ax, 0, .48, LEAD, label="Generalist", above=True)
            pts = [(-.62, -.28), (.62, -.28)]
            for x, y in pts:
                node(ax, x, y, SPEC); arrow(ax, (0, .48), (x, y))
            arrow(ax, pts[0], pts[1])
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=7, mfc=LEAD, mec="white",
                               label="Attending / generalist  (orchestrator)"),
                        Line2D([], [], ls="none", marker="o", ms=7, mfc=SPEC, mec="white",
                               label="Specialist"),
                        Line2D([], [], color=ARROW, lw=1.0, marker=None,
                               label="Bidirectional message channel")],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, columnspacing=1.5,
               handletextpad=0.5, frameon=True, framealpha=0.92, edgecolor="#c4c4c4",
               fancybox=False, fontsize=7.8)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig0_architectures.{e}", dpi=200, bbox_inches="tight")
    print("fig0_architectures ok")


if __name__ == "__main__":
    main()
