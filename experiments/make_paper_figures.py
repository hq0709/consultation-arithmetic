"""论文全部图：Fig1 主曲线 · Fig2 性能-成本前沿 · Fig3 45%阈值 · Fig4 协调动力学 · Fig5 难度分层"""
from __future__ import annotations
import sys, pathlib, json, glob, collections, argparse, math
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.analyze import load, wilson
from mechanisms.nmi_metrics import config_metrics, fit_turn_powerlaw, fit_msg_density_log
from panels.architectures import NMI_CLASS

FIG = ROOT / "paper/figures"; FIG.mkdir(parents=True, exist_ok=True)
TIER = {"gpt-4.1-nano": "T1 gpt-4.1-nano", "gpt-5-nano": "T2 gpt-5-nano", "gpt-5-mini": "T3 gpt-5-mini"}
TORDER = ["gpt-4.1-nano", "gpt-5-nano", "gpt-5-mini"]
BENCH = {"medxpertqa": "MedXpertQA", "medqa": "MedQA", "medagentsbench": "MedAgentsBench-hard"}
ARCH = [("independent", "Independent", "#1b6ca8"), ("centralized", "Centralized", "#2a9d4a"),
        ("discussion", "Decentralized", "#c1440e"), ("tiered", "Hybrid", "#7b4397")]


def get(files):
    rows = load(files)
    c = collections.defaultdict(list)
    for r in rows:
        c[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    return rows, c


def fig1(cells):
    models = [m for m in TORDER if any(k[0] == m for k in cells)]
    benches = [b for b in BENCH if any(k[1] == b for k in cells)]
    if not models or not benches: return
    fig, axes = plt.subplots(len(benches), len(models),
                             figsize=(2.05 * len(models), 1.80 * len(benches)), squeeze=False)
    for bi, b in enumerate(benches):
        for mi, m in enumerate(models):
            ax = axes[bi][mi]
            for key, lab, col in ARCH:
                Ns = sorted(k[3] for k in cells if k[:3] == (m, b, key))
                if not Ns: continue
                xs, ys, lo, hi = [], [], [], []
                for N in Ns:
                    v = cells[(m, b, key, N)]
                    p, l, h = wilson(sum(x["correct"] for x in v), len(v))
                    xs.append(N); ys.append(p*100); lo.append(l*100); hi.append(h*100)
                ax.plot(xs, ys, "o-", color=col, lw=1.8, ms=4, label=lab)
                ax.fill_between(xs, lo, hi, color=col, alpha=0.10)
            sc = sorted([k[3] for k in cells if k[:3] == (m, b, "sc")])
            if sc:
                xs = [k for k in sc if k <= 9]
                ys = [sum(x["correct"] for x in cells[(m,b,"sc",k)])/len(cells[(m,b,"sc",k)])*100 for k in xs]
                ax.plot(xs, ys, "s--", color="#888", lw=1.4, ms=3.5, label="Self-consistency")
            z = cells.get((m, b, "cot", 1))
            if z:
                ax.axhline(sum(x["correct"] for x in z)/len(z)*100, ls=":", c="#333", lw=1.2,
                           label="Single (CoT)")
            if bi == 0: ax.set_title(TIER.get(m, m), fontsize=10)
            if mi == 0: ax.set_ylabel(f"{BENCH.get(b,b)}\naccuracy (%)", fontsize=9)
            if bi == len(benches)-1: ax.set_xlabel("panel size $N$", fontsize=9)
            ax.grid(alpha=0.25); ax.set_xticks([1,3,5,7,9]); ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=6.5, loc="best")
    fig.suptitle("Accuracy vs panel size, by architecture / capability tier / benchmark", y=1.00, fontsize=11)
    fig.tight_layout(); fig.savefig(FIG/"fig1_curves.pdf", bbox_inches="tight"); fig.savefig(FIG/"fig1_curves.png", dpi=160, bbox_inches="tight")
    print("fig1 ok")


def fig2(cells):
    fig, ax = plt.subplots(figsize=(3.15, 2.45))
    mk = {"gpt-4.1-nano": "o", "gpt-5-nano": "s", "gpt-5-mini": "^"}
    for key, lab, col in ARCH + [("cot", "Single (SAS)", "#333"), ("sc", "Self-consistency", "#888")]:
        for m in TORDER:
            pts = [(sum(x["cost"]["usd"] for x in v)/len(v)*1000,
                    sum(x["correct"] for x in v)/len(v)*100)
                   for k, v in cells.items() if k[0] == m and k[2] == key]
            if not pts: continue
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            ax.scatter(x, y, c=col, marker=mk.get(m, "o"), s=32, alpha=.75,
                       label=f"{lab}" if m == TORDER[0] else None, edgecolors="none")
    ax.set_xscale("log"); ax.set_xlabel("USD per 1000 questions (nominal)")
    ax.set_ylabel("accuracy (%)"); ax.grid(alpha=.25)
    ax.set_title("Performance–cost frontier\n(marker = capability tier, colour = architecture)", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(FIG/"fig2_cost.pdf", bbox_inches="tight"); fig.savefig(FIG/"fig2_cost.png", dpi=160, bbox_inches="tight")
    print("fig2 ok")


def fig3(cells):
    """NMI 的核心断言：P_SA > 45% 后协作增益转负。我们发现医学上还有一个下界。"""
    MAS = [a[0] for a in ARCH]
    pts = []
    for (m, b, arch, N), v in cells.items():
        if arch not in MAS: continue
        base = cells.get((m, b, "cot", 1))
        if not base: continue
        ids = {x["qid"] for x in v} & {x["qid"] for x in base}
        if len(ids) < 50: continue
        acc = sum(x["correct"] for x in v if x["qid"] in ids)/len(ids)
        psa = sum(x["correct"] for x in base if x["qid"] in ids)/len(ids)
        pts.append((psa*100, (acc-psa)*100, arch, m, N))
    if len(pts) < 5:
        print("fig3 数据不足"); return
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.40),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.axvspan(25, 50, color="#ffd88a", alpha=.45, zorder=0)
    ax.text(37.5, ax.get_ylim()[1] if False else 8.2, "collaboration window",
            ha="center", fontsize=8.5, color="#8a6d00")
    cmap = {a[0]: a[2] for a in ARCH}
    lab = {a[0]: a[1] for a in ARCH}
    mk = {"gpt-4.1-nano": "o", "gpt-5-nano": "s", "gpt-5-mini": "^"}
    for key in MAS:
        sub = [p for p in pts if p[2] == key]
        if not sub: continue
        ax.scatter([p[0] for p in sub], [p[1] for p in sub], c=cmap[key],
                   marker="o", s=30, alpha=.75, edgecolors="none", label=lab[key])
    ax.axvline(45, c="k", ls="--", lw=1.5, label="NMI ceiling 45%")
    ax.axhline(0, c="#666", lw=.9)
    ax.set_xlabel("single-doctor baseline $P_{SA}$ (%)")
    ax.set_ylabel("collaboration gain (pp)")
    ax.set_title("(a) gain vs single-agent baseline", fontsize=10)
    ax.grid(alpha=.22); ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    bins = [(0, 25, "<25"), (25, 35, "25–35"), (35, 50, "35–50"),
            (50, 70, "50–70"), (70, 101, ">70")]
    xs, ys, es, fr = [], [], [], []
    for lo, hi, l in bins:
        s_ = [p[1] for p in pts if lo <= p[0] < hi]
        if not s_: continue
        xs.append(l); ys.append(np.mean(s_))
        es.append(np.std(s_)/max(1, np.sqrt(len(s_))))
        fr.append(np.mean([v > 0 for v in s_])*100)
    cols = ["#c0c0c0" if not (25 <= b[0] < 50) else "#e08214" for b in bins if
            any(b[0] <= p[0] < b[1] for p in pts)]
    ax.bar(xs, ys, yerr=es, color=cols, capsize=3)
    for i, (y, f) in enumerate(zip(ys, fr)):
        ax.text(i, y + (0.25 if y >= 0 else -0.55), f"{f:.0f}% +ve", ha="center", fontsize=7.5)
    ax.axhline(0, c="#333", lw=.9)
    ax.set_xlabel("single-doctor baseline $P_{SA}$ (%)")
    ax.set_ylabel("mean collaboration gain (pp)")
    ax.set_title("(b) the window, binned", fontsize=10)
    ax.grid(alpha=.22, axis="y")
    fig.suptitle("Collaboration pays only inside a window of task difficulty", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG/"fig3_threshold.pdf", bbox_inches="tight")
    fig.savefig(FIG/"fig3_threshold.png", dpi=160, bbox_inches="tight")
    print("fig3 ok")


def fig4(cells):
    md = []
    for (m, b, arch, N), v in cells.items():
        if arch in ("cot", "zeroshot"): continue
        base = cells.get((m, b, "cot", 1))
        if not base: continue
        r = config_metrics(v, base)
        if r: r.update({"arch": arch, "cls": NMI_CLASS.get(arch, arch), "model": m}); md.append(r)
    if len(md) < 5: print("fig4 数据不足"); return
    fig, axes = plt.subplots(1, 3, figsize=(6.30, 2.00))
    ax = axes[0]
    for key, lab, col in ARCH:
        s = [(r["n_agents"], r["turns"]) for r in md if r["arch"] == key]
        if s: ax.scatter([x[0] for x in s], [x[1] for x in s], c=col, s=26, label=lab, alpha=.75)
    pl = fit_turn_powerlaw([(r["n_agents"], r["turns"]) for r in md])
    if pl:
        xs = np.linspace(1, 10, 60)
        ax.plot(xs, pl["a"]*(xs+0.5)**pl["exponent"], "k-", lw=1.5,
                label="$T=%.2f(n+0.5)^{%.2f}$, $R^2$=%.2f" % (pl["a"], pl["exponent"], pl["r2"]))
        ax.plot(xs, 2.72*(xs+0.5)**1.724, "k--", lw=1.2, alpha=.6, label="NMI: $2.72(n+0.5)^{1.72}$")
    ax.set_xlabel("agents $n_a$"); ax.set_ylabel("turns $T$"); ax.set_yscale("log")
    ax.set_title("(a) turn-count power law", fontsize=10); ax.grid(alpha=.25); ax.legend(fontsize=6.5)
    ax = axes[1]
    for key, lab, col in ARCH:
        s = [(r["msg_density"], r["accuracy"]*100) for r in md if r["arch"] == key and r["msg_density"] > 0]
        if s: ax.scatter([x[0] for x in s], [x[1] for x in s], c=col, s=26, label=lab, alpha=.75)
    ml = fit_msg_density_log([(r["msg_density"], r["accuracy"]) for r in md])
    if ml:
        xs = np.linspace(0.05, max(2, max(r["msg_density"] for r in md)), 60)
        ax.plot(xs, (ml["intercept"]+ml["slope"]*np.log(xs))*100, "k-", lw=1.5,
                label="$S$=%.2f%+.2f$\\ln c$, $R^2$=%.2f" % (ml["intercept"], ml["slope"], ml["r2"]))
    ax.axvline(0.39, c="crimson", ls="--", lw=1.2, label="NMI plateau $c^*{=}0.39$")
    ax.set_xlabel("message density $c$"); ax.set_ylabel("accuracy (%)")
    ax.set_title("(b) message density", fontsize=10); ax.grid(alpha=.25); ax.legend(fontsize=6.5)
    ax = axes[2]
    cls = sorted({r["cls"] for r in md})
    vals = [np.mean([r["error_amp"] for r in md if r["cls"] == c]) for c in cls]
    ab = [np.mean([r["absorb"] for r in md if r["cls"] == c])*100 for c in cls]
    xp = np.arange(len(cls))
    ax.bar(xp-0.2, vals, 0.4, label="$A_e$ error amplification", color="#c1440e")
    ax.bar(xp+0.2, ab, 0.4, label="error absorption (%)", color="#2a9d4a")
    ax.axhline(1.0, c="#666", lw=.8, ls=":")
    ax.set_xticks(xp); ax.set_xticklabels(cls, rotation=18, ha="right", fontsize=7.5)
    ax.set_title("(c) error dynamics by architecture", fontsize=10)
    ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG/"fig4_coordination.pdf", bbox_inches="tight"); fig.savefig(FIG/"fig4_coordination.png", dpi=160, bbox_inches="tight")
    print("fig4 ok")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--glob", default="G_*.jsonl")
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT/"results"/a.glob)))
    if not files: print("无结果文件"); return
    rows, cells = get(files)
    print(f"{len(rows)} episodes / {len(cells)} cells")
    fig1(cells); fig2(cells); fig3(cells); fig4(cells)


if __name__ == "__main__":
    main()
