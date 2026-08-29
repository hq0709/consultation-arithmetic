"""Figure 1: accuracy vs N per architecture, with Wilson CI bands + the economics panel."""
from __future__ import annotations
import json, sys, pathlib, collections, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.analyze import wilson, load

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", default="results/fig1_pilot.png")
ap.add_argument("--title", default="")
a = ap.parse_args()

rows = load([ROOT / "results" / f for f in a.files])
rows = [r for r in rows if "error" not in r]
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["arch"], r["N"])].append(r)

panels = ["independent", "discussion", "tiered", "debate"]
panels = [p for p in panels if any(k[0] == p for k in cells)]
fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 3.8), squeeze=False)
COL = {"independent": "#1b6ca8", "discussion": "#c1440e", "tiered": "#2a9d4a", "debate": "#7b4397"}

sc = {k[1]: v for k, v in cells.items() if k[0] == "sc"}
base = {b: cells.get((b, 1), []) for b in ("zeroshot", "cot")}

for ax, arch in zip(axes[0], panels):
    Ns = sorted(k[1] for k in cells if k[0] == arch)
    xs, ys, los, his = [], [], [], []
    for N in Ns:
        v = cells[(arch, N)]
        p, lo, hi = wilson(sum(x["correct"] for x in v), len(v))
        xs.append(N); ys.append(p * 100); los.append(lo * 100); his.append(hi * 100)
    ax.plot(xs, ys, "o-", color=COL.get(arch, "k"), lw=2, label=arch)
    ax.fill_between(xs, los, his, color=COL.get(arch, "k"), alpha=0.15)
    # budget-matched self-consistency at the same mean sample count
    sxs, sys_ = [], []
    for N in Ns:
        v = cells[(arch, N)]
        samp = sum(x["cost"]["samples"] for x in v) / len(v)
        best = min(sc, key=lambda k: abs(k - samp)) if sc else None
        if best is not None:
            w = sc[best]
            sxs.append(N); sys_.append(sum(x["correct"] for x in w) / len(w) * 100)
    if sxs:
        ax.plot(sxs, sys_, "s--", color="#888", lw=1.6, label="self-consistency\n(budget-matched)")
    for b, style in (("cot", ":"), ("zeroshot", "-.")):
        if base[b]:
            y = sum(x["correct"] for x in base[b]) / len(base[b]) * 100
            ax.axhline(y, ls=style, c="#444", lw=1.2, label=f"single {b}")
    ax.set_xlabel("number of expert agents N"); ax.set_title(arch)
    ax.set_xticks(Ns); ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="best")
axes[0][0].set_ylabel("accuracy (%)")

ax = axes[0][-1]
for arch in panels:
    Ns = sorted(k[1] for k in cells if k[0] == arch)
    x, y = [], []
    for N in Ns:
        v = cells[(arch, N)]
        usd = sum(i["cost"]["usd"] for i in v) / len(v)
        if usd:
            x.append(usd * 1000); y.append(sum(i["correct"] for i in v) / len(v) * 100)
    ax.plot(x, y, "o-", color=COL.get(arch, "k"), lw=2, label=arch)
if sc:
    x = [sum(i["cost"]["usd"] for i in sc[k]) / len(sc[k]) * 1000 for k in sorted(sc)]
    y = [sum(i["correct"] for i in sc[k]) / len(sc[k]) * 100 for k in sorted(sc)]
    ax.plot(x, y, "s--", c="#888", lw=1.6, label="self-consistency")
ax.set_xscale("log"); ax.set_xlabel("USD per 1000 questions"); ax.set_title("economics frontier")
ax.grid(alpha=0.25); ax.legend(fontsize=7)
fig.suptitle(a.title or "Consultation dose–response", y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / a.out, dpi=160, bbox_inches="tight")
print("wrote", a.out)
