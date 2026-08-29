"""The pilot's headline figure: the N-curve reverses sign across difficulty strata."""
import sys, pathlib, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.analyze import load, wilson

tags = json.load(open(ROOT / "results/difficulty_medxpertqa_pilot200.json"))
rows = load([ROOT / "results/pilot_mxq_T2.jsonl"])
for r in rows:
    r["difficulty"] = tags.get(r["qid"], {}).get("difficulty", "?")
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["arch"], r["N"], r["difficulty"])].append(r)

strata = [("easy", 13), ("medium", 62), ("hard", 125)]
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=False)
COL = {"independent": "#1b6ca8", "discussion": "#c1440e", "sc": "#888888"}
LBL = {"independent": "independent panel (vote)", "discussion": "panel discussion",
       "sc": "self-consistency (same model)"}
for ax, (s, n) in zip(axes, strata):
    for arch, Ns in (("independent", [1, 5, 9]), ("discussion", [1, 5, 9]), ("sc", [1, 5, 9])):
        xs, ys, lo, hi = [], [], [], []
        for N in Ns:
            v = cells.get((arch, N, s), [])
            if not v:
                continue
            p, l, h = wilson(sum(x["correct"] for x in v), len(v))
            xs.append(N); ys.append(p * 100); lo.append(l * 100); hi.append(h * 100)
        ls = "--" if arch == "sc" else "-"
        ax.plot(xs, ys, "o" + ls, color=COL[arch], lw=2, label=LBL[arch])
        ax.fill_between(xs, lo, hi, color=COL[arch], alpha=0.10)
    z = cells.get(("zeroshot", 1, s), [])
    if z:
        ax.axhline(sum(x["correct"] for x in z) / len(z) * 100, ls=":", c="#333", lw=1.3,
                   label="single call (zero-shot)")
    ax.set_title(f"{s}  (n={n})"); ax.set_xticks([1, 5, 9]); ax.grid(alpha=0.25)
    ax.set_xlabel("number of expert agents N")
axes[0].set_ylabel("accuracy (%)"); axes[0].legend(fontsize=7.5, loc="lower right")
fig.suptitle("Consultation helps on easy cases and HARMS on hard ones — "
             "MedXpertQA, gpt-5-nano, 200 items", y=1.03, fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / "results/fig_strata_pilot.png", dpi=160, bbox_inches="tight")
print("wrote results/fig_strata_pilot.png")
