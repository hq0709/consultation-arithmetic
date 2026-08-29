"""Plan §3.5: re-plot every contrast per difficulty stratum."""
import sys, pathlib, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from experiments.analyze import load, wilson, mcnemar

tags = json.load(open(ROOT / "results/difficulty_medxpertqa_pilot200.json"))
rows = load([ROOT / "results" / f for f in sys.argv[1:]])
for r in rows:
    r["difficulty"] = tags.get(r["qid"], {}).get("difficulty", "?")
    r["pass_rate"] = tags.get(r["qid"], {}).get("pass_rate")

n_by = collections.Counter(tags[q]["difficulty"] for q in tags)
print("strata (3 mid-tier models x k=5 pass rate):", dict(n_by), "\n")

cells = collections.defaultdict(list)
for r in rows:
    cells[(r["arch"], r["N"], r["difficulty"])].append(r)

order = [("zeroshot", 1), ("cot", 1), ("independent", 1), ("independent", 5), ("independent", 9),
         ("discussion", 1), ("discussion", 5), ("discussion", 9),
         ("sc", 1), ("sc", 5), ("sc", 9), ("sc", 27)]
strata = ["easy", "medium", "hard"]
print(f"{'arch':13s} {'N':>3s} " + "".join(f"{s+' (n)':>20s}" for s in strata))
for a, N in order:
    line = f"{a:13s} {N:3d} "
    for s in strata:
        v = cells.get((a, N, s), [])
        if not v:
            line += f"{'-':>20s}"; continue
        p, lo, hi = wilson(sum(x["correct"] for x in v), len(v))
        line += f"{p*100:8.1f}% [{lo*100:4.1f},{hi*100:4.1f}]"[:20].rjust(20)
    print(line)

print("\nConsultation gain vs single call (independent N=1), per stratum:")
print(f"{'contrast':34s} " + "".join(f"{s:>16s}" for s in strata))
for a, N in [("independent", 5), ("independent", 9), ("discussion", 5), ("discussion", 9),
             ("sc", 9), ("sc", 27)]:
    line = f"{a} N={N} vs single".ljust(34)
    for s in strata:
        A = {x["qid"]: x["correct"] for x in cells.get((a, N, s), [])}
        B = {x["qid"]: x["correct"] for x in cells.get(("independent", 1, s), [])}
        ks = set(A) & set(B)
        if len(ks) < 5:
            line += f"{'-':>16s}"; continue
        d = (sum(A[q] for q in ks) - sum(B[q] for q in ks)) / len(ks) * 100
        w, l, p = mcnemar(B, A)
        line += f"{d:+8.1f}pp p={p:.2f}"[:16].rjust(16)
    print(line)
