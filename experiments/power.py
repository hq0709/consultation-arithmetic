"""What the pilot is FOR: how many items does the full grid actually need?

McNemar power: with discordance rate d and split p10/(p10+p01), the required number of
item pairs for 80% power at alpha=0.05 (two-sided, normal approximation, Connor 1987):
    n = (z_a/2 * sqrt(d) + z_b * sqrt(d - delta^2))^2 / delta^2 ,  delta = p10 - p01
"""
import sys, pathlib, json, math, collections, itertools
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from experiments.analyze import load, wilson

Z_A, Z_B = 1.959964, 0.8416212

rows = load([ROOT / "results" / f for f in sys.argv[1:]])
by = collections.defaultdict(dict)
for r in rows:
    by[(r["arch"], r["N"])][r["qid"]] = r["correct"]

print("Observed paired contrasts and the n needed for 80% power at the OBSERVED effect:\n")
print(f"{'contrast':44s} {'acc A':>7s} {'acc B':>7s} {'diff':>7s} {'disc%':>6s} {'n@80%':>9s}")
pairs = [(("independent", 5), ("independent", 1)), (("independent", 9), ("independent", 1)),
         (("discussion", 5), ("discussion", 1)), (("discussion", 9), ("discussion", 1)),
         (("discussion", 5), ("independent", 5)), (("discussion", 9), ("independent", 9)),
         (("independent", 5), ("sc", 5)), (("independent", 9), ("sc", 9)),
         (("discussion", 5), ("sc", 9)), (("discussion", 9), ("sc", 15)),
         (("sc", 9), ("sc", 1)), (("independent", 1), ("zeroshot", 1))]
for a, b in pairs:
    A, B = by.get(a, {}), by.get(b, {})
    ks = set(A) & set(B)
    if len(ks) < 50:
        continue
    n = len(ks)
    p10 = sum(1 for q in ks if A[q] == 1 and B[q] == 0) / n
    p01 = sum(1 for q in ks if A[q] == 0 and B[q] == 1) / n
    d = p10 + p01
    delta = p10 - p01
    need = (f"{(Z_A*math.sqrt(d)+Z_B*math.sqrt(max(d-delta**2,1e-9)))**2/delta**2:,.0f}"
            if abs(delta) > 1e-9 else "inf")
    print(f"{a[0]+' N='+str(a[1]):>21s} vs {b[0]+' N='+str(b[1]):<20s} "
          f"{sum(A[q] for q in ks)/n*100:6.1f}% {sum(B[q] for q in ks)/n*100:6.1f}% "
          f"{delta*100:+6.1f}% {d*100:5.1f}% {need:>9s}")

print("\nMinimum detectable difference (MDD) at 80% power for the item counts we can afford,")
print("using the median observed discordance rate:")
ds = []
for a, b in pairs:
    A, B = by.get(a, {}), by.get(b, {})
    ks = set(A) & set(B)
    if len(ks) >= 50:
        ds.append(sum(1 for q in ks if A[q] != B[q]) / len(ks))
d = sorted(ds)[len(ds) // 2]
print(f"  median discordance d = {d*100:.1f}%")
for n in (200, 500, 1000, 2000, 4000):
    # solve delta from n = (z_a sqrt(d) + z_b sqrt(d-delta^2))^2/delta^2
    lo, hi = 1e-4, 0.5
    for _ in range(200):
        mid = (lo + hi) / 2
        need = (Z_A * math.sqrt(d) + Z_B * math.sqrt(max(d - mid ** 2, 1e-9))) ** 2 / mid ** 2
        if need > n:
            lo = mid
        else:
            hi = mid
    print(f"  n={n:5d} items -> MDD = {(lo+hi)/2*100:.1f} percentage points")
