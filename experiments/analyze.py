"""Per-cell accuracy (Wilson 95% CI), cost, budget-matched comparisons, McNemar."""
from __future__ import annotations
import argparse, json, math, pathlib, sys, collections

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(a: dict, b: dict):
    """Exact-ish McNemar on the shared item set. a/b: qid -> correct(0/1)."""
    keys = set(a) & set(b)
    b01 = sum(1 for q in keys if a[q] == 0 and b[q] == 1)
    b10 = sum(1 for q in keys if a[q] == 1 and b[q] == 0)
    n = b01 + b10
    if n == 0:
        return 0, 0, 1.0
    # two-sided exact binomial
    p = 2 * sum(math.comb(n, i) for i in range(0, min(b01, b10) + 1)) / (2 ** n)
    return b10, b01, min(1.0, p)


def load(paths):
    rows = []
    for p in paths:
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            # R1 #6: infrastructure failures are excluded from accuracy, never scored 0.
            if r.get("status") == "error" or "error" in r:
                continue
            rows.append(r)
    return rows


def n_errors(paths):
    n = 0
    for p in paths:
        for l in open(p):
            try:
                if json.loads(l).get("status") == "error":
                    n += 1
            except Exception:
                pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--by-stratum", action="store_true")
    a = ap.parse_args()
    rows = load([ROOT / "results" / f if not pathlib.Path(f).exists() else f for f in a.files])
    if a.bench:
        rows = [r for r in rows if r.get("bench") == a.bench]
    print(f"{len(rows)} episodes\n")

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)

    hdr = (f"{'model':13s} {'bench':14s} {'arch':12s} {'N':>3s} {'n':>5s} {'acc':>6s} "
           f"{'95% CI':>15s} {'cw':>6s} {'abst':>5s} {'samp':>6s} {'$/item':>9s} {'acc/$':>8s}")
    print(hdr); print("-" * len(hdr))
    table = {}
    for k in sorted(cells):
        v = cells[k]
        n = len(v); c = sum(x["correct"] for x in v)
        p, lo, hi = wilson(c, n)
        cw = sum(x.get("correct_cw", 0) for x in v) / n
        ab = sum(x.get("abstain", 0) for x in v) / n
        usd = sum(x["cost"]["usd"] for x in v) / n
        samp = sum(x["cost"]["samples"] for x in v) / n
        perdollar = p / usd / 1000 if usd else float("nan")
        print(f"{k[0]:13s} {k[1]:14s} {k[2]:12s} {k[3]:3d} {n:5d} {p*100:5.1f}% "
              f"[{lo*100:4.1f},{hi*100:4.1f}] {cw*100:5.1f}% {ab*100:4.1f}% {samp:6.1f} "
              f"{usd:9.5f} {perdollar:8.2f}")
        table[k] = {q: r["correct"] for q, r in ((x["qid"], x) for x in v)}

    # R1 #7: the honest budget unit is nominal USD, not sample count. For each panel cell we
    # find the SC cell whose mean per-item USD is closest, and test against THAT.
    usd_of = {k: sum(x["cost"]["usd"] for x in v) / len(v) for k, v in cells.items()}
    print("\nPaired McNemar (Holm-corrected within family):")
    tests = []
    for (m, b, arch, N), t in table.items():
        if arch in ("zeroshot", "cot", "sc") or N == 1:
            continue
        base = table.get((m, b, arch, 1)) or table.get((m, b, "independent", 1))
        if base:
            w, l, p = mcnemar(base, t)
            tests.append((f"{m}/{b}/{arch} N={N} vs N=1", w, l, p))
        scs = [(k, usd_of[k]) for k in cells if k[0] == m and k[1] == b and k[2] == "sc"]
        if scs:
            target = usd_of[(m, b, arch, N)]
            kbest = min(scs, key=lambda kv: abs(kv[1] - target))[0]
            w, l, p = mcnemar(table[kbest], t)
            tests.append((f"{m}/{b}/{arch} N={N} vs $-matched SC k={kbest[3]} "
                          f"(${target:.5f} vs ${usd_of[kbest]:.5f})", w, l, p))
    tests.sort(key=lambda x: x[3])
    mtests = len(tests)
    for i, (name, w, l, p) in enumerate(tests):
        ph = min(1.0, p * (mtests - i))
        print(f"  {name:64s} b10={w:4d} b01={l:4d} p={p:.4g} p_holm={ph:.4g}"
              + ("  *" if ph < 0.05 else ""))
    print("\nNote (R1 #1): rows within an item share the routed roster and the round-0 prefix "
          "(common random numbers). McNemar on item-paired outcomes is valid; any GLMM must "
          "cluster at item and item:seed, not (1|item) alone. The N x arch x tier x difficulty "
          "model is fitted on the four PANEL architectures only (R1 #13); zeroshot/cot/sc are "
          "controls with a different meaning of N.")

    if a.by_stratum:
        print("\n(stratum breakdown requires results/difficulty_tags.json)")


if __name__ == "__main__":
    main()
