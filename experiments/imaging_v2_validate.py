"""Summarise the protocol-v2 validation runs (results/IMG2V_*.jsonl, results/OWV_*.jsonl).

V1  parse rate and accuracy per model (single reader, real image) and Independent N=3 parse rate
V2  single-reader accuracy for res256 / res128 / inv / clahe vs real on the same 50 items
    pass rule: mean accuracy of inv and of clahe within 5 points of real (averaged over sets and models)
V3  memory (shared-image, shared-text): parse rate >= 95% with up to four images per call
V5  report pathway: report length (words), parse rate, accuracy
V6  open-weight models: accuracy on S1/S2 real (>= 55%), probability mass sanity
"""
import collections, glob, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
rows = []
for p in sorted(glob.glob(str(ROOT / "results/IMG2V_*.jsonl"))):
    for l in open(p):
        r = json.loads(l)
        if r.get("status") == "ok":
            r["_file"] = pathlib.Path(p).stem
            rows.append(r)

key = lambda r: (r["model"], r["bench"], r["arch"], r["N"], r.get("image_mode", "real"), r.get("memory", "-") if False else r["_file"].endswith("_mem") and "mem" or "-")
cells = collections.defaultdict(list)
for r in rows:
    tag = "report" if r["arch"] == "report" else ("mem" if r["_file"].endswith("_mem") else "")
    cells[(r["model"], r["bench"], r["arch"], r["N"], r.get("image_mode", "real"), tag, r["cfg_hash"])].append(r)


def parse_rate(v):
    ops = [o for r in v for o in (r.get("rounds") or [[]])[0]]
    return sum(1 for o in ops if o.get("answer")) / max(1, len(ops)) * 100


print("V1/V2/V3/V5 — model | bench | arch N | image | tag : n, parse %, accuracy %")
acc = {}
for k in sorted(cells):
    v = cells[k]
    a = sum(r["correct"] for r in v) / len(v) * 100
    acc[k[:6]] = a
    extra = ""
    if k[5] == "report":
        w = [len(t.split()) for r in v for t in r.get("reports", [])]
        extra = f"  report words median {sorted(w)[len(w)//2] if w else 0}, max {max(w) if w else 0}"
    print(f"  {k[0]:<26} {k[1]:<15} {k[2]:<11} {k[3]:<2} {k[4]:<7} {k[5]:<6} n={len(v):<3} parse {parse_rate(v):5.1f}  acc {a:5.1f}{extra}")

print("\nV2 — single-reader accuracy minus real, same items (pass: inv and clahe within 5 points on average)")
d = collections.defaultdict(list)
for (m, b, arch, n, mode, tag), a in acc.items():
    if arch == "cot" and tag == "" and mode != "real" and (m, b, "cot", 1, "real", "") in acc:
        d[mode].append(a - acc[(m, b, "cot", 1, "real", "")])
for mode, xs in sorted(d.items()):
    print(f"  {mode:<7} mean {sum(xs)/len(xs):+5.1f}  range [{min(xs):+.1f}, {max(xs):+.1f}]  over {len(xs)} model x set cells")

ow = collections.defaultdict(list)
for p in sorted(glob.glob(str(ROOT / "results/OWV_*.jsonl"))):
    for l in open(p):
        r = json.loads(l)
        ow[(r["model"], r["bench"], r["condition"])].append(r)
if ow:
    print("\nV6 — open-weight accuracy (pass: S1/S2 real >= 55%)")
    for k in sorted(ow):
        v = ow[k]
        print(f"  {k[0]:<22} {k[1]:<10} {k[2]:<6} n={len(v):<3} acc {sum(r['correct'] for r in v)/len(v)*100:5.1f}")
