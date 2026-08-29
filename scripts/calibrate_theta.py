"""Calibrate the tiered-referral confidence threshold theta on a DISJOINT dev slice.

Objective: maximise Youden's J = TPR(refer | generalist wrong) - FPR(refer | generalist right).
That is exactly 'referral appropriateness': escalate on the cases the generalist would miss.
The chosen theta is then FROZEN for all evaluation runs.
"""
import sys, pathlib, json, argparse, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import pmap, LEDGER
from panels.architectures import Meter, _ask
from panels.base import parse_opinion, user_prompt
from panels.roles import role_system

ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--effort", default=None)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default=None)
a = ap.parse_args()

items = [json.loads(l) for l in open(ROOT / a.items)]


def gen(it):
    m = Meter()
    txt = _ask(m, a.model, role_system("internal medicine", generic=True), user_prompt(it),
               a.seed * 100 + 90, 0.3, a.effort, tag="gen")
    o = parse_opinion(txt, list(it["options"]), "generalist")
    return {"qid": it["qid"], "conf": o.confidence, "correct": int(o.answer == it["answer"])}


rows = pmap(gen, items, workers=8)
acc = sum(r["correct"] for r in rows) / len(rows)
dist = collections.Counter(int(r["conf"] // 10) * 10 for r in rows)
print(f"model={a.model} effort={a.effort} n={len(rows)} generalist_acc={acc*100:.1f}%")
print("confidence histogram (bin -> count | acc in bin):")
for b in sorted(dist):
    inb = [r for r in rows if int(r["conf"] // 10) * 10 == b]
    print(f"  {b:3d}-{b+9:3d}: {dist[b]:4d}  acc={sum(x['correct'] for x in inb)/len(inb)*100:5.1f}%")

best = None
cands = sorted({r["conf"] for r in rows} | {0.0, 101.0})
print("\ntheta  refer%  TPR(refer|wrong)  FPR(refer|right)  Youden J")
for th in cands:
    ref = [r for r in rows if r["conf"] < th]
    wrong = [r for r in rows if not r["correct"]]; right = [r for r in rows if r["correct"]]
    tpr = sum(1 for r in wrong if r["conf"] < th) / len(wrong) if wrong else 0
    fpr = sum(1 for r in right if r["conf"] < th) / len(right) if right else 0
    j = tpr - fpr
    print(f"{th:5.1f}  {len(ref)/len(rows)*100:5.1f}%  {tpr:15.3f}  {fpr:15.3f}  {j:8.3f}")
    if best is None or j > best[1]:
        best = (th, j, len(ref) / len(rows), tpr, fpr)
print(f"\nCHOSEN theta = {best[0]:.1f}  (Youden J={best[1]:.3f}, referral rate={best[2]*100:.1f}%, "
      f"TPR={best[3]:.3f}, FPR={best[4]:.3f})")
out = a.out or f"results/theta_{a.model.replace('.','')}_{pathlib.Path(a.items).stem}.json"
(ROOT / out).write_text(json.dumps({"model": a.model, "effort": a.effort, "items": a.items,
                                    "theta": best[0], "youden_j": best[1],
                                    "referral_rate": best[2], "tpr": best[3], "fpr": best[4],
                                    "generalist_acc": acc, "rows": rows}, indent=1))
print("wrote", out)
print(LEDGER.report())
