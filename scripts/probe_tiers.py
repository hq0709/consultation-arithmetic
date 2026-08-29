"""Pick the capability ladder: solo CoT accuracy + cost per candidate OpenAI model."""
import sys, pathlib, json, time
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import LEDGER, pmap, global_spend_usd
from panels.architectures import run_episode

items = [json.loads(l) for l in open(ROOT / "data/medxpertqa_pilot200.jsonl")][:40]
items += [json.loads(l) for l in open(ROOT / "data/medqa_pilot200.jsonl")][:40]
CANDIDATES = [("gpt-4.1-nano", None), ("gpt-4o-mini", None), ("gpt-4.1-mini", None),
              ("gpt-5-nano", "low"), ("gpt-5-mini", "low"), ("gpt-5.4-nano", "low")]
print(f"{'model':14s} {'effort':7s} {'MedXpert':>9s} {'MedQA':>8s} {'$/1k items':>11s} {'s/item':>7s}")
rows = []
for m, eff in CANDIDATES:
    t0 = time.time()
    cfg = {"arch": "cot", "N": 1, "model": m, "seed": 1, "temp": 0.3, "effort": eff}
    res = pmap(lambda it: run_episode(it, cfg), items, workers=8)
    by = {}
    for r in res:
        by.setdefault(r["bench"], []).append(r)
    usd = sum(r["cost"]["usd"] for r in res) / len(res) * 1000
    a1 = sum(r["correct"] for r in by.get("medxpertqa", [])) / max(1, len(by.get("medxpertqa", [])))
    a2 = sum(r["correct"] for r in by.get("medqa", [])) / max(1, len(by.get("medqa", [])))
    print(f"{m:14s} {str(eff):7s} {a1*100:8.1f}% {a2*100:7.1f}% {usd:10.2f}$ {(time.time()-t0)/len(items):7.2f}")
    rows.append({"model": m, "effort": eff, "medxpertqa": a1, "medqa": a2, "usd_per_1k": usd})
(ROOT / "results/tier_probe.json").write_text(json.dumps(rows, indent=1))
print("\n" + LEDGER.report()); print(f"today: ${global_spend_usd():.4f}")
