"""A1 —— 对标 NMI Figure 4：Agent Heterogeneity Effects。
四种能力配置 x {Centralized, Decentralized}，在窗口内的 MedXpertQA 上。
NMI 的关键操纵是 orchestrator 与 sub-agent 的能力错配。"""
import sys, pathlib, json, argparse, collections, time
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import pmap, LEDGER, BudgetExceeded
from panels.architectures import run_episode

HI, MID, LO = "gpt-5-mini", "gpt-5-nano", "gpt-4.1-nano"

CONFIGS = [
    ("homog-high",  dict(model=HI, models=[HI] * 3, orch_model=HI)),
    ("homog-low",   dict(model=LO, models=[LO] * 3, orch_model=LO)),
    ("hi-orch/lo-sub", dict(model=LO, models=[LO] * 3, orch_model=HI)),
    ("lo-orch/hi-sub", dict(model=HI, models=[HI] * 3, orch_model=LO)),
    ("mixed-panel", dict(model=MID, models=[HI, MID, LO], orch_model=MID)),
]

ap = argparse.ArgumentParser()
ap.add_argument("--items", default="data/medxpertqa_250.jsonl")
ap.add_argument("--limit", type=int, default=250)
ap.add_argument("--out", default="results/H_heterogeneity.jsonl")
ap.add_argument("--workers", type=int, default=7)
a = ap.parse_args()

items = [json.loads(l) for l in open(ROOT / a.items)][: a.limit]
outp = ROOT / a.out
done = set()
if outp.exists():
    for l in outp.open():
        try:
            r = json.loads(l)
            if r.get("status") == "ok":
                done.add((r["qid"], r["hetero"], r["arch"]))
        except Exception:
            pass
fh = outp.open("a")
t0 = time.time()
for arch in ("centralized", "discussion"):
    for name, spec in CONFIGS:
        cfg = dict(spec, arch=arch, N=3, seed=1, temp=0.7, effort="low", use_router=True)

        def work(it, cfg=cfg, name=name, arch=arch):
            if (it["qid"], name, arch) in done:
                return None
            try:
                r = run_episode(it, cfg)
                r.update(status="ok", hetero=name, orch_model=cfg.get("orch_model"),
                         sub_models=cfg.get("models"))
                return r
            except Exception as e:
                return {"qid": it["qid"], "arch": arch, "hetero": name, "status": "error",
                        "error": f"{type(e).__name__}: {e}"[:200], "gold": it["answer"], "cost": {}}
        try:
            res = pmap(work, items, workers=a.workers)
        except BudgetExceeded as e:
            print("预算停止", e); break
        ok = [r for r in res if r and r.get("status") == "ok"]
        for r in res:
            if r:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        if ok:
            acc = sum(x["correct"] for x in ok) / len(ok) * 100
            usd = sum(x["cost"]["usd"] for x in ok) / len(ok) * 1000
            print(f"{arch:12s} {name:16s} acc={acc:5.1f}%  n={len(ok):4d}  ${usd:6.2f}/1k  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
fh.close()
print(LEDGER.report())
