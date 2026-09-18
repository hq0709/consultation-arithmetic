"""在我们的题集上跑 MedAgents / MDAgents 的官方实现，并记录能测 phi / oracle / kappa 的字段。

输出格式与主网格的 episode 兼容（qid/gold/pred/correct/rounds），所以后续分析
可以直接复用 ceiling_numbers 那一套。
"""
import sys, pathlib, json, argparse, time, threading
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import pmap
from frameworks import medagents, mdagents

FW = {"medagents": medagents, "mdagents": mdagents}
_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True, choices=list(FW))
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(ROOT / a.items) if l.strip()][:a.limit]
    outp = ROOT / "results" / a.out
    done = set()
    if outp.exists():
        for l in outp.open():
            try: done.add(json.loads(l)["qid"])
            except Exception: pass
    todo = [it for it in items if it["qid"] not in done]
    print(f"[{a.framework}] {len(items)} 题，已完成 {len(done)}，待跑 {len(todo)}")
    mod = FW[a.framework]
    t0 = time.time()

    def one(it):
        meter = {"calls": 0}
        try:
            pred, tr = mod.run(it, a.model, meter)
            rec = {"qid": it["qid"], "gold": it.get("answer") or it.get("gold"), "pred": pred,
                   "correct": int(pred == (it.get("answer") or it.get("gold"))), "status": "ok",
                   "framework": a.framework, "model": a.model,
                   "bench": pathlib.Path(a.items).stem.rsplit("_", 1)[0],
                   "arch": a.framework, "N": tr.get("n_agents", len(tr.get("domains", []))),
                   "rounds": [tr.get("first_round", [])], "trace": tr,
                   "cost": {"calls": meter["calls"]}}
        except Exception as e:
            rec = {"qid": it["qid"], "status": "error", "error": f"{type(e).__name__}: {e}",
                   "framework": a.framework, "model": a.model}
        with _lock:
            with outp.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    recs = pmap(one, todo, workers=a.workers)
    ok = [r for r in recs if r.get("status") == "ok"]
    if ok:
        acc = sum(r["correct"] for r in ok) / len(ok) * 100
        cal = sum(r["cost"]["calls"] for r in ok) / len(ok)
        print(f"  {len(ok)}/{len(todo)} ok · 准确率 {acc:.1f}% · 单题 {cal:.1f} 次调用 "
              f"· {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
