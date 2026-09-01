"""Grid runner. One row per (item, arch, N, seed) episode -> results/<name>.jsonl"""
from __future__ import annotations
import argparse, json, pathlib, sys, time, collections, os, threading, hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.llm import LEDGER, global_spend_usd, BudgetExceeded, pmap  # noqa: E402
from panels.architectures import run_episode  # noqa: E402
from panels.roles import route  # noqa: E402

_lock = threading.Lock()

# 单个 cell 里"无法解析出答案"的比例超过此值即中止整个网格。
# 正常运行远低于 1%；坏配置会一次性冲到 50% 以上。
BAD_OUTPUT_ABORT = float(os.environ.get("BAD_OUTPUT_ABORT", "0.15"))


def load_items(path, limit=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arches", default="independent")
    ap.add_argument("--Ns", default="1,3,5")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--sc-ks", default="")
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--generic-roles", action="store_true")
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = load_items(ROOT / a.items if not os.path.isabs(a.items) else a.items, a.limit)
    Ns = [int(x) for x in a.Ns.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]
    arches = [x for x in a.arches.split(",") if x]
    sc_ks = [int(x) for x in a.sc_ks.split(",") if x]

    cells = []
    for arch in arches:
        for seed in seeds:
            if arch == "sc":
                for k in sc_ks:
                    cells.append({"arch": arch, "N": k, "k": k, "seed": seed})
            elif arch in ("zeroshot", "cot"):
                cells.append({"arch": arch, "N": 1, "seed": seed})
            elif arch == "debate":
                for N in Ns:
                    if N >= 2:          # R1 #3
                        cells.append({"arch": arch, "N": N, "seed": seed})
            else:
                for N in Ns:
                    cells.append({"arch": arch, "N": N, "seed": seed})
    for c in cells:
        c.update({"model": a.model, "temp": a.temp, "effort": a.effort,
                  "use_router": not a.no_router, "generic_roles": a.generic_roles})
        if a.theta is not None:
            c["theta"] = a.theta

    outp = ROOT / "results" / a.out
    outp.parent.mkdir(exist_ok=True)
    # R1 #5: the resume key must cover EVERY setting that changes the episode, or a rerun
    # with a different temp/effort/theta/roles silently appends incomparable rows.
    def cfg_hash(c):
        keys = ("arch", "N", "k", "model", "seed", "temp", "effort", "theta",
                "use_router", "generic_roles")
        return hashlib.sha256(json.dumps({k: c.get(k) for k in keys},
                                         sort_keys=True).encode()).hexdigest()[:12]

    done = set()
    if outp.exists():
        for l in outp.open():
            try:
                r = json.loads(l)
                # An errored episode is NOT done -- a resume must retry it, or a rate-limit
                # burst permanently deletes those items from the cell.
                if r.get("status") == "error" or "error" in r:
                    continue
                done.add((r["qid"], r.get("cfg_hash")))
            except Exception:
                pass
    print(f"[grid] {len(items)} items x {len(cells)} cells = {len(items)*len(cells)} episodes "
          f"({len(done)} already done) -> {outp.name}", flush=True)

    # Pre-route once per item (cheap, shared by every cell).
    if not a.no_router:
        print("[grid] routing specialties...", flush=True)
        pmap(lambda it: route(it, True), items, workers=8)

    # 输出文件独占锁：两个进程写同一个文件会产生重复 episode（并不损坏 JSON，
    # 但会让准确率按重复份数加权）。2026-08-31 因为杀进程的模式没匹配上，
    # 旧的并发设置与新的并发设置同时写了 2250 条重复。
    import fcntl
    lockp = outp.with_suffix(outp.suffix + ".lock")
    lockf = lockp.open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"!! 另一个进程正在写 {outp.name}；本次退出以免产生重复记录。", flush=True)
        sys.exit(2)
    lockf.write(str(os.getpid())); lockf.flush()

    fh = outp.open("a")
    t0 = time.time()
    stats = collections.defaultdict(lambda: [0, 0, 0.0, 0])   # n, correct, usd, samples

    def work(job):
        it, cfg = job
        ch = cfg_hash(cfg)
        if (it["qid"], ch) in done:
            return None
        try:
            r = run_episode(it, cfg)
            r["status"] = "ok"; r["cfg_hash"] = ch
            return r
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            # R1 #6: an infrastructure failure is NOT a wrong answer. Never score it as one.
            return {"qid": it["qid"], "arch": cfg["arch"], "N": cfg["N"], "seed": cfg["seed"],
                    "model": cfg["model"], "status": "error", "cfg_hash": ch,
                    "error": f"{type(e).__name__}: {e}"[:300],
                    "gold": it["answer"], "cost": {}}

    for ci, cfg in enumerate(cells):
        jobs = [(it, cfg) for it in items]
        try:
            res = pmap(work, jobs, workers=a.workers)
        except BudgetExceeded as e:
            print(f"\n!! BUDGET STOP: {e}", flush=True); break
        n_err = n_blank = n_unparsed = 0
        for r in res:
            if r is None:
                continue
            with _lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r.get("status") == "error":
                n_err += 1
                continue
            k = (r["arch"], r["N"], r["seed"])
            s = stats[k]
            s[0] += 1; s[1] += r.get("correct", 0)
            s[2] += r.get("cost", {}).get("usd", 0.0); s[3] += r.get("cost", {}).get("samples", 0)
            # 质量守卫：模型返回了内容却解析不出答案，或干脆返回空串。
            # 2026-08-31 的教训：effort=high 让推理吃光 token 预算，
            # 245/250 条是空输出，却一路跑完 6 个 cell 才被发现。
            ops = (r.get("rounds") or [[]])[0]
            if ops:
                first = ops[0]
                if not str(first.get("raw", "")).strip():
                    n_blank += 1
                elif not first.get("answer"):
                    n_unparsed += 1
        fh.flush()
        k = (cfg["arch"], cfg["N"], cfg["seed"])
        s = stats[k]
        if s[0]:
            nres = sum(1 for r in res if r is not None)
            bad = (n_blank + n_unparsed) / max(1, nres)
            qual = ""
            if n_blank or n_unparsed:
                qual = f"  blank={n_blank} unparsed={n_unparsed}"
            print(f"[{ci+1}/{len(cells)}] {cfg['arch']:12s} N={cfg['N']:<2d} seed={cfg['seed']} "
                  f"acc={s[1]/s[0]*100:5.1f}%  n={s[0]:4d}  ${s[2]:.4f}  samples/item={s[3]/s[0]:.1f}  "
                  f"err={n_err:<3d}{qual} [{time.time()-t0:.0f}s]", flush=True)
            if bad > BAD_OUTPUT_ABORT:
                print(f"\n!! QUALITY STOP: {bad*100:.0f}% of this cell produced no parsable "
                      f"answer (blank={n_blank}, unparsed={n_unparsed} of {nres}).\n"
                      f"   Most often the model's reasoning consumed the whole token budget; "
                      f"raise --max-tokens / REASONING_MIN_TOKENS and rerun.\n"
                      f"   Aborting so the rest of the grid is not wasted.", flush=True)
                break
    fh.close()
    print("\n" + LEDGER.report())
    print(f"actual spend this run: ${LEDGER.total_cost():.4f} | today total: ${global_spend_usd():.4f}")


if __name__ == "__main__":
    main()
