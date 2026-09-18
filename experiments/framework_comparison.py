"""对 MedAgents / MDAgents 的官方实现测与主网格完全相同的三个量。

不比准确率高低 —— 比的是：它们的 agent 独立吗（phi / N_eff）、正确答案在不在
房间里（headroom）、它们的聚合捞回多少（kappa）。单医生基线取主网格里同一模型、
同一题集的 CoT 成绩，所以三者可比。
"""
import sys, pathlib, json, glob, collections, itertools
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.grid_files import load_main

BENCH = ("medxpertqa", "medagentsbench", "medqa")


def phi_of(eps):
    """首轮意见之间的平均两两错误相关。"""
    by = collections.defaultdict(dict)
    for e in eps:
        fr = (e.get("rounds") or [[]])[0]
        g = e.get("gold")
        for i, o in enumerate(fr):
            if o.get("answer") is not None:
                by[e["qid"]][i] = int(o["answer"] != g)
    npos = max((max(v) + 1 for v in by.values() if v), default=0)
    rs = []
    for a, b in itertools.combinations(range(npos), 2):
        x = [(v[a], v[b]) for v in by.values() if a in v and b in v]
        if len(x) < 50:
            continue
        xa = np.array([p[0] for p in x], float); xb = np.array([p[1] for p in x], float)
        if xa.std() == 0 or xb.std() == 0:
            continue
        rs.append(np.corrcoef(xa, xb)[0, 1])
    return (float(np.mean(rs)), len(rs)) if rs else (np.nan, 0)


def main():
    base = {}
    for r in load_main():
        if r["model"] == "gpt-4.1-nano" and r["arch"] == "cot" and r["N"] == 1:
            base.setdefault(r["bench"], {})[r["qid"]] = r["correct"]

    out = {}
    print(f"{'框架':<12}{'benchmark':<16}{'n':>5}{'单医生':>8}{'框架':>8}{'oracle':>8}"
          f"{'headroom':>10}{'kappa':>8}{'phi':>7}{'N_eff':>7}")
    for fw in ("medagents", "mdagents"):
        for b in BENCH:
            p = ROOT / f"results/FW_{fw}_{b}.jsonl"
            if not p.exists():
                continue
            eps = []
            for l in p.open():
                if not l.strip():
                    continue
                try: r = json.loads(l)
                except Exception: continue
                if r.get("status") == "ok":
                    eps.append(r)
            if len(eps) < 50:
                continue
            ids = {e["qid"] for e in eps} & set(base.get(b, {}))
            eps = [e for e in eps if e["qid"] in ids]
            if len(eps) < 50:
                continue
            psa = np.mean([base[b][q] for q in ids]) * 100
            acc = np.mean([e["correct"] for e in eps]) * 100
            fr = [(e.get("rounds") or [[]])[0] for e in eps]
            orc = np.mean([int(any(o.get("answer") == e.get("gold") for o in f))
                           for e, f in zip(eps, fr) if len(f) >= 2]) * 100
            hd = orc - psa
            kp = (acc - psa) / hd * 100 if hd > 0 else np.nan
            ph, npair = phi_of(eps)
            neff = 9 / (1 + 8 * ph) if not np.isnan(ph) else np.nan
            n_ag = int(np.mean([len(f) for f in fr]))
            neff_n = n_ag / (1 + (n_ag - 1) * ph) if not np.isnan(ph) else np.nan
            out[f"{fw}|{b}"] = dict(n=len(eps), psa=psa, acc=acc, oracle=orc, headroom=hd,
                                    kappa=kp, phi=ph, n_agents=n_ag, neff=neff_n)
            print(f"  {fw:<10}{b:<16}{len(eps):>5}{psa:>8.1f}{acc:>8.1f}{orc:>8.1f}"
                  f"{hd:>+10.1f}{kp:>+8.1f}{ph:>7.3f}{neff_n:>7.2f}")

    # MedAgents 的共识循环
    print("\n" + "=" * 70)
    print("MedAgents 的共识循环：它的终止条件是「没人反对」")
    print("=" * 70)
    for b in BENCH:
        p = ROOT / f"results/FW_medagents_{b}.jsonl"
        if not p.exists(): continue
        eps = [json.loads(l) for l in p.open() if l.strip()]
        eps = [e for e in eps if e.get("status") == "ok"]
        if len(eps) < 50: continue
        rounds = [e["trace"]["n_vote_rounds"] for e in eps]
        reached = [e["trace"]["reached_consensus"] for e in eps]
        acc_c = np.mean([e["correct"] for e in eps if e["trace"]["reached_consensus"]]) * 100
        one = np.mean([r == 1 for r in rounds]) * 100
        out[f"consensus|{b}"] = dict(n=len(eps), pct_reached=np.mean(reached)*100,
                                     pct_one_round=one, acc_given_consensus=acc_c)
        print(f"  {b:<16}达成一致 {np.mean(reached)*100:5.1f}% · 一轮就一致 {one:5.1f}% · "
              f"一致时准确率 {acc_c:5.1f}%")

    # MDAgents 的复杂度分诊
    print("\n" + "=" * 70)
    print("MDAgents 的复杂度分诊：它决定单人还是开会")
    print("=" * 70)
    for b in BENCH:
        p = ROOT / f"results/FW_mdagents_{b}.jsonl"
        if not p.exists(): continue
        eps = [json.loads(l) for l in p.open() if l.strip()]
        eps = [e for e in eps if e.get("status") == "ok"]
        if len(eps) < 50: continue
        c = collections.Counter(e["trace"]["level"] for e in eps)
        accs = {k: np.mean([e["correct"] for e in eps if e["trace"]["level"] == k])*100
                for k in c}
        out[f"triage|{b}"] = dict(n=len(eps), dist={k: v/len(eps)*100 for k,v in c.items()},
                                  acc_by_level=accs)
        print(f"  {b:<16}" + " · ".join(f"{k} {v/len(eps)*100:.0f}%(acc {accs[k]:.0f})"
                                        for k, v in c.most_common()))

    (ROOT / "results/framework_comparison.json").write_text(json.dumps(out, indent=1))
    print("\n写入 results/framework_comparison.json")


if __name__ == "__main__":
    main()
