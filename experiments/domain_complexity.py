"""B1 —— Appendix B：Domain Complexity D，严格照 NMI 的三分量定义。

D = mean( 1 - p_max ,  sigma/mu ,  1 - p_best )
  性能天花板：1 - 任一系统达到的最高性能
  变异系数  ：所有配置性能的 sigma/mu
  最佳模型基线：1 - 该数据集上最强单模型性能
NMI 报告临界阈值 D ~ 0.40：低于此多智能体净正回报，高于此协调开销吞噬推理资源。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load
from experiments.grid_files import main_grid

BENCH = {"medxpertqa": "MedXpertQA", "medqa": "MedQA (USMLE)",
         "medagentsbench": "MedAgentsBench-hard"}


def main():
    rows = load(main_grid())
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    out = {}
    print(f"{'benchmark':22s} {'1-p_max':>8s} {'sigma/mu':>9s} {'1-p_best':>9s} {'D':>7s} {'MAS 净增益':>11s}")
    for b in ("medxpertqa", "medagentsbench", "medqa"):
        accs = [sum(x["correct"] for x in v) / len(v)
                for k, v in cells.items() if k[1] == b and k[2] != "zeroshot"]
        if not accs:
            continue
        singles = [sum(x["correct"] for x in v) / len(v)
                   for k, v in cells.items() if k[1] == b and k[2] == "cot"]
        p_max = max(accs); p_best = max(singles) if singles else p_max
        ceiling = 1 - p_max
        cv = float(np.std(accs) / np.mean(accs))
        baseline = 1 - p_best
        D = float(np.mean([ceiling, cv, baseline]))
        # 该 benchmark 上 MAS 相对单医生的净增益
        gains = []
        for (m, bb, a, N), v in cells.items():
            if bb != b or a not in ("independent", "centralized", "discussion", "tiered"):
                continue
            base = cells.get((m, b, "cot", 1))
            if not base:
                continue
            ids = {x["qid"] for x in v} & {x["qid"] for x in base}
            if len(ids) < 50:
                continue
            gains.append(sum(x["correct"] for x in v if x["qid"] in ids) / len(ids)
                         - sum(x["correct"] for x in base if x["qid"] in ids) / len(ids))
        g = float(np.mean(gains)) * 100 if gains else float("nan")
        out[b] = dict(ceiling=ceiling, cv=cv, baseline=baseline, D=D, mas_gain_pp=g,
                      p_max=p_max, p_best=p_best, n_config=len(accs))
        print(f"{BENCH[b]:22s} {ceiling:8.3f} {cv:9.3f} {baseline:9.3f} {D:7.3f} {g:+10.2f}pp")
    print("\n  [NMI 临界阈值 D ~ 0.40：低于此 MAS 净正回报，高于此协调开销主导]")
    if len(out) >= 2:
        xs = [v["D"] for v in out.values()]; ys = [v["mas_gain_pp"] for v in out.values()]
        if len(xs) > 2:
            r = float(np.corrcoef(xs, ys)[0, 1])
            print(f"  D 与 MAS 净增益的相关: r = {r:+.3f}  (n={len(xs)} benchmarks)")
        # 我们的零交叉
        if len(xs) > 1:
            b1, b0 = np.polyfit(xs, ys, 1)
            if b1 != 0:
                print(f"  线性外推的零交叉 D* = {-b0/b1:.3f}   [NMI: 0.40]")
    (ROOT / "results/domain_complexity.json").write_text(json.dumps(out, indent=1))
    print("\n写入 results/domain_complexity.json")


if __name__ == "__main__":
    main()
