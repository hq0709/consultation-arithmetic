"""核心机制检验：协作的价值是否来自意见的独立性？

假说 H：多智能体会诊的收益来自 agent 之间**不相关的错误**。
  - 题太难：所有 agent 用同一套错误知识 -> 错误高度相关 -> 投票放大共同错误
  - 题太易：单 agent 已对 -> 无独立信息可加
  - 中间地带：agent 各持部分正确信息且错误不相关 -> 协作才可能有效
  - 讨论进一步摧毁独立性（一致率 66%->98%），把分布式信息压成单点 -> 一致但错

可检验的预测：
  P1  错误相关性 phi 随题目难度上升
  P2  协作增益随 phi 下降
  P3  讨论后的 phi 高于讨论前
  P4  phi 对增益的解释力强于 panel 规模
"""
import sys, pathlib, glob, json, collections, itertools, math
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import stats
from experiments.analyze import load
from experiments.grid_files import main_grid

MAS = ("independent", "centralized", "discussion", "tiered")


def phi(cellrows, round_idx=0):
    """agent 位次之间的错误相关性（phi 系数），在给定轮次上计算。"""
    slot = collections.defaultdict(dict)
    for ep in cellrows:
        rs = ep.get("rounds") or []
        if len(rs) <= round_idx:
            continue
        r = rs[round_idx]
        if len(r) < 2:
            continue
        for i, o in enumerate(r):
            slot[i][ep["qid"]] = int(o.get("answer") != ep.get("gold"))
    vals = []
    for i, j in itertools.combinations(sorted(slot), 2):
        qs = set(slot[i]) & set(slot[j])
        if len(qs) < 30:
            continue
        x = np.array([slot[i][q] for q in qs], float)
        y = np.array([slot[j][q] for q in qs], float)
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(vals)) if vals else None


def main():
    rows = load(main_grid())
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)

    recs = []
    for (m, b, a, N), v in cells.items():
        if a not in MAS or N < 3:
            continue
        base = cells.get((m, b, "cot", 1))
        if not base:
            continue
        ids = {x["qid"] for x in v} & {x["qid"] for x in base}
        if len(ids) < 50:
            continue
        p0 = phi(v, 0)
        if p0 is None:
            continue
        psa = sum(x["correct"] for x in base if x["qid"] in ids) / len(ids)
        acc = sum(x["correct"] for x in v if x["qid"] in ids) / len(ids)
        pl = phi(v, -1) if len((v[0].get("rounds") or [])) > 1 else None
        recs.append(dict(model=m, bench=b, arch=a, N=N, psa=psa, gain=acc - psa,
                         phi0=p0, phi_last=pl))

    print(f"{len(recs)} 个配置有可算的错误相关性\n")
    print("=" * 74)
    print("P1  错误相关性随题目难度上升？")
    print("=" * 74)
    print(f"{'P_SA 区间':>12s} {'n':>4s} {'平均 phi':>9s} {'平均增益':>10s}")
    for lo, hi, lab in [(0, .25, "<25%"), (.25, .35, "25–35%"), (.35, .50, "35–50%"),
                        (.50, .70, "50–70%"), (.70, 1.01, ">70%")]:
        s = [r for r in recs if lo <= r["psa"] < hi]
        if s:
            print(f"{lab:>12s} {len(s):4d} {np.mean([r['phi0'] for r in s]):9.3f} "
                  f"{np.mean([r['gain'] for r in s])*100:+9.2f}pp")
    r_, p_ = stats.pearsonr([r["psa"] for r in recs], [r["phi0"] for r in recs])
    print(f"\n  phi 与 P_SA 的相关: r = {r_:+.3f} (p={p_:.2e})")
    print("  -> 基线越低（题越难），错误相关性越" + ("高" if r_ < 0 else "低"))

    print("\n" + "=" * 74)
    print("P2  协作增益随错误相关性下降？")
    print("=" * 74)
    r2, p2 = stats.pearsonr([r["phi0"] for r in recs], [r["gain"] for r in recs])
    print(f"  phi 与增益的相关: r = {r2:+.3f} (p={p2:.2e}, n={len(recs)})")
    for lo, hi in [(-1, .1), (.1, .2), (.2, .3), (.3, 1)]:
        s = [r for r in recs if lo <= r["phi0"] < hi]
        if s:
            print(f"    phi ∈ [{lo:.1f},{hi:.1f})  n={len(s):3d}  "
                  f"平均增益 {np.mean([r['gain'] for r in s])*100:+6.2f}pp")

    print("\n" + "=" * 74)
    print("P3  讨论是否进一步摧毁独立性？")
    print("=" * 74)
    d = [r for r in recs if r["arch"] == "discussion" and r["phi_last"] is not None]
    if d:
        a0 = np.mean([r["phi0"] for r in d]); a1 = np.mean([r["phi_last"] for r in d])
        t, p = stats.ttest_rel([r["phi0"] for r in d], [r["phi_last"] for r in d])
        print(f"  Decentralized: 讨论前 phi = {a0:.3f} -> 讨论后 phi = {a1:.3f} "
              f"({a1-a0:+.3f}, 配对 t={t:.2f}, p={p:.2e}, n={len(d)})")

    print("\n" + "=" * 74)
    print("P4  phi 对增益的解释力 vs panel 规模")
    print("=" * 74)
    import statsmodels.api as sm
    import pandas as pd
    df = pd.DataFrame(recs)
    for cols, lab in [(["phi0"], "仅错误相关性 phi"), (["N"], "仅 panel 规模 n_a"),
                      (["psa"], "仅单医生基线 P_SA"), (["phi0", "psa"], "phi + P_SA"),
                      (["phi0", "psa", "N"], "phi + P_SA + n_a")]:
        X = sm.add_constant(df[cols].astype(float))
        m = sm.OLS(df["gain"].astype(float), X).fit()
        print(f"  {lab:22s} R² = {m.rsquared:.3f}")

    json.dump(recs, open(ROOT / "results/independence.json", "w"), indent=1)
    print("\n写入 results/independence.json")


if __name__ == "__main__":
    main()
