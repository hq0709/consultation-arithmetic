"""协作窗口边界的参数化估计 + 自助置信区间。

原设计的问题：P_SA 是 (模型 x benchmark) 格子的属性，180 个配置里只有 9 个不同取值。
"180 个观测"实际是 9 个簇，边界只能被识别到观测值之间的空隙里。

修法：用**留一法难度**把每个格子的题目分层 —— 每题的难度取其余模型的平均正确率，
与目标模型完全独立，因此不存在「用结果挑样本」的选择偏差。分层后 P_SA 取值数
从 9 增到 ~27，再拟合分段模型并对断点做自助。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load

MAS = ("independent", "centralized", "discussion", "tiered")
N_STRATA = 3


def loo_difficulty(solo):
    """每题难度 = 其余模型的平均正确率（留一，故与目标模型独立）。"""
    per_item = collections.defaultdict(dict)          # (bench, qid) -> model -> correct
    for (m, b), d in solo.items():
        for q, c in d.items():
            per_item[(b, q)][m] = c
    return per_item


def main():
    files = sorted(glob.glob(str(ROOT / "results/G_*.jsonl"))) + \
            sorted(glob.glob(str(ROOT / "results/PHI_*.jsonl")))
    rows = load(files)
    solo = collections.defaultdict(dict)
    for r in rows:
        if r["arch"] == "cot" and r["N"] == 1:
            solo[(r["model"], r["bench"])][r["qid"]] = int(r["correct"])
    per_item = loo_difficulty(solo)
    n_models = len({m for m, _ in solo})
    print(f"用于留一难度的模型数: {n_models}")

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)

    pts = []
    for (m, b, a, N), v in cells.items():
        if a not in MAS:
            continue
        base = cells.get((m, b, "cot", 1))
        if not base:
            continue
        A = {x["qid"]: x["correct"] for x in v}
        B = {x["qid"]: x["correct"] for x in base}
        ids = [q for q in set(A) & set(B) if len(per_item[(b, q)]) >= 3]
        if len(ids) < 60:
            continue
        # 留一难度：排除目标模型自己
        diff = {q: np.mean([c for mm, c in per_item[(b, q)].items() if mm != m]) for q in ids}
        order = sorted(ids, key=lambda q: diff[q])
        k = len(order) // N_STRATA
        rs = np.random.RandomState(hash((m, b, a, N)) % (2 ** 31))
        for s in range(N_STRATA):
            chunk = order[s * k: (s + 1) * k] if s < N_STRATA - 1 else order[s * k:]
            if len(chunk) < 50:
                continue
            # **split-half**：P_SA 与 gain 若用同一批题，基线 B 同时出现在两边
            # （P_SA=B，gain=A-B），会制造回归到均值的假相关。各用一半。
            perm = rs.permutation(len(chunk))
            hA = [chunk[i] for i in perm[: len(chunk) // 2]]
            hB = [chunk[i] for i in perm[len(chunk) // 2:]]
            psa = np.mean([B[q] for q in hA]) * 100
            gain = np.mean([A[q] - B[q] for q in hB]) * 100
            # 地板/天花板区排除：P_SA 近 0 时 gain 不可能为负，近 100 时不可能为正
            if not (12.0 <= psa <= 88.0):
                continue
            pts.append(dict(model=m, bench=b, arch=a, N=N, stratum=s,
                            n=len(chunk), psa=float(psa), gain=float(gain)))
    if not pts:
        print("数据不足"); return

    P = np.array([p["psa"] for p in pts]); G = np.array([p["gain"] for p in pts])
    uniq = np.unique(np.round(P, 1))
    print(f"分层后配置数 = {len(pts)}，不同 P_SA 取值 = {len(uniq)} 个"
          f"（分层前为 9 个）")
    print(f"P_SA 覆盖 {P.min():.1f}% .. {P.max():.1f}%  (已排除 <12% / >88% 的地板天花板区)")
    gaps = np.diff(np.sort(uniq))
    print(f"相邻 P_SA 取值的最大空隙 = {gaps.max():.1f}pp（分层前为 21.6pp）\n")

    # ---- 分段模型：gain 在 [lo, hi] 内为正，外为负；网格搜索 + 自助
    def fit(Pv, Gv):
        best, bl, bh = -1e18, None, None
        for lo in np.arange(10, 55, 1.0):
            for hi in np.arange(lo + 8, 85, 1.0):
                inw = (Pv >= lo) & (Pv < hi)
                if inw.sum() < 8 or (~inw).sum() < 8:
                    continue
                # 组间均值差最大化（等价于二元分组的最小二乘）
                sep = Gv[inw].mean() - Gv[~inw].mean()
                if sep > best:
                    best, bl, bh = sep, lo, hi
        return bl, bh, best

    lo, hi, sep = fit(P, G)
    print(f"点估计: 窗口 = [{lo:.0f}%, {hi:.0f}%)   窗口内外平均增益差 = {sep:+.2f}pp")

    rng = np.random.RandomState(0)
    # 按 (模型,benchmark) 整簇自助 —— 同一格子内的分层不是独立观测
    clusters = collections.defaultdict(list)
    for i, p in enumerate(pts):
        clusters[(p["model"], p["bench"])].append(i)
    keys = list(clusters)
    los, his = [], []
    for _ in range(1000):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = [i for k in pick for i in clusters[keys[k]]]
        l2, h2, _ = fit(P[idx], G[idx])
        if l2 is not None:
            los.append(l2); his.append(h2)
    if los:
        print(f"自助 95% CI（按格子整簇重抽，B=1000）:")
        print(f"   下界 {np.percentile(los,2.5):.0f}% .. {np.percentile(los,97.5):.0f}%"
              f"   (中位 {np.median(los):.0f}%)")
        print(f"   上界 {np.percentile(his,2.5):.0f}% .. {np.percentile(his,97.5):.0f}%"
              f"   (中位 {np.median(his):.0f}%)")
    (ROOT / "results/window_boundary.json").write_text(json.dumps(
        dict(points=pts, lo=lo, hi=hi,
             lo_ci=[float(np.percentile(los, 2.5)), float(np.percentile(los, 97.5))] if los else None,
             hi_ci=[float(np.percentile(his, 2.5)), float(np.percentile(his, 97.5))] if los else None),
        indent=1))
    print("\n写入 results/window_boundary.json")


if __name__ == "__main__":
    main()
