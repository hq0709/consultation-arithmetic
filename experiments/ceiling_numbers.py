"""全文引用的「可用空间 / 捕获率」权威数字。单一口径，一次算清。

口径：对每个配置 (model, bench, arch, N)，预言机上限由**该配置自己**的首轮意见算出
      —— 只要任一成员答对即算对。这是任何看不到标签的聚合规则的上界。
      headroom = oracle - P_SA;   kappa = (acc - P_SA) / headroom
      随机挑一个成员的期望准确率 ~= P_SA，故 kappa 的零点是「随机」，100 是「预言机」。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load
from experiments.grid_files import main_grid

AL = {"independent": "Independent", "centralized": "Centralized",
      "discussion": "Decentralized", "tiered": "Hybrid", "sc": "Self-consistency"}
MAS = ("independent", "centralized", "discussion", "tiered")


def main():
    rows = load(main_grid())
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)

    recs = []
    for (m, b, a, N), v in cells.items():
        if a not in MAS + ("sc",) or N < 3:
            continue
        base = cells.get((m, b, "cot", 1))
        if not base:
            continue
        B = {x["qid"]: x["correct"] for x in base}
        fr = [(ep.get("rounds") or [[]])[0] for ep in v]
        oc = [int(any(o.get("answer") == ep.get("gold") for o in f))
              for ep, f in zip(v, fr) if len(f) >= 2]
        if len(oc) < 50:
            continue
        ids = {x["qid"] for x in v} & set(B)
        psa = float(np.mean([B[q] for q in ids]) * 100)
        orc = float(np.mean(oc) * 100)
        acc = float(np.mean([x["correct"] for x in v]) * 100)
        # 可用空间过小时 kappa 的分母不稳，但准确率与预言机本身是好的 ——
        # 过去这里整条丢弃，把准确率最高的 cell（可用空间最小）一并排除，
        # 使"单医生"均值被压低 2.2pp。现在只在 kappa 上打标记，不丢数据。
        thin = (orc - psa) < 1.0
        recs.append(dict(model=m, bench=b, arch=a, N=N, psa=psa, acc=acc, oracle=orc,
                         headroom=orc - psa, thin=thin,
                         kappa=(None if thin else (acc - psa) / (orc - psa) * 100)))

    out = {}
    print("=" * 78)
    print("A. N=9 的汇总（引言 / 摘要引用这一组）")
    print("=" * 78)
    n9 = [r for r in recs if r["N"] == 9 and r["arch"] == "discussion"]
    out["single"] = float(np.mean([r["psa"] for r in n9]))
    out["oracle_n9"] = float(np.mean([r["oracle"] for r in n9]))
    best = []
    for r in n9:
        if r["thin"]:
            continue
        cand = [x["acc"] for x in recs if (x["model"], x["bench"], x["N"]) ==
                (r["model"], r["bench"], 9) and x["arch"] in MAS]
        best.append((max(cand) - r["psa"]) / r["headroom"] * 100)
    out["best_acc_n9"] = float(np.mean([max(x["acc"] for x in recs
                                            if (x["model"], x["bench"], x["N"]) ==
                                            (r["model"], r["bench"], 9) and x["arch"] in MAS)
                                        for r in n9]))
    out["kappa_best_n9"] = float(np.mean(best))
    print(f"  单医生            {out['single']:.1f}%")
    print(f"  预言机 (面板 N=9)  {out['oracle_n9']:.1f}%      可用空间 +{out['oracle_n9']-out['single']:.1f}pp")
    print(f"  最好的多智能体架构  {out['best_acc_n9']:.1f}%      kappa = {out['kappa_best_n9']:.0f}%")

    print("\n" + "=" * 78)
    print("B. 各架构的捕获率 kappa（全部 N>=3 的配置）")
    print("=" * 78)
    out["kappa_by_arch"] = {}
    for a in ("sc",) + MAS:
        s = [r["kappa"] for r in recs if r["arch"] == a and r["kappa"] is not None]
        if not s:
            continue
        out["kappa_by_arch"][AL[a]] = float(np.mean(s))
        print(f"  {AL[a]:18s} kappa = {np.mean(s):+6.1f}%   n={len(s):3d}")
    ks = [v for v in out["kappa_by_arch"].values()]
    print(f"  -> 五种机制的范围: {min(ks):+.0f}% .. {max(ks):+.0f}%")

    print("\n" + "=" * 78)
    print("C. 按 P_SA 分箱的分解（窗口的机制解释）")
    print("=" * 78)
    mas_only = [r for r in recs if r["arch"] in MAS]
    out["bins"] = []
    print(f"{'P_SA 区间':14s}{'n':>5s}{'可用空间':>10s}{'捕获率':>9s}{'增益':>9s}")
    for lo, hi, lab in [(0, 25, "<25"), (25, 50, "25-50"), (50, 70, "50-70"), (70, 101, ">70")]:
        s = [r for r in mas_only if lo <= r["psa"] < hi]
        if not s:
            continue
        row = dict(band=lab, n=len(s),
                   headroom=float(np.mean([r["headroom"] for r in s])),
                   kappa=float(np.mean([r["kappa"] for r in s if r["kappa"] is not None])),
                   gain=float(np.mean([r["acc"] - r["psa"] for r in s])))
        out["bins"].append(row)
        print(f"  {lab:12s}{row['n']:5d}{row['headroom']:+9.1f}pp{row['kappa']:+8.1f}%"
              f"{row['gain']:+8.2f}pp")

    (ROOT / "results/ceiling_numbers.json").write_text(json.dumps(out, indent=1))
    print("\n写入 results/ceiling_numbers.json")


if __name__ == "__main__":
    main()
