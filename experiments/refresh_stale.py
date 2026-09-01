"""重建三份没有生产者的结果文件。

results/ 里曾经有五个 JSON 停在 8-29 的 180 配置网格上，生成它们的脚本在分析流水线
重写时被删掉，输出留在原地，论文继续引用其中的数字。这里在当前 420 配置网格上
一次算清，口径写在每一节的注释里。

  A. 置信度判别力      -> 安全节 (§4.7)
  B. 一致率与条件错误率 -> 安全节 (§4.7)，与 consensus.json 同定义、按架构拆开
  C. 每一分准确率的成本 -> 成本节 (§4.4)
"""
import sys, pathlib, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load
from experiments.grid_files import main_grid, load_main

MAS = ("independent", "centralized", "discussion", "tiered")
AL = {"cot": "cot", "independent": "Independent", "centralized": "Centralized",
      "discussion": "Decentralized", "tiered": "Hybrid", "sc": "Self-consistency"}


def auc(pos, neg):
    """Mann-Whitney U / (n1*n2) —— 与 ROC-AUC 相等，不需要 sklearn。"""
    if not pos or not neg:
        return float("nan")
    x = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    r = np.empty(len(x))
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    i = 0
    while i < len(xs):                      # 并列取平均秩
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = len(pos)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))


def panel_conf(ep):
    """面板对外给出的置信度 = 最后一轮所有意见的均值。单智能体就是它自己那一条。"""
    rds = ep.get("rounds") or []
    if not rds:
        return None
    c = [o["confidence"] for o in rds[-1] if o.get("confidence") is not None]
    return float(np.mean(c)) if c else None


def unanimous(ep):
    """systems.tex 的定义：最后一轮每个 panelist 给出**同一个有效选项**。
    含未解析答案的面板不算一致。少于两个 agent 不定义。"""
    # 取**最后一个有两条以上意见的轮次**：Centralized 的末轮是编排者的单条裁决，
    # 直接看 rounds[-1] 会把这个架构整个过滤掉。Hybrid 未触发转诊的 episode
    # 只有一条意见，本来就没有一致性可言，返回 None 排除。
    rds = [r for r in (ep.get("rounds") or []) if len(r) >= 2]
    if not rds:
        return None
    ans = [o.get("answer") for o in rds[-1]]
    return all(a is not None and a == ans[0] for a in ans)


def main():
    rows = load_main()
    out = {}

    # ---- A. 置信度判别力 -------------------------------------------------
    print("=" * 74)
    print("A. 自述置信度区分「本次答对/答错」的能力")
    print("=" * 74)
    out["confidence"] = {}
    print(f"{'架构':<18}{'答对时':>9}{'答错时':>9}{'差':>7}{'AUC':>8}{'n':>8}")
    for a in ("cot",) + MAS:
        r = [(panel_conf(e), e["correct"]) for e in rows if e["arch"] == a]
        r = [(c, k) for c, k in r if c is not None]
        if len(r) < 200:
            continue
        pos = [c for c, k in r if k]
        neg = [c for c, k in r if not k]
        d = dict(right=float(np.mean(pos)), wrong=float(np.mean(neg)),
                 auc=float(auc(pos, neg)), n=len(r))
        out["confidence"][AL[a]] = d
        print(f"  {AL[a]:<16}{d['right']:>9.1f}{d['wrong']:>9.1f}"
              f"{d['right']-d['wrong']:>7.1f}{d['auc']:>8.3f}{d['n']:>8d}")

    # ---- B. 一致率与条件错误率 -------------------------------------------
    # 只在真正召集了面板的 episode 上算（N>=3）：N=1 没有一致性可言，
    # 把它算进去会用单智能体把一致率稀释掉。
    print("\n" + "=" * 74)
    print("B. 一致率与「一致时仍然错」的概率（N>=3）")
    print("=" * 74)
    out["consensus"] = {}
    print(f"{'架构':<18}{'一致率':>9}{'P(错|一致)':>12}{'n':>9}")
    for a, nlo in [(x, n) for n in (3, 9) for x in MAS]:
        v = [(unanimous(e), e["correct"]) for e in rows
             if e["arch"] == a and (e["N"] >= 3 if nlo == 3 else e["N"] == 9)]
        v = [(u, k) for u, k in v if u is not None]
        if len(v) < 200:
            continue
        una = [k for u, k in v if u]
        d = dict(unanimity=len(una) / len(v) * 100,
                 p_wrong_given_unanimous=(1 - float(np.mean(una))) * 100,
                 n=len(v), n_unanimous=len(una))
        out["consensus"][f"{AL[a]}|N>=3" if nlo == 3 else f"{AL[a]}|N=9"] = d
        print(f"  {AL[a]+('' if nlo==3 else '  (N=9)'):<16}{d['unanimity']:>8.1f}%"
              f"{d['p_wrong_given_unanimous']:>11.1f}%{d['n']:>9d}")

    # ---- C. 每一分准确率的成本 -------------------------------------------
    # 与单医生同 cell 比：增益 (pp) 与每 1000 题的额外美元。
    print("\n" + "=" * 74)
    print("C. 成本归一化（每个架构在其最好 N 上，跨 cell 取中位数）")
    print("=" * 74)
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    acc = {k: float(np.mean([x["correct"] for x in v])) * 100 for k, v in cells.items()}
    usd = {k: float(np.mean([x["cost"]["usd"] for x in v])) for k, v in cells.items()}
    recs = []
    for (m, b, a, N), v in cells.items():
        if a not in MAS + ("sc",):
            continue
        base = (m, b, "cot", 1)
        if base not in acc:
            continue
        g = acc[(m, b, a, N)] - acc[base]
        dx = (usd[(m, b, a, N)] - usd[base]) * 1000
        recs.append(dict(model=m, bench=b, arch=a, N=N, gain=g, extra_usd_per_1k=dx,
                         pp_per_dollar=(g / dx if dx > 0 else None),
                         mult=usd[(m, b, a, N)] / usd[base] if usd[base] else None))
    out["cost"] = recs
    print(f"{'架构':<18}{'最好N':>6}{'增益':>9}{'每千题$':>11}{'倍数':>8}{'pp/$':>9}")
    for a in ("sc",) + MAS:
        s = [r for r in recs if r["arch"] == a]
        if not s:
            continue
        byN = collections.defaultdict(list)
        for r in s:
            byN[r["N"]].append(r["gain"])
        bn = max(byN, key=lambda n: np.mean(byN[n]))
        w = [r for r in s if r["N"] == bn]
        pp = [r["pp_per_dollar"] for r in w if r["pp_per_dollar"] is not None]
        mult = [r["mult"] for r in w if r["mult"] is not None]
        print(f"  {AL[a]:<16}{bn:>6}{np.mean([r['gain'] for r in w]):>+8.2f}pp"
              f"{np.median([r['extra_usd_per_1k'] for r in w]):>11.2f}"
              f"{(np.median(mult) if mult else float('nan')):>7.1f}x"
              f"{(np.median(pp) if pp else float('nan')):>9.1f}")

    # ---- D. tab:ppd —— 三个能力层 x 三个 benchmark，每格最好的多智能体配置 ----
    # sc 不算「架构」：它是预算对照。含进去 MedQA 的两格会换成 sc。
    print("\n" + "=" * 74)
    print("D. 每一分准确率的成本（tab:ppd 的来源）")
    print("=" * 74)
    TIER = {"gpt-4.1-nano": "T1", "gpt-5-nano": "T2", "gpt-5-mini": "T3"}
    BN = {"medxpertqa": "MedXpertQA", "medqa": "MedQA", "medagentsbench": "MedAgentsBench"}
    out["ppd"] = []
    print(f"{'Benchmark':<16}{'Tier':<5}{'best cfg':<20}{'gain':>7}{'$/1k':>9}{'xcost':>7}{'pp/$':>8}")
    for m, tier in TIER.items():
        for b, bl in BN.items():
            base = (m, b, "cot", 1)
            cand = [(k, acc[k]) for k in acc if k[:2] == (m, b) and k[2] in MAS]
            if base not in acc or not cand:
                continue
            k, a = max(cand, key=lambda x: x[1])
            g = a - acc[base]
            dx = (usd[k] - usd[base]) * 1000
            row = dict(bench=bl, tier=tier, arch=k[2], N=k[3], gain=g, extra_usd_per_1k=dx,
                       mult=usd[k] / usd[base], pp_per_dollar=(g / dx if dx > 0 else None))
            out["ppd"].append(row)
            pp = f"{row['pp_per_dollar']:.1f}" if row["pp_per_dollar"] else "---"
            print(f"  {bl:<14}{tier:<5}{k[2] + ' N=' + str(k[3]):<20}{g:>+6.1f}"
                  f"{dx:>9.2f}{row['mult']:>6.1f}x{pp:>8}")

    (ROOT / "results/stale_refresh.json").write_text(json.dumps(out, indent=1))
    print("\n写入 results/stale_refresh.json")


if __name__ == "__main__":
    main()
