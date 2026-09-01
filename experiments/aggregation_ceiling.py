"""面板里到底有没有信息，以及为什么取不出来。

三个量，构成全文的因果链：
  1. **意见对的四分解**：两个专科医生 {一致,分歧} x {有人对,都错}。
     "分歧"看起来像会诊，但其中相当一部分是两人都错、只是错到不同选项 —— 这种分歧
     不携带任何可用于纠错的信息。
  2. **预言机上限**：若总能从首轮意见里挑出正确答案，面板能达到多少。
     这是任何无监督聚合规则的上界。
  3. **已捕获比例**：实测最佳架构拿到了这个上界的百分之多少。
     失败不在"面板里没信息"，而在"没有规则能认出哪个成员是对的" ——
     与 4.6 节测到的置信度无判别力（Youden J = 0.071）是同一件事。
"""
import sys, pathlib, glob, json, collections, itertools
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load
from experiments.grid_files import main_grid, load_main

MAS = ("independent", "centralized", "discussion", "tiered")
PAIR_ARCH = ("independent", "centralized", "discussion")


def pair_decomposition(cells):
    """两个 agent 位次的意见对，按 {一致,分歧} x {有人对,都错} 四分。"""
    cat = collections.defaultdict(collections.Counter)
    for (m, b, a, N), v in cells.items():
        if a not in PAIR_ARCH or N < 3:
            continue
        slot = collections.defaultdict(dict); gold = {}
        for ep in v:
            rs = ep.get("rounds") or []
            if not rs or len(rs[0]) < 2:
                continue
            gold[ep["qid"]] = ep.get("gold")
            for i, o in enumerate(rs[0]):
                if o.get("answer"):
                    slot[i][ep["qid"]] = o["answer"]
        for i, j in itertools.combinations(sorted(slot), 2):
            for q in set(slot[i]) & set(slot[j]):
                x, y, g = slot[i][q], slot[j][q], gold[q]
                ok_x, ok_y = x == g, y == g
                if x == y:
                    cat[b]["agree_right" if ok_x else "agree_wrong"] += 1
                else:
                    cat[b]["split_one_right" if (ok_x ^ ok_y) else "split_both_wrong"] += 1
    return cat


def oracle_ceiling(cells):
    """首轮意见里只要有一个对就算对 —— 任何无监督聚合规则的上界。"""
    out = []
    for (m, b, a, N), v in sorted(cells.items()):
        if a != "discussion" or N < 9:
            continue
        base = cells.get((m, b, "cot", 1))
        if not base:
            continue
        B = {x["qid"]: x["correct"] for x in base}
        orc = [int(any(o.get("answer") == ep.get("gold") for o in (ep.get("rounds") or [[]])[0]))
               for ep in v if ep.get("rounds")]
        if not orc:
            continue
        ids = {x["qid"] for x in v} & set(B)
        psa = float(np.mean([B[q] for q in ids]) * 100)
        oc = float(np.mean(orc) * 100)
        best = max(float(np.mean([x["correct"] for x in vv]) * 100)
                   for (m2, b2, a2, _), vv in cells.items()
                   if (m2, b2) == (m, b) and a2 in MAS)
        out.append(dict(model=m, bench=b, psa=psa, best=best, oracle=oc,
                        headroom=oc - psa, captured=(best - psa) / (oc - psa) * 100
                        if oc > psa else float("nan")))
    return out


def main():
    rows = load_main()
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)

    KEYS = ["agree_right", "agree_wrong", "split_one_right", "split_both_wrong"]
    ZH = {"agree_right": "一致·都对", "agree_wrong": "一致·都错",
          "split_one_right": "分歧·一人对", "split_both_wrong": "分歧·都错"}
    cat = pair_decomposition(cells)
    print("=" * 84)
    print("1. 两个专科医生的意见对，四分解")
    print("=" * 84)
    print(f"{'benchmark':20s}" + "".join(f"{ZH[k]:>14s}" for k in KEYS))
    tot = collections.Counter()
    for b in sorted(cat):
        n = sum(cat[b].values()); tot.update(cat[b])
        print(f"  {b:18s}" + "".join(f"{cat[b][k]/n*100:13.1f}%" for k in KEYS))
    n = sum(tot.values())
    print(f"  {'合计':18s}" + "".join(f"{tot[k]/n*100:13.1f}%" for k in KEYS))
    dis = tot["split_one_right"] + tot["split_both_wrong"]
    print(f"\n  所有分歧中 {tot['split_both_wrong']/dis*100:.1f}% 是两人都错（只是错到不同选项），"
          f"不携带任何纠错信息")
    print(f"  {tot['agree_wrong']/n*100:.1f}% 的病例上两人一致地错 —— 加多少成员都无法改变")

    oc = oracle_ceiling(cells)
    print("\n" + "=" * 84)
    print("2-3. 预言机上限与已捕获比例（N=9）")
    print("=" * 84)
    print(f"{'benchmark':18s}{'model':14s}{'单医生':>8s}{'实测最佳':>9s}{'预言机':>8s}"
          f"{'可用空间':>9s}{'已捕获':>8s}")
    for r in oc:
        print(f"  {r['bench']:16s}{r['model']:14s}{r['psa']:7.1f}%{r['best']:8.1f}%"
              f"{r['oracle']:7.1f}%{r['headroom']:+8.1f}pp{r['captured']:7.0f}%")
    A = [r for r in oc if r["captured"] == r["captured"]]
    print(f"\n  平均：单医生 {np.mean([r['psa'] for r in A]):.1f}%  →  "
          f"实测最佳 {np.mean([r['best'] for r in A]):.1f}%  →  "
          f"预言机 {np.mean([r['oracle'] for r in A]):.1f}%")
    print(f"  面板可用空间 {np.mean([r['headroom'] for r in A]):+.1f}pp，"
          f"实测仅捕获 {np.mean([r['captured'] for r in A]):.0f}%")
    print("\n  => 失败不在缺信息，而在无法辨认哪个成员是对的。")

    (ROOT / "results/aggregation_ceiling.json").write_text(json.dumps(
        dict(pair_decomposition={b: dict(cat[b]) for b in cat},
             pair_total=dict(tot), oracle=oc), indent=1))
    print("\n写入 results/aggregation_ceiling.json")


if __name__ == "__main__":
    main()
