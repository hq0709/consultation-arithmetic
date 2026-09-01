"""样本外检验：窗口是在 OpenAI 网格上定出来的，Gemini 是没参与定义的第二个厂商。

窗口的主张：单医生基线落在 25--50% 时协作有正增益，落在窗外则没有或为负。
两个 Gemini 模型正好分居窗内外：
    gemini-3.5-flash-lite  MedXpert 34.8 / MedAgents 31.2  -> 窗内
    gemini-3.7-flash       MedXpert 60.4 / MedAgents 56.4  -> 窗外 (50-70 档)
    两者 MedQA 84.8 / 95.2                                  -> 窗外 (>70 档)
因此这是一个**双向**预测，而不是只挑有利的一侧。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load, mcnemar

MAS = ("independent", "centralized", "discussion", "tiered")


def band(psa):
    if psa < 25:   return "<25"
    if psa < 50:   return "25-50 (in window)"
    if psa < 70:   return "50-70"
    return ">70"


def main():
    # 只纳入错误率 < 2% 的 (模型,benchmark) 组合。gemini-3.7-flash 的 MedQA 在
    # 每日配额耗尽时只跑出 434/6750，用它会得到 12 个 episode 的"准确率"。
    import json as _json
    good = []
    for f in sorted(glob.glob(str(ROOT / "results/GEM_*.jsonl"))):
        ok = er = 0
        for l in open(f):
            try:
                er += (_json.loads(l).get("status") == "error")
                ok += 1
            except Exception:
                pass
        if ok and er / ok < 0.02:
            good.append(f)
        else:
            print(f"  [排除] {pathlib.Path(f).name}  错误率 {er/max(ok,1)*100:.1f}%")
    rows = load(sorted(good))
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)

    recs = []
    for (m, b, a, N), v in cells.items():
        if a not in MAS:
            continue
        base = cells.get((m, b, "cot", 1))
        if not base:
            continue
        ids = {x["qid"] for x in v} & {x["qid"] for x in base}
        if len(ids) < 50:
            continue
        A = {x["qid"]: x["correct"] for x in v if x["qid"] in ids}
        B = {x["qid"]: x["correct"] for x in base if x["qid"] in ids}
        psa = sum(B.values()) / len(ids) * 100
        gain = float(np.mean([A[q] - B[q] for q in ids]) * 100)
        _, _, p = mcnemar(B, A)
        recs.append(dict(model=m, bench=b, arch=a, N=N, psa=psa, gain=gain, p=p,
                         band=band(psa)))
    if not recs:
        print("Gemini 网格数据不足"); return

    print("=" * 76)
    print("按 (模型, benchmark) 看：预测 vs 实测")
    print("=" * 76)
    print(f"{'model':22s}{'bench':16s}{'P_SA':>7s}{'band':>20s}{'best gain':>11s}")
    for (m, b) in sorted({(r["model"], r["bench"]) for r in recs}):
        s = [r for r in recs if r["model"] == m and r["bench"] == b]
        best = max(s, key=lambda r: r["gain"])
        print(f"  {m:20s}{b:16s}{best['psa']:6.1f}%{best['band']:>20s}"
              f"{best['gain']:+10.1f}pp")

    print("\n" + "=" * 76)
    print("按窗口档位汇总（这是预测本身）")
    print("=" * 76)
    print(f"{'band':22s}{'n cfg':>7s}{'mean gain':>11s}{'sig better':>12s}{'sig worse':>11s}")
    out = []
    for bd in ["<25", "25-50 (in window)", "50-70", ">70"]:
        s = [r for r in recs if r["band"] == bd]
        if not s:
            continue
        up = sum(1 for r in s if r["p"] < .05 and r["gain"] > 0)
        dn = sum(1 for r in s if r["p"] < .05 and r["gain"] < 0)
        g = float(np.mean([r["gain"] for r in s]))
        out.append(dict(band=bd, n=len(s), gain=g, sig_up=up, sig_dn=dn))
        print(f"  {bd:20s}{len(s):7d}{g:+10.2f}pp{up:12d}{dn:11d}")

    inw = [r for r in recs if r["band"].startswith("25-50")]
    outw = [r for r in recs if not r["band"].startswith("25-50")]
    print(f"\n  窗内 n={len(inw)}: 平均 {np.mean([r['gain'] for r in inw]):+.2f}pp, "
          f"显著↑ {sum(1 for r in inw if r['p']<.05 and r['gain']>0)}, "
          f"显著↓ {sum(1 for r in inw if r['p']<.05 and r['gain']<0)}")
    print(f"  窗外 n={len(outw)}: 平均 {np.mean([r['gain'] for r in outw]):+.2f}pp, "
          f"显著↑ {sum(1 for r in outw if r['p']<.05 and r['gain']>0)}, "
          f"显著↓ {sum(1 for r in outw if r['p']<.05 and r['gain']<0)}")
    (ROOT / "results/gemini_holdout.json").write_text(
        json.dumps(dict(bands=out, configs=recs), indent=1))
    print("\n写入 results/gemini_holdout.json")


if __name__ == "__main__":
    main()
