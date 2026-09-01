"""医疗领域特有的分析 —— NMI 的通用域设计拿不到这些。

M1 临床任务类型分解：Diagnosis / Treatment / Basic Science（MedXpertQA 自带标签）
M2 危险的一致性：panel 全体一致却错误的比率，以及协作对错误答案置信度的影响
M3 难度分层 x 架构（完整数据）
M4 身体系统分解
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import stats
from experiments.analyze import load, wilson, mcnemar
from experiments.grid_files import main_grid

MAS = ("independent", "centralized", "discussion", "tiered")
ANAME = {"independent": "Independent", "centralized": "Centralized",
         "discussion": "Decentralized", "tiered": "Hybrid", "cot": "SAS"}


def load_items():
    items = {}
    for f in glob.glob(str(ROOT / "data/*_250.jsonl")) + glob.glob(str(ROOT / "data/*_500.jsonl")):
        for l in open(f):
            r = json.loads(l); items[r["qid"]] = r
    return items


def main():
    items = load_items()
    rows = load(main_grid())
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)

    # ---------------- M1 临床任务类型 ----------------
    print("=" * 78)
    print("M1  协作收益按临床任务类型分解（MedXpertQA，自带 medical_task 标签）")
    print("=" * 78)
    tasks = ("Diagnosis", "Treatment", "Basic Science")
    print(f"{'model':13s} {'task':15s} {'n':>5s} {'单医生':>8s} {'最佳 MAS':>9s} {'增益':>9s} {'p':>8s}")
    for m in ("gpt-5-nano", "gpt-5-mini"):
        base_all = cells.get((m, "medxpertqa", "cot", 1), [])
        if not base_all:
            continue
        for tk in tasks:
            qs = {q for q, it in items.items()
                  if it.get("meta", {}).get("task") == tk and it["bench"] == "medxpertqa"}
            base = {x["qid"]: x["correct"] for x in base_all if x["qid"] in qs}
            if len(base) < 25:
                continue
            best, bg, bp = None, -9, 1
            for a in MAS:
                for N in sorted(k[3] for k in cells if k[:3] == (m, "medxpertqa", a)):
                    v = {x["qid"]: x["correct"] for x in cells[(m, "medxpertqa", a, N)]
                         if x["qid"] in base}
                    if len(v) < 25:
                        continue
                    g = sum(v.values()) / len(v) - sum(base[q] for q in v) / len(v)
                    if g > bg:
                        bg, best = g, f"{ANAME[a]}/{N}"
                        _, _, bp = mcnemar({q: base[q] for q in v}, v)
            if best:
                st = "***" if bp < .001 else ("**" if bp < .01 else ("*" if bp < .05 else ""))
                print(f"{m:13s} {tk:15s} {len(base):5d} {sum(base.values())/len(base)*100:7.1f}% "
                      f"{sum(base.values())/len(base)*100+bg*100:8.1f}% {bg*100:+8.2f}pp {bp:8.4f}{st}")

    # ---------------- M2 危险的一致性 ----------------
    print("\n" + "=" * 78)
    print("M2  临床安全：panel 全体一致却错误 —— 临床上最危险的失败模式")
    print("=" * 78)
    print(f"{'架构':>14s} {'一致率':>8s} {'一致且错':>9s} {'P(错|一致)':>11s} {'错答均置信':>11s} {'对答均置信':>11s}")
    for a in ("independent", "centralized", "discussion", "tiered"):
        una = wrong_una = 0; tot = 0
        cw, cr = [], []
        for (m, b, aa, N), v in cells.items():
            if aa != a or N < 3:
                continue
            for ep in v:
                # 一致性必须在「最后一个含 >=2 个 agent 的轮次」上定义。用末轮会出错：
                # Centralized 的末轮只有 orchestrator 一条意见，必然「一致」，
                # 会把一致率虚报为 100%。
                rs = [r for r in (ep.get("rounds") or []) if len(r) >= 2]
                if not rs:
                    continue
                r0 = rs[-1]
                ans = [o.get("answer") for o in r0 if o.get("answer")]
                if not ans:
                    continue
                tot += 1
                confs = [o.get("confidence", 50) for o in r0 if o.get("answer")]
                (cr if ep["correct"] else cw).append(np.mean(confs) if confs else 50)
                if len(set(ans)) == 1 and len(ans) == len(r0):
                    una += 1
                    if ans[0] != ep.get("gold"):
                        wrong_una += 1
        if tot:
            print(f"{ANAME[a]:>14s} {una/tot*100:7.1f}% {wrong_una/tot*100:8.1f}% "
                  f"{wrong_una/una*100 if una else 0:10.1f}% "
                  f"{np.mean(cw) if cw else 0:10.1f} {np.mean(cr) if cr else 0:10.1f}")
    # 单医生对照
    sw, sr = [], []
    for (m, b, aa, N), v in cells.items():
        if aa != "cot":
            continue
        for ep in v:
            o = (ep.get("rounds") or [[{}]])[0][0]
            (sr if ep["correct"] else sw).append(o.get("confidence", 50))
    print(f"{'SAS (单医生)':>14s} {'—':>7s} {'—':>8s} {'—':>10s} "
          f"{np.mean(sw) if sw else 0:10.1f} {np.mean(sr) if sr else 0:10.1f}")
    print("\n  解读：P(错|一致) 是「会诊给出统一意见但集体误诊」的条件概率。")

    # ---------------- M3 难度分层 x 架构 ----------------
    tagf = ROOT / "results/difficulty_medxpertqa_pilot200.json"
    if tagf.exists():
        print("\n" + "=" * 78)
        print("M3  协作收益 x 题目难度（完整数据；难度由 3 个中档模型 k=5 通过率定义）")
        print("=" * 78)
        tags = json.loads(tagf.read_text())
        print(f"{'model':13s} {'难度':>7s} {'n':>5s} {'单医生':>8s} {'平均增益':>10s} {'最佳增益':>10s}")
        for m in ("gpt-5-nano", "gpt-5-mini"):
            base_all = cells.get((m, "medxpertqa", "cot", 1), [])
            for dif in ("easy", "medium", "hard"):
                qs = {q for q, t in tags.items() if t["difficulty"] == dif}
                base = {x["qid"]: x["correct"] for x in base_all if x["qid"] in qs}
                if len(base) < 10:
                    continue
                gs = []
                for a in MAS:
                    for N in sorted(k[3] for k in cells if k[:3] == (m, "medxpertqa", a)):
                        v = {x["qid"]: x["correct"] for x in cells[(m, "medxpertqa", a, N)]
                             if x["qid"] in base}
                        if len(v) < 10:
                            continue
                        gs.append(sum(v.values()) / len(v) - sum(base[q] for q in v) / len(v))
                if gs:
                    p0 = sum(base.values()) / len(base) * 100
                    # 同时报平均与最佳：只报最佳等于在每格挑赢家，是选择偏倚
                    print(f"{m:13s} {dif:>7s} {len(base):5d} {p0:7.1f}% "
                          f"{np.mean(gs)*100:+9.2f}pp {max(gs)*100:+9.2f}pp")

    # ---------------- M4 身体系统 ----------------
    print("\n" + "=" * 78)
    print("M4  协作收益按身体系统（MedXpertQA, gpt-5-mini，最佳 MAS 对单医生）")
    print("=" * 78)
    m = "gpt-5-mini"
    base_all = cells.get((m, "medxpertqa", "cot", 1), [])
    res = []
    for sysname in sorted({it.get("meta", {}).get("body_system") for it in items.values()
                           if it["bench"] == "medxpertqa" and it.get("meta", {}).get("body_system")}):
        qs = {q for q, it in items.items()
              if it["bench"] == "medxpertqa" and it.get("meta", {}).get("body_system") == sysname}
        base = {x["qid"]: x["correct"] for x in base_all if x["qid"] in qs}
        if len(base) < 12:
            continue
        gs = []
        for a in MAS:
            for N in sorted(k[3] for k in cells if k[:3] == (m, "medxpertqa", a)):
                v = {x["qid"]: x["correct"] for x in cells[(m, "medxpertqa", a, N)] if x["qid"] in base}
                if len(v) < 12:
                    continue
                gs.append(sum(v.values()) / len(v) - sum(base[q] for q in v) / len(v))
        if gs:
            res.append((sysname, len(base), sum(base.values()) / len(base) * 100,
                        float(np.mean(gs)) * 100))
    for s_, n, p0, g in sorted(res, key=lambda x: -x[3]):
        w = 1.96 * np.sqrt(0.25 / n) * 100
        print(f"  {s_:16s} n={n:3d}  单医生 {p0:5.1f}%  平均增益 {g:+6.2f}pp  (随机波动 ±{w:.1f}pp)")


if __name__ == "__main__":
    main()
