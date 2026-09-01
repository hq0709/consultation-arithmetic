"""面板能不能认出自己什么时候是对的 —— 把所有候选信号一次测完。

分诊门要工作，需要一个在**召集面板之前或之中**就能算出来的量，
它得能区分「这次面板会答对」和「这次会答错」。候选：
  自陈置信度 / 答案分布熵 / 多数票占比 / 最低成员置信度 / 置信度离散度。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load
from experiments.grid_files import main_grid, load_main
from sklearn.metrics import roc_auc_score

PANEL = ("independent", "centralized", "discussion")


def youden(x, y, grid=400):
    th = np.unique(x)
    step = max(1, len(th) // grid)
    best = 0.0
    npos, nneg = max(1, (y == 1).sum()), max(1, (y == 0).sum())
    for t in th[::step]:
        pr = x >= t
        best = max(best, (pr & (y == 1)).sum() / npos - (pr & (y == 0)).sum() / nneg)
    return float(best)


def main():
    rows = load_main()
    sig = collections.defaultdict(list)
    for r in rows:
        if r["arch"] not in PANEL or r["N"] < 3:
            continue
        f = (r.get("rounds") or [[]])[0]
        ans = [o.get("answer") for o in f if o.get("answer")]
        cf = [o.get("confidence", 50) for o in f if o.get("answer")]
        if len(ans) < 3:
            continue
        c = collections.Counter(ans); n = len(ans)
        p = np.array([v / n for v in c.values()])
        ent = float(-(p * np.log(p + 1e-12)).sum())
        y = int(r["correct"])
        sig["stated confidence (mean)"].append((float(np.mean(cf)), y))
        sig["answer entropy"].append((-ent, y))
        sig["majority share"].append((max(c.values()) / n, y))
        sig["lowest member confidence"].append((float(min(cf)), y))
        sig["confidence spread"].append((-float(np.std(cf)), y))

    out = {}
    print("=" * 74)
    print("面板自身可得的路由信号，区分「本次面板答对/答错」的能力")
    print("=" * 74)
    print(f"{'signal':30s}{'AUC':>9s}{'Youden J':>11s}{'n':>9s}")
    for k, v in sig.items():
        x = np.array([a for a, _ in v]); y = np.array([b for _, b in v])
        auc = float(roc_auc_score(y, x)); j = youden(x, y)
        out[k] = dict(auc=auc, youden=j, n=len(v))
        print(f"  {k:28s}{auc:9.3f}{j:11.3f}{len(v):9d}")
    aucs = [v["auc"] for v in out.values()]
    print(f"\n  全部信号的 AUC 落在 [{min(aucs):.3f}, {max(aucs):.3f}] —— 极差 {max(aucs)-min(aucs):.3f}")
    (ROOT / "results/selection_signals.json").write_text(json.dumps(out, indent=1))
    print("\n写入 results/selection_signals.json")


if __name__ == "__main__":
    main()
