"""能不能训练出一个选择器，把面板里的信息取出来？

第二根柱子现在的证据是「五个启发式信号都不行」。对此最自然的反驳是
「你没试过训练一个验证器」。本实验就是把这个反驳做实：
给一个学习出来的选择器**一切不公平的优势** ——
  · 在同一分布上训练，见过真值
  · 拿到面板第 0 轮的全部可观测输出（票数、置信度、专科身份、理由）
  · 按题目分组交叉验证（不泄漏），但训练与测试同域
如果它仍然取不出那部分可用空间，"面板认不出谁对"就不再是一句经验观察。

任务设成**候选选择**而不是"预测对错"：面板给出若干互异答案，
选择器为每个候选打分并取 argmax。这样它与多数投票、预言机在同一刻度上可比。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load
from experiments.grid_files import main_grid, load_main

PANEL = ("independent", "centralized", "discussion")


def episode_features(ops, gold):
    """把一条 episode 的第 0 轮意见拆成 (候选, 特征, 是否正确) 三元组。"""
    votes = collections.Counter(o["answer"] for o in ops)
    n = len(ops)
    confs = collections.defaultdict(list)
    lens = collections.defaultdict(list)
    for o in ops:
        confs[o["answer"]].append(float(o.get("confidence", 50)))
        lens[o["answer"]].append(len(str(o.get("reason", ""))))
    allc = [float(o.get("confidence", 50)) for o in ops]
    ranked = votes.most_common()
    top_v = ranked[0][1]
    second_v = ranked[1][1] if len(ranked) > 1 else 0
    rows = []
    for a, v in votes.items():
        ca, la = confs[a], lens[a]
        rows.append((
            a,
            [v / n,                                   # 得票占比
             float(v == top_v),                       # 是否多数票
             (v - second_v) / n,                      # 与次席的票差
             float(np.mean(ca)) / 100,                # 该候选的平均置信度
             float(np.max(ca)) / 100,                 # 最高置信度
             float(np.min(ca)) / 100,                 # 最低置信度
             float(np.std(ca)) / 100 if len(ca) > 1 else 0.0,
             (float(np.mean(ca)) - float(np.mean(allc))) / 100,   # 相对全体的置信度
             float(np.mean(la)) / 400,                # 理由长度
             float(np.max(la)) / 400,
             len(votes) / n,                          # 面板整体的分散度
             float(-sum((c / n) * np.log(c / n + 1e-12) for c in votes.values())),
             float(np.mean(allc)) / 100,
             float(np.std(allc)) / 100,
             n / 9.0],
            int(a == gold)))
    return rows


def main():
    rows = load_main()
    eps = []
    for r in rows:
        if r["arch"] not in PANEL or r["N"] < 3 or r.get("status") == "error":
            continue
        f0 = [o for o in (r.get("rounds") or [[]])[0] if o.get("answer")]
        if len(f0) < 3:
            continue
        g = r.get("gold")
        cand = episode_features(f0, g)
        if len(cand) < 2:            # 面板全体一致：任何选择器都无从选起
            eps.append(dict(qid=r["qid"], bench=r["bench"], cand=cand, unanimous=True))
        else:
            eps.append(dict(qid=r["qid"], bench=r["bench"], cand=cand, unanimous=False))
    print(f"可用 episode: {len(eps)}，其中面板内部有分歧的 "
          f"{sum(1 for e in eps if not e['unanimous'])} "
          f"({sum(1 for e in eps if not e['unanimous'])/len(eps)*100:.1f}%)")

    # 按题目分组的 5 折交叉验证，避免同一题同时进训练和测试
    qids = sorted({e["qid"] for e in eps})
    rng = np.random.RandomState(0)
    fold = {q: i for q, i in zip(qids, rng.randint(0, 5, len(qids)))}

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    def run(model_fn, name):
        picked, major, orac = [], [], []
        for k in range(5):
            tr = [e for e in eps if fold[e["qid"]] != k]
            te = [e for e in eps if fold[e["qid"]] == k]
            X = np.array([f for e in tr for _, f, _ in e["cand"]])
            y = np.array([lab for e in tr for _, _, lab in e["cand"]])
            clf = model_fn().fit(X, y)
            for e in te:
                F = np.array([f for _, f, _ in e["cand"]])
                s = clf.predict_proba(F)[:, 1]
                labs = [lab for _, _, lab in e["cand"]]
                picked.append(labs[int(np.argmax(s))])
                major.append(labs[int(np.argmax([f[0] for f in F]))])
                orac.append(max(labs))
        return np.mean(picked) * 100, np.mean(major) * 100, np.mean(orac) * 100

    print("\n" + "=" * 74)
    print("候选选择：学习选择器 vs 多数投票 vs 预言机（按题目分组 5 折 CV）")
    print("=" * 74)
    print(f"{'selector':34s}{'accuracy':>10s}{'vs 多数票':>11s}{'捕获率':>9s}")
    for fn, name in [(lambda: LogisticRegression(max_iter=2000, C=1.0), "logistic regression"),
                     (lambda: HistGradientBoostingClassifier(max_iter=300,
                                                             random_state=0), "gradient boosting")]:
        acc, maj, orc = run(fn, name)
        cap = (acc - maj) / (orc - maj) * 100 if orc > maj else float("nan")
        print(f"  {name:32s}{acc:9.2f}%{acc-maj:+10.2f}{cap:8.1f}%")
    print(f"  {'majority vote (reference)':32s}{maj:9.2f}%{0.0:+10.2f}{0.0:8.1f}%")
    print(f"  {'oracle (upper bound)':32s}{orc:9.2f}%{orc-maj:+10.2f}{100.0:8.1f}%")

    out = dict(n_episodes=len(eps), majority=maj, oracle=orc)
    (ROOT / "results/learned_selector.json").write_text(json.dumps(out, indent=1))
    print("\n写入 results/learned_selector.json")


if __name__ == "__main__":
    main()
