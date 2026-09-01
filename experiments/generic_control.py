"""专科角色 vs 通用内科医生：专科身份本身值多少。

面板增益可能来自两处：把问题问了 N 遍，或者让这 N 遍以不同专科的身份去问。
本对照把后者拿掉 —— 每个 panelist 都是无差别的通用内科主治，面板规模是唯一
变化的东西。两条臂共用同一批 250 道题、同一模型、同一温度与架构。

注意：通用角色下每个 agent 的系统提示与用户提示完全相同，agent 索引只体现在
seed 上，而推理模型会在进缓存键之前丢掉 seed。因此 round0_opinions 必须把
索引写进 cache tag，否则 N 个 agent 会命中同一条缓存、返回同一个答案，
"零多样性" 就成了构造出来的假象而不是测量结果。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load, mcnemar


def main():
    gen = load([str(ROOT / "results/CTRL_T2_generic.jsonl")])
    spec = load([str(ROOT / "results/G_T2_medxpertqa.jsonl")])

    def cells(rows):
        c = collections.defaultdict(dict)
        for r in rows:
            c[(r["arch"], r["N"])][r["qid"]] = r["correct"]
        return c

    G, S = cells(gen), cells(spec)
    base = S.get(("cot", 1), {})

    print("=" * 76)
    print("专科角色 vs 通用内科医生（gpt-5-nano / MedXpertQA, 同一批 250 题）")
    print("=" * 76)
    print(f"{'arch':14s}{'N':>3s}{'专科':>9s}{'通用内科':>10s}{'差':>8s}{'p':>9s}")
    rows_out = []
    for k in sorted(G):
        if k not in S:
            continue
        ids = sorted(set(G[k]) & set(S[k]))
        if len(ids) < 50:
            continue
        a = {q: S[k][q] for q in ids}; b = {q: G[k][q] for q in ids}
        sa = float(np.mean(list(a.values())) * 100); sb = float(np.mean(list(b.values())) * 100)
        _, _, p = mcnemar(b, a)
        rows_out.append(dict(arch=k[0], N=k[1], spec=sa, gen=sb, diff=sa - sb, p=p))
        print(f"  {k[0]:12s}{k[1]:3d}{sa:8.1f}%{sb:9.1f}%{sa - sb:+8.1f}{p:9.3f}")

    d = [r["diff"] for r in rows_out]
    nsig = sum(1 for r in rows_out if r["p"] < 0.05)
    print(f"\n  平均差异 {np.mean(d):+.2f}pp，{nsig}/{len(d)} 个对比显著")

    if base:
        p0 = float(np.mean(list(base.values())) * 100)
        bg = max((r["gen"] for r in rows_out if r["N"] >= 3), default=float("nan"))
        bs = max((r["spec"] for r in rows_out if r["N"] >= 3), default=float("nan"))
        print("\n" + "=" * 76)
        print("把面板增益拆成「问 N 遍」和「以不同专科身份问」")
        print("=" * 76)
        print(f"  单医生 (CoT)                       {p0:5.1f}%")
        print(f"  通用内科面板，最佳 N>=3            {bg:5.1f}%   +{bg - p0:.1f}pp  <- 只靠问 N 遍")
        print(f"  专科面板，最佳 N>=3                {bs:5.1f}%   +{bs - p0:.1f}pp")
        if bs > p0:
            print(f"\n  专科身份贡献 {bs - bg:+.1f}pp，占面板总增益的 {(bs - bg) / (bs - p0) * 100:.0f}%")

    (ROOT / "results/generic_control.json").write_text(json.dumps(rows_out, indent=1))
    print("\n写入 results/generic_control.json")


if __name__ == "__main__":
    main()
