"""从结果重算能力指数 I 并回写 vizstyle.CAPABILITY。

I = 单医生 CoT 在三个 benchmark 上的准确率均值，与 paper/tables/table6_capability_index.tex
同源。这个值是 fig1 / fig6 的 x 轴坐标，手工维护过一次就漂移了：Gemini 两个模型曾被写成
54.2 / 81.7，真值是 50.3 / 70.7 —— 论文"在匹配能力上做跨厂商对比"的论断正依赖这个数。
所以改成从数据重算，不再手填。三个 benchmark 齐全的模型才会写入。
"""
import sys, pathlib, glob, re, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load

BENCHES = ("medxpertqa", "medagentsbench", "medqa")


def capability_from_results():
    files = [f for f in sorted(glob.glob(str(ROOT / "results/*.jsonl")))
             if re.search(r"/(G|GEM|CLA|OPEN|OR)_", f) and "generic" not in f]
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load(files):
        if r.get("arch") == "cot" and r.get("N") == 1:
            acc[r["model"]][r["bench"]].append(r["correct"])
    out = {}
    for m, d in acc.items():
        if all(b in d and len(d[b]) >= 200 for b in BENCHES):
            out[m] = round(float(np.mean([100 * np.mean(d[b]) for b in BENCHES])), 1)
    return out, acc


def main():
    cap, acc = capability_from_results()
    p = ROOT / "experiments/vizstyle.py"
    src = p.read_text()
    body = ",\n              ".join(f'"{m}": {v}' for m, v in sorted(cap.items(), key=lambda kv: kv[1]))
    new = f"CAPABILITY = {{{body}}}"
    src2 = re.sub(r"CAPABILITY = \{[^}]*\}", new, src, count=1)
    if src2 != src:
        p.write_text(src2)
    print(f"{'model':<30}{'I':>7}   分量")
    for m, v in sorted(cap.items(), key=lambda kv: kv[1]):
        comp = "  ".join(f"{b[:4]}={100*np.mean(acc[m][b]):.1f}" for b in BENCHES)
        print(f"{m:<30}{v:>7.1f}   {comp}")
    skipped = [m for m in acc if m not in cap]
    if skipped:
        print("\n数据不全、未写入:")
        for m in skipped:
            print(f"  {m}: 只有 {sorted(b for b in BENCHES if b in acc[m])}")
    return cap


if __name__ == "__main__":
    main()
