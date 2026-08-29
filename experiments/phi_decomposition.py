"""把「家族多样性」与「能力差距」对误差相关性的贡献分开。

两个要点：
  1. **phi 有数学上限。** 两个二元变量的相关系数在边缘分布不同时达不到 1：
         phi_max = sqrt( p(1-q) / (q(1-p)) ),  p = min(错误率), q = max(错误率)
     能力不同的 agent 混编时错误率必然不同，于是原始 phi 被机械地压低。
     任何用原始一致率/重叠度声称"异质集成去相关"的分析都吃这个偏差。
     正确的量是归一化相关 phi / phi_max。
  2. 有了归一化量，就能把 phi 回归到「是否同家族」和「能力差距」上，
     看换家族本身到底买不买得到独立性。
"""
import sys, pathlib, glob, json, collections, itertools
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load

# 家族 = 一次独立的训练。OpenAI 内部的 4o / 4.1 / 5 是三次独立训练；
# 开源权重模型则各自属于完全不同的生态（不同机构、不同数据、不同目标）。
FAMILY = {
    # ---- OpenAI 生态 ----
    "gpt-4o-mini": "4o", "gpt-4.1-nano": "4.1", "gpt-4.1-mini": "4.1",
    "gpt-5-nano": "5", "gpt-5-mini": "5", "gpt-5.4-nano": "5",
    # ---- 开源权重（每个是独立生态）----
    "local/medgemma-4b": "gemma", "local/medgemma-27b": "gemma",
    "local/lingshu-7b": "lingshu", "local/lingshu-32b": "lingshu",
    "local/llava-med": "llava-med",
    "local/huatuo-7b": "huatuo",
    "local/qwen2.5-7b": "qwen",
}
# 生态 = 训练它的组织。跨生态比跨家族更彻底。
ECOSYSTEM = {"4o": "openai", "4.1": "openai", "5": "openai",
             "gemma": "google", "lingshu": "alibaba", "llava-med": "microsoft",
             "huatuo": "freedomintel", "qwen": "alibaba"}
LABEL = {"gpt-4o-mini": "4o-mini", "gpt-4.1-nano": "4.1-nano", "gpt-4.1-mini": "4.1-mini",
         "gpt-5-nano": "5-nano", "gpt-5-mini": "5-mini", "gpt-5.4-nano": "5.4-nano",
         "local/medgemma-4b": "MedGemma-4B", "local/medgemma-27b": "MedGemma-27B",
         "local/lingshu-7b": "Lingshu-7B", "local/lingshu-32b": "Lingshu-32B",
         "local/llava-med": "LLaVA-Med", "local/huatuo-7b": "HuatuoGPT-V-7B",
         "local/qwen2.5-7b": "Qwen2.5-7B"}


def phi_max(p, q):
    """边缘分布固定时二元相关系数的上确界。"""
    lo, hi = min(p, q), max(p, q)
    if not (0 < lo and hi < 1):
        return np.nan
    return float(np.sqrt(lo * (1 - hi) / (hi * (1 - lo))))


def pair_phi(err_a, err_b):
    """两个模型在共同题目上的错误相关性，连同上限与归一化值。"""
    qs = sorted(set(err_a) & set(err_b))
    if len(qs) < 30:
        return None
    x = np.array([err_a[q] for q in qs], float)
    y = np.array([err_b[q] for q in qs], float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    pm = phi_max(x.mean(), y.mean())
    return dict(n=len(qs), phi=r, p_a=float(x.mean()), p_b=float(y.mean()),
                phi_max=pm, phi_norm=r / pm if pm == pm and pm > 0 else np.nan)


def collect_solo():
    """每个模型在每个 benchmark 上的逐题错误指示（单医生 CoT）。"""
    files = (sorted(glob.glob(str(ROOT / "results/G_*.jsonl")))
             + sorted(glob.glob(str(ROOT / "results/PHI_*.jsonl")))
             + sorted(glob.glob(str(ROOT / "results/OPEN_*.jsonl"))))
    rows = load(files)
    err = collections.defaultdict(dict)
    for r in rows:
        if r["arch"] != "cot" or r["N"] != 1:
            continue
        err[(r["model"], r["bench"])][r["qid"]] = int(not r["correct"])
    return err


def main():
    err = collect_solo()
    models = sorted({k[0] for k in err} & set(FAMILY))
    benches = sorted({k[1] for k in err})
    print(f"模型 {len(models)} 个: {[LABEL[m] for m in models]}")
    print(f"benchmark {len(benches)} 个: {benches}\n")

    recs = []
    for b in benches:
        for a, c in itertools.combinations(models, 2):
            if (a, b) not in err or (c, b) not in err:
                continue
            d = pair_phi(err[(a, b)], err[(c, b)])
            if not d:
                continue
            d.update(bench=b, m_a=a, m_b=c,
                     same_family=int(FAMILY[a] == FAMILY[c]),
                     same_ecosystem=int(ECOSYSTEM.get(FAMILY[a]) == ECOSYSTEM.get(FAMILY[c])),
                     acc_gap=abs(d["p_a"] - d["p_b"]))
            recs.append(d)
    if not recs:
        print("数据不足"); return

    print("=" * 92)
    print("两两模型的误差相关性（单医生 CoT，同一批题）")
    print("=" * 92)
    print(f"{'bench':16s} {'模型对':24s} {'同族':>4s} {'错误率差':>8s} "
          f"{'phi':>7s} {'phi_max':>8s} {'phi/phi_max':>11s}")
    for r in sorted(recs, key=lambda r: (r["bench"], -r["phi_norm"])):
        print(f"{r['bench']:16s} {LABEL[r['m_a']]+' | '+LABEL[r['m_b']]:24s} "
              f"{'是' if r['same_family'] else '否':>4s} {r['acc_gap']:8.3f} "
              f"{r['phi']:7.3f} {r['phi_max']:8.3f} {r['phi_norm']:11.3f}")

    print("\n" + "=" * 92)
    print("分解：换家族本身买到独立性了吗？")
    print("=" * 92)
    for lab, sel in [("同家族", [r for r in recs if r["same_family"]]),
                     ("跨家族·同生态", [r for r in recs
                                  if not r["same_family"] and r["same_ecosystem"]]),
                     ("跨生态", [r for r in recs if not r["same_ecosystem"]])]:
        if not sel:
            continue
        print(f"  {lab}  n={len(sel):3d}   原始 phi = {np.mean([r['phi'] for r in sel]):.3f}"
              f"   phi_max = {np.mean([r['phi_max'] for r in sel]):.3f}"
              f"   归一化 = {np.nanmean([r['phi_norm'] for r in sel]):.3f}")

    # 控制能力差距后再看家族效应
    cols = [np.ones(len(recs)), [r["same_family"] for r in recs],
            [r["acc_gap"] for r in recs]]
    names = ["截距", "同家族", "错误率差"]
    if len({r["same_ecosystem"] for r in recs}) > 1:      # 有跨生态数据时才加这一项
        cols.insert(2, [r["same_ecosystem"] for r in recs]); names.insert(2, "同生态")
    X = np.column_stack(cols)
    for resp in ("phi", "phi_norm"):
        y = np.array([r[resp] for r in recs], float)
        ok = ~np.isnan(y)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        yhat = X[ok] @ beta
        ss = 1 - ((y[ok] - yhat) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
        # 系数的自助置信区间
        bs = []
        rng = np.random.RandomState(0)
        idx = np.where(ok)[0]
        for _ in range(2000):
            s = rng.choice(idx, len(idx), replace=True)
            try:
                bs.append(np.linalg.lstsq(X[s], y[s], rcond=None)[0])
            except np.linalg.LinAlgError:
                pass
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        print(f"\n  回归 {resp} ~ 同家族 + 错误率差    R^2 = {ss:.3f}")
        for k, nm in enumerate(names):
            star = "" if lo[k] <= 0 <= hi[k] else "  *"
            print(f"    {nm:8s} beta = {beta[k]:+.3f}   95% CI [{lo[k]:+.3f}, {hi[k]:+.3f}]{star}")

    (ROOT / "results/phi_decomposition.json").write_text(json.dumps(recs, indent=1))
    print(f"\n写入 results/phi_decomposition.json  ({len(recs)} 个模型对)")


if __name__ == "__main__":
    main()
