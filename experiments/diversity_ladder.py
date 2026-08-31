"""多样性阶梯：论文 Table 5 的唯一数字来源。

四档，依次放宽"两个成员有多不一样"：
  1. 同一个模型，不同专科角色提示
  2. 同一家族内的不同 checkpoint
  3. 同一生态内的不同家族（OpenAI 的 4o / 4.1 / 5）
  4. 不同生态（OpenAI <-> Google Gemini）
关键量是归一化相关 phi/phi_max —— 原始 phi 在成员错误率不同时会被机械压低。
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np


def neff(phi, N=9):
    return N / (1 + (N - 1) * phi)


def main():
    recs = json.loads((ROOT / "results/phi_decomposition.json").read_text())
    ind = json.loads((ROOT / "results/independence.json").read_text())
    FULL = ("independent", "centralized", "discussion")
    role_phi = float(np.mean([r["phi0"] for r in ind if r["arch"] in FULL and r["N"] >= 3]))

    rows = [("One model, specialty prompts", role_phi, 0.958, None)]
    for lab, sel in [
        ("Distinct models, one family",
         [r for r in recs if r["same_family"]]),
        ("Distinct families, one vendor",
         [r for r in recs if not r["same_family"] and r["same_ecosystem"]]),
        ("Distinct vendors (OpenAI/Google)",
         [r for r in recs if not r["same_ecosystem"]]),
    ]:
        rows.append((lab, float(np.mean([r["phi"] for r in sel])),
                     float(np.mean([r["phi_max"] for r in sel])), len(sel)))

    print("=" * 78)
    print("多样性阶梯（Table 5 的权威数字）")
    print("=" * 78)
    print(f"{'来源':34s}{'n':>5s}{'phi':>8s}{'phi_max':>9s}{'归一化':>8s}{'N_eff@9':>9s}")
    out = []
    for lab, ph, pm, n in rows:
        out.append(dict(source=lab, n=n, phi=ph, phi_max=pm,
                        phi_norm=ph / pm, neff9=neff(ph)))
        print(f"  {lab:32s}{(n if n else '--'):>5}{ph:8.3f}{pm:9.3f}{ph/pm:8.3f}{neff(ph):9.2f}")
    print(f"  {'fully independent':32s}{'--':>5}{0:8.3f}{1:9.3f}{0:8.3f}{9:9.2f}")

    # 能力匹配子集：phi_max 接近 1，机械假象最小
    print("\n" + "=" * 78)
    print("只保留能力匹配的配对（两成员错误率差 < 0.05）")
    print("=" * 78)
    matched = {}
    for lab, cond in [("same vendor", lambda r: r["same_ecosystem"]),
                      ("cross vendor", lambda r: not r["same_ecosystem"])]:
        s = [r for r in recs if r["acc_gap"] < 0.05 and cond(r) and not r["same_family"]]
        if not s:
            continue
        ph = float(np.mean([r["phi"] for r in s])); pm = float(np.mean([r["phi_max"] for r in s]))
        matched[lab] = dict(n=len(s), phi=ph, phi_max=pm, phi_norm=ph / pm, neff9=neff(ph))
        print(f"  {lab:16s} n={len(s):2d}  phi={ph:.3f}  phi_max={pm:.3f}"
              f"  归一化={ph/pm:.3f}  N_eff@9={neff(ph):.2f}")

    (ROOT / "results/diversity_ladder.json").write_text(
        json.dumps(dict(ladder=out, matched=matched), indent=1))
    print("\n写入 results/diversity_ladder.json")


if __name__ == "__main__":
    main()
