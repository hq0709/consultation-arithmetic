"""Fit NMI 2026's scaling principle on the medical grid, and test its three claims.

NMI's functional form (arXiv:2512.08296):
  P = b0 + b1*I + b2*I^2 + b3*log(1+T) + b4*log(1+n_a) + b5*log(1+O%) + b6*c + b7*R
        + b8*E_c + b9*log(1+A_e) + b10*P_SA + interactions

Our tasks are TOOL-FREE, so T = 0 and every log(1+T) term vanishes identically. We fit the
T = 0 slice and say so: it is the limiting case NMI's own design could not reach, and it is
the cleanest possible test of the capability-ceiling claim with the tool confound removed.

I (capability index) is task-grounded: each model's single-doctor CoT accuracy on MedQA
(NMI report R^2 = 0.413 with a task-grounded metric vs 0.373 with a generic one).
"""
from __future__ import annotations
import sys, pathlib, json, math, argparse, collections, itertools
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import statsmodels.api as sm
from experiments.analyze import load
from mechanisms.nmi_metrics import (config_metrics, fit_turn_powerlaw, fit_msg_density_log)
from panels.architectures import NMI_CLASS

# task-grounded capability index I: single-doctor CoT accuracy on MedQA (tier probe)
CAPABILITY = {"gpt-4.1-nano": 0.750, "gpt-4o-mini": 0.875, "gpt-4.1-mini": 0.850,
              "gpt-5-nano": 0.950, "gpt-5.4-nano": 0.950, "gpt-5-mini": 0.975}


def build_table(files, canonical=True):
    """canonical=True 时只保留每个 benchmark 的官方 250 题。见 grid_files.load_main
    的说明：网格分批跑，早期几批用的是 500 题文件，不过滤就不是同题比较。"""
    rows = load([ROOT / "results" / f for f in files])
    if canonical:
        from experiments.grid_files import canonical_items
        rows = [r for r in rows if r.get("qid") in canonical_items(r.get("bench"))]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["bench"], r["arch"], r["N"], r["seed"])].append(r)
    sas = {k: v for k, v in by.items() if k[2] == "cot"}
    recs = []
    for k, eps in by.items():
        model, bench, arch, N, seed = k
        if arch in ("cot", "zeroshot"):
            continue
        base = sas.get((model, bench, "cot", 1, seed)) or sas.get((model, bench, "cot", 1, 1))
        if not base:
            continue
        m = config_metrics(eps, base)
        if not m:
            continue
        m.update({"model": model, "bench": bench, "arch": arch, "N": N, "seed": seed,
                  "nmi_class": NMI_CLASS.get(arch, arch), "I": CAPABILITY.get(model, np.nan),
                  "T_tools": 0})
        recs.append(m)
    return pd.DataFrame(recs)


def fit(df, label=""):
    d = df.dropna(subset=["I", "accuracy", "sas_baseline"]).copy()
    if len(d) < 12:
        print(f"[{label}] 配置数 {len(d)} 太少，跳过回归"); return None
    d["log_na"] = np.log1p(d["n_agents"])
    d["log_O"] = np.log1p(d["overhead_pct"].clip(lower=0))
    d["log_Ae"] = np.log1p(d["error_amp"].clip(lower=0))
    d["I2"] = d["I"] ** 2
    terms = ["I", "I2", "log_na", "log_O", "msg_density", "redundancy", "efficiency",
             "log_Ae", "sas_baseline"]
    d["I_x_Ec"] = d["I"] * d["efficiency"]
    d["Ae_x_PSA"] = d["error_amp"] * d["sas_baseline"]
    d["R_x_na"] = d["redundancy"] * d["n_agents"]
    d["c_x_I"] = d["msg_density"] * d["I"]
    d["PSA_x_logna"] = d["sas_baseline"] * d["log_na"]
    terms += ["I_x_Ec", "Ae_x_PSA", "R_x_na", "c_x_I", "PSA_x_logna"]
    terms = [t for t in terms if d[t].std() > 1e-9]
    # drop collinear columns so the design matrix is identifiable
    keep, M = [], np.empty((len(d), 0))
    for t in terms:
        cand = np.column_stack([M, d[t].astype(float).values]) if M.size else d[[t]].astype(float).values
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(d)), cand])) == cand.shape[1] + 1:
            keep.append(t); M = cand
    terms = keep
    if not terms:
        print(f"[{label}] 所有预测变量共线，跳过"); return None
    X = sm.add_constant(d[terms].astype(float))
    y = d["accuracy"].astype(float)
    # cluster-robust SEs by benchmark (R1 #1); needs >1 cluster
    if d["bench"].nunique() > 1:
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["bench"]})
    else:
        res = sm.OLS(y, X).fit(cov_type="HC1")
    print(f"\n=== NMI 标度律拟合{label} (n_config={len(d)}, T=0 切面) ===")
    print(f"R² = {res.rsquared:.3f}   adj R² = {res.rsquared_adj:.3f}")
    print(f"{'项':16s} {'β':>9s} {'SE':>8s} {'p':>9s}")
    for t in ["const"] + terms:
        b, se, p = res.params[t], res.bse[t], res.pvalues[t]
        star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
        print(f"{t:16s} {b:9.3f} {se:8.3f} {p:9.4f} {star}")
    # k-fold CV (NMI report cross-validated R^2)
    idx = np.arange(len(d)); rng = np.random.RandomState(0); rng.shuffle(idx)
    folds = np.array_split(idx, min(5, len(d)))
    ss_res = ss_tot = 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        if len(tr) < len(terms) + 2:
            continue
        m = sm.OLS(y.iloc[tr], X.iloc[tr]).fit()
        pred = m.predict(X.iloc[f])
        ss_res += ((y.iloc[f] - pred) ** 2).sum()
        ss_tot += ((y.iloc[f] - y.mean()) ** 2).sum()
    if ss_tot:
        print(f"5-fold 交叉验证 R² = {1 - ss_res/ss_tot:.3f}   (NMI: 0.513 预印本 / 0.373 正刊)")
    return res


def fit_gain(df):
    """因变量改为协作增益 (accuracy - P_SA)。

    必须这样做：NMI 的原式以 accuracy 为因变量，而 P_SA 是同一批题目上同一模型的
    单医生准确率。在我们的设计里（同题、同模型、只换协调结构）两者近乎同义，R^2
    被 P_SA 主导而虚高到 0.998。NMI 自己没有这个问题，因为它跨 6 个差异极大的
    benchmark 和 3 个模型家族。以增益为因变量，才是在问"协调本身贡献了什么"。"""
    d = df.dropna(subset=["I", "accuracy", "sas_baseline"]).copy()
    d["gain"] = d["accuracy"] - d["sas_baseline"]
    d["log_na"] = np.log1p(d["n_agents"]); d["I2"] = d["I"] ** 2
    d["PSA2"] = d["sas_baseline"] ** 2
    d["log_O"] = np.log1p(d["overhead_pct"].clip(lower=0))
    d["log_Ae"] = np.log1p(d["error_amp"].clip(lower=0))
    d["PSA_x_logna"] = d["sas_baseline"] * d["log_na"]
    terms = ["I", "I2", "log_na", "sas_baseline", "PSA2", "log_O", "msg_density",
             "redundancy", "efficiency", "log_Ae", "PSA_x_logna"]
    terms = [t for t in terms if d[t].std() > 1e-9]
    X = sm.add_constant(d[terms].astype(float)); y = d["gain"].astype(float)
    # Breusch-Pagan p=0.0001 -> 残差异方差，普通 SE 不可用。同时按 benchmark 聚类
    # （同一 benchmark 内的配置共享题目集），并用 HC3 兜底。
    res = (sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["bench"]})
           if d["bench"].nunique() > 2 else sm.OLS(y, X).fit(cov_type="HC3"))
    print(f"\n=== 增益模型 gain = acc - P_SA  (n={len(d)}) ===")
    print(f"R² = {res.rsquared:.3f}")
    print(f"{'项':16s} {'β':>9s} {'SE':>8s} {'p':>9s}")
    for t in ["const"] + terms:
        b, se, pv = res.params[t], res.bse[t], res.pvalues[t]
        star = "***" if pv < .001 else ("**" if pv < .01 else ("*" if pv < .05 else ""))
        print(f"{t:16s} {b:9.4f} {se:8.4f} {pv:9.4f} {star}")
    # 二次式的顶点：协作最有效的 P_SA
    if "PSA2" in terms and abs(res.params["PSA2"]) > 1e-9:
        vtx = -res.params["sas_baseline"] / (2 * res.params["PSA2"])
        curv = "倒 U（有最优带）" if res.params["PSA2"] < 0 else "正 U"
        print(f"\n  P_SA 的二次项 β={res.params['PSA2']:+.3f} (p={res.pvalues['PSA2']:.4f}) -> {curv}")
        if 0 < vtx < 1:
            print(f"  协作收益最大的单医生基线 P_SA = {vtx*100:.1f}%")
    idx = np.arange(len(d)); rng = np.random.RandomState(0); rng.shuffle(idx)
    ss_res = ss_tot = 0.0
    for f in np.array_split(idx, 5):
        tr = np.setdiff1d(idx, f)
        m = sm.OLS(y.iloc[tr], X.iloc[tr]).fit()
        pr = m.predict(X.iloc[f])
        ss_res += ((y.iloc[f] - pr) ** 2).sum(); ss_tot += ((y.iloc[f] - y.mean()) ** 2).sum()
    if ss_tot:
        print(f"  5-fold 交叉验证 R² = {1-ss_res/ss_tot:.3f}")
    return res


def test_capability_ceiling(df):
    """NMI claim 1: coordination yields NEGATIVE returns once P_SA > 45%."""
    d = df.dropna(subset=["sas_baseline"]).copy()
    d["gain"] = d["accuracy"] - d["sas_baseline"]
    print("\n=== NMI 断言 1：45% 能力天花板在医学上成立吗？ ===")
    print(f"{'单医生基线 P_SA':>16s} {'n_config':>9s} {'平均协作增益':>12s} {'>0 的比例':>10s}")
    for lo, hi, lab in [(0, .30, "<30%"), (.30, .45, "30–45%"), (.45, .60, "45–60%"), (.60, 1.01, ">60%")]:
        s = d[(d.sas_baseline >= lo) & (d.sas_baseline < hi)]
        if len(s) == 0:
            continue
        print(f"{lab:>16s} {len(s):9d} {s.gain.mean()*100:+11.2f}pp {(s.gain>0).mean()*100:9.0f}%")
    below = d[d.sas_baseline < .45]["gain"]; above = d[d.sas_baseline >= .45]["gain"]
    if len(below) > 2 and len(above) > 2:
        from scipy import stats
        t, p = stats.ttest_ind(below, above, equal_var=False)
        print(f"\n阈值下 ({len(below)} 配置) 增益 {below.mean()*100:+.2f}pp  vs  "
              f"阈值上 ({len(above)} 配置) {above.mean()*100:+.2f}pp   Welch t={t:.2f}, p={p:.4f}")
    if len(d) > 8:
        X = sm.add_constant(d[["sas_baseline"]].astype(float))
        r = sm.OLS(d["gain"].astype(float), X).fit()
        b = r.params["sas_baseline"]
        print(f"gain ~ P_SA 斜率 β = {b:.3f} (p={r.pvalues['sas_baseline']:.4f})   "
              f"[NMI 报告 β = −0.408, p<0.001]")
        if b < 0 and r.params["const"] != 0:
            print(f"医学上的零交叉点 P_SA* = {-r.params['const']/b*100:.1f}%   [NMI: 45%]")


def test_error_amplification(df):
    """NMI claim 3: Independent amplifies 17.2x, Centralized contains to 4.4x."""
    print("\n=== NMI 断言 3：错误放大是架构相关的吗？ ===")
    print(f"{'NMI 架构类':>18s} {'n':>4s} {'A_e 错误放大':>12s} {'Absorb 吸收':>12s} {'消息密度 c':>11s} {'轮数 T':>8s}")
    for cls, g in df.groupby("nmi_class"):
        print(f"{cls:>18s} {len(g):4d} {g.error_amp.mean():12.2f} "
              f"{g.absorb.mean()*100:+11.1f}% {g.msg_density.mean():11.3f} {g.turns.mean():8.1f}")
    print("  [NMI: Independent 17.2x 放大 / +4.6% 无纠错；Centralized 4.4x；"
          "有验证的架构平均吸收 22.7% (95% CI 20.1–25.3)]")


def decision_boundary(res, df):
    """NMI §4.3: P_SA* = beta_4 / beta_17 (standardized), denormalised ~ 0.45.
    beta_4 = log(1+n_a) main effect, beta_17 = P_SA x log(1+n_a) interaction."""
    if res is None:
        return
    b4 = res.params.get("log_na"); b17 = res.params.get("PSA_x_logna")
    print("\n=== 决策边界（NMI §4.3 的公式）===")
    if b4 is None or b17 is None or abs(b17) < 1e-9:
        print("  log_na 或 PSA x log_na 项不可识别，跳过"); return
    star = b4 / -b17 if b17 < 0 else b4 / b17
    print(f"  beta_4 (log n_a)      = {b4:+.4f}")
    print(f"  beta_17 (P_SA x log n_a) = {b17:+.4f}")
    print(f"  P_SA* = beta_4/|beta_17| = {star:.3f}   "
          f"[NMI: 0.063/0.408 = 0.154 标准化 -> 约 0.45 原始尺度]")


def success_per_1k_tokens(df):
    """NMI §5.1: SAS 67.7 / Centralized 21.5 / Decentralized 23.9 / Hybrid 13.6."""
    print("\n=== 每千 token 的成功数（NMI §5.1）===")
    print(f"{'NMI 架构类':>18s} {'成功/1k tok':>12s} {'相对 SAS':>10s}")
    d = df.copy()
    d["succ_per_1k"] = d["accuracy"] / (d["tokens"] / 1000.0)
    base = None
    g = d.groupby("nmi_class")["succ_per_1k"].mean().sort_values(ascending=False)
    for cls, v in g.items():
        if base is None:
            base = v
        print(f"{cls:>18s} {v:12.1f} {base/v:9.1f}x")
    print("  [NMI: SAS 67.7 / Decentralized 23.9 (2.8x) / Centralized 21.5 (3.1x) / Hybrid 13.6 (5.0x)]")


def cross_domain(df):
    """NMI §4.4: 架构排名跨域稳定性 Kendall tau=0.89；留一域交叉验证 R^2=0.89。"""
    from scipy import stats as st
    print("\n=== 跨 benchmark 泛化（NMI §4.4）===")
    benches = sorted(df.bench.unique())
    if len(benches) < 2:
        print("  benchmark 不足 2 个，跳过"); return
    rank = {}
    for b in benches:
        g = df[df.bench == b].groupby("nmi_class")["accuracy"].mean()
        rank[b] = g
    common = set.intersection(*[set(r.index) for r in rank.values()])
    if len(common) < 3:
        print("  共同架构类不足，跳过"); return
    taus = []
    for i in range(len(benches)):
        for j in range(i + 1, len(benches)):
            a = [rank[benches[i]][c] for c in sorted(common)]
            b_ = [rank[benches[j]][c] for c in sorted(common)]
            t, p = st.kendalltau(a, b_)
            taus.append(t)
            print(f"  {benches[i]} vs {benches[j]}: Kendall tau = {t:+.3f} (p={p:.3f})")
    if taus:
        print(f"  平均 tau = {sum(taus)/len(taus):+.3f}   [NMI: 0.89]")
    # 留一 benchmark 交叉验证
    d = df.dropna(subset=["I", "accuracy", "sas_baseline"]).copy()
    if len(d) > 20 and d.bench.nunique() > 2:
        d["log_na"] = np.log1p(d["n_agents"]); d["I2"] = d["I"] ** 2
        terms = [t for t in ["I", "I2", "log_na", "sas_baseline", "efficiency", "redundancy"]
                 if d[t].std() > 1e-9]
        ss_res = ss_tot = 0.0
        for b in benches:
            tr = d[d.bench != b]; te = d[d.bench == b]
            if len(tr) < len(terms) + 3 or len(te) == 0:
                continue
            m = sm.OLS(tr["accuracy"].astype(float),
                       sm.add_constant(tr[terms].astype(float))).fit()
            X = sm.add_constant(te[terms].astype(float), has_constant="add")
            pred = m.predict(X)
            ss_res += ((te["accuracy"] - pred) ** 2).sum()
            ss_tot += ((te["accuracy"] - d["accuracy"].mean()) ** 2).sum()
        if ss_tot:
            print(f"  留一 benchmark 交叉验证 R² = {1-ss_res/ss_tot:.3f}   [NMI: 0.89]")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/scaling_law_table.csv")
    a = ap.parse_args()
    df = build_table(a.files)
    if df.empty:
        print("没有可用配置"); return
    df.to_csv(ROOT / a.out, index=False)
    print(f"{len(df)} 个配置 -> {a.out}")
    print(df.groupby(["model", "nmi_class"]).agg(
        n=("n", "size"), acc=("accuracy", "mean"), turns=("turns", "mean"),
        A_e=("error_amp", "mean")).round(3).to_string())
    pl = fit_turn_powerlaw([(r.n_agents, r.turns) for r in df.itertuples()])
    if pl:
        print(f"\n轮数幂律 T = {pl['a']:.2f}*(n+0.5)^{pl['exponent']:.3f}, "
              f"R²={pl['r2']:.3f}   [NMI: 2.72*(n+0.5)^1.724, R²=0.974]")
    md = fit_msg_density_log([(r.msg_density, r.accuracy) for r in df.itertuples()])
    if md:
        print(f"消息密度 S = {md['intercept']:.3f} + {md['slope']:.3f}*ln(c), "
              f"R²={md['r2']:.3f}   [NMI: 0.73 + 0.28*ln(c), R²=0.68, c*=0.39]")
    test_capability_ceiling(df)
    test_error_amplification(df)
    success_per_1k_tokens(df)
    cross_domain(df)
    res = fit(df)
    decision_boundary(res, df)
    fit_gain(df)


if __name__ == "__main__":
    main()
