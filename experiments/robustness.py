"""B3–B7 —— 严格照 NMI §4.3 的统计稳健性套件：
  B3 Bootstrap n=1000 系数稳定性
  B4 残差诊断（Shapiro–Wilk 正态性、Breusch–Pagan 同方差性、残差标准误）
  B5 Lasso / Ridge 正则化对比（10 折 CV 选 lambda）
  B6 嵌套模型对比（完整 vs 仅智能指数 vs 仅协调指标 vs 仅基线）
  B7 外推 n=6–10 的 bootstrap 预测区间
"""
import sys, pathlib, json, glob
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats as st
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from experiments.scaling_law import build_table
from mechanisms.nmi_metrics import fit_turn_powerlaw

TERMS = ["I", "I2", "log_na", "sas_baseline", "PSA2", "log_O", "msg_density",
         "redundancy", "efficiency", "log_Ae", "PSA_x_logna"]


def prep(df):
    d = df.dropna(subset=["I", "accuracy", "sas_baseline"]).copy()
    d["gain"] = d["accuracy"] - d["sas_baseline"]
    d["log_na"] = np.log1p(d["n_agents"]); d["I2"] = d["I"] ** 2
    d["PSA2"] = d["sas_baseline"] ** 2
    d["log_O"] = np.log1p(d["overhead_pct"].clip(lower=0))
    d["log_Ae"] = np.log1p(d["error_amp"].clip(lower=0))
    d["PSA_x_logna"] = d["sas_baseline"] * d["log_na"]
    return d, [t for t in TERMS if d[t].std() > 1e-9]


def main():
    df = build_table(sorted(p.name for p in (ROOT / "results").glob("G_*.jsonl")))
    d, terms = prep(df)
    X = sm.add_constant(d[terms].astype(float)); y = d["gain"].astype(float)
    res = sm.OLS(y, X).fit()
    print(f"增益模型: n={len(d)}, R²={res.rsquared:.3f}, 残差标准误 σ̂={np.sqrt(res.scale):.4f}"
          f"   [NMI: σ̂=0.118]")

    # ---- B3 Bootstrap ----
    rng = np.random.RandomState(0); B = 1000
    boot = np.zeros((B, X.shape[1]))
    for b in range(B):
        idx = rng.randint(0, len(d), len(d))
        try:
            boot[b] = sm.OLS(y.iloc[idx], X.iloc[idx]).fit().params.values
        except Exception:
            boot[b] = np.nan
    bse = np.nanstd(boot, axis=0)
    big = [(t, res.params[t], bse[i]) for i, t in enumerate(X.columns) if abs(res.params[t]) > 0.1]
    print(f"\nB3 Bootstrap (n={B}): |β|>0.1 的项共 {len(big)} 个，最大 bootstrap SE = "
          f"{max((b[2] for b in big), default=0):.4f}   [NMI: 全部 <0.015]")
    for t, b_, se in sorted(big, key=lambda x: -abs(x[1]))[:6]:
        print(f"    {t:16s} β={b_:+.4f}  bootSE={se:.4f}")

    # ---- B4 残差诊断 ----
    sw = st.shapiro(res.resid)
    bp = sm.stats.diagnostic.het_breuschpagan(res.resid, X)
    print(f"\nB4 残差诊断: Shapiro–Wilk p={sw.pvalue:.4f} [NMI 0.412] · "
          f"Breusch–Pagan p={bp[1]:.4f} [NMI 0.298]")
    print(f"    正态性{'通过' if sw.pvalue > .05 else '不通过'}；"
          f"同方差性{'通过' if bp[1] > .05 else '不通过'}")

    # ---- B5 正则化 ----
    Z = StandardScaler().fit_transform(d[terms].astype(float))
    la = LassoCV(cv=10, random_state=0, max_iter=20000).fit(Z, y)
    ri = RidgeCV(alphas=np.logspace(-3, 3, 60), cv=10).fit(Z, y)
    kept = int(np.sum(np.abs(la.coef_) > 1e-8))

    def cv_r2(model):
        idx = np.arange(len(d)); rng2 = np.random.RandomState(0); rng2.shuffle(idx)
        sr = stt = 0.0
        for f in np.array_split(idx, 5):
            tr = np.setdiff1d(idx, f)
            m = model.__class__(**model.get_params()).fit(Z[tr], y.iloc[tr])
            pr = m.predict(Z[f])
            sr += ((y.iloc[f] - pr) ** 2).sum(); stt += ((y.iloc[f] - y.mean()) ** 2).sum()
        return 1 - sr / stt
    print(f"\nB5 正则化: Lasso 保留 {kept}/{len(terms)} 个预测变量, CV R²={cv_r2(la):.3f} "
          f"[NMI: 16/20, 0.506] · Ridge CV R²={cv_r2(ri):.3f} [NMI: 0.509]")

    # ---- B6 嵌套模型 ----
    def cv_ols(cols):
        if not cols:
            return 0.0
        Xc = sm.add_constant(d[cols].astype(float))
        idx = np.arange(len(d)); rng2 = np.random.RandomState(0); rng2.shuffle(idx)
        sr = stt = 0.0
        for f in np.array_split(idx, 5):
            tr = np.setdiff1d(idx, f)
            m = sm.OLS(y.iloc[tr], Xc.iloc[tr]).fit()
            pr = m.predict(Xc.iloc[f])
            sr += ((y.iloc[f] - pr) ** 2).sum(); stt += ((y.iloc[f] - y.mean()) ** 2).sum()
        return 1 - sr / stt
    coord = [t for t in ("msg_density", "redundancy", "efficiency", "log_Ae", "log_O") if t in terms]
    print(f"\nB6 嵌套模型对比 (5 折 CV R²):")
    print(f"    完整模型              {cv_ols(terms):6.3f}   [NMI 完整: 0.513]")
    print(f"    仅能力指数 I, I²      {cv_ols([t for t in ('I','I2') if t in terms]):6.3f}"
          f"   [NMI 仅 intelligence: 0.28]")
    print(f"    仅协调指标            {cv_ols(coord):6.3f}")
    print(f"    仅单医生基线 P_SA     {cv_ols([t for t in ('sas_baseline','PSA2') if t in terms]):6.3f}")
    print(f"    仅 panel 规模 n_a     {cv_ols([t for t in ('log_na',) if t in terms]):6.3f}")

    # ---- B7 轮数外推 ----
    pts = [(r.n_agents, r.turns) for r in df.itertuples() if r.turns > 0]
    pl = fit_turn_powerlaw(pts)
    rng3 = np.random.RandomState(0)
    preds = {n: [] for n in range(6, 11)}
    arr = np.array(pts)
    for _ in range(1000):
        s_ = arr[rng3.randint(0, len(arr), len(arr))]
        f = fit_turn_powerlaw([tuple(x) for x in s_])
        if f:
            for n in preds:
                preds[n].append(f["a"] * (n + .5) ** f["exponent"])
    print(f"\nB7 轮数外推 (幂律 T={pl['a']:.2f}(n+0.5)^{pl['exponent']:.3f}, bootstrap n=1000):")
    for n in range(6, 11):
        v = np.array(preds[n])
        print(f"    n={n:2d}:  T = {np.median(v):5.1f}  95% PI [{np.percentile(v,2.5):.1f}, "
              f"{np.percentile(v,97.5):.1f}]")
    print(f"    [NMI: n=6 时 12.8–20.1 轮；Centralized 外推可达 85–130]")

    out = {"sigma": float(np.sqrt(res.scale)), "shapiro_p": float(sw.pvalue),
           "bp_p": float(bp[1]), "lasso_kept": kept, "lasso_cv_r2": float(cv_r2(la)),
           "ridge_cv_r2": float(cv_r2(ri)), "cv_full": float(cv_ols(terms)),
           "cv_intelligence_only": float(cv_ols([t for t in ("I", "I2") if t in terms])),
           "cv_coord_only": float(cv_ols(coord)),
           "max_boot_se": float(max((b[2] for b in big), default=0)),
           "turn_extrap": {n: [float(np.percentile(preds[n], 2.5)),
                               float(np.median(preds[n])),
                               float(np.percentile(preds[n], 97.5))] for n in preds}}
    (ROOT / "results/robustness.json").write_text(json.dumps(out, indent=1))
    print("\n写入 results/robustness.json")


if __name__ == "__main__":
    main()
