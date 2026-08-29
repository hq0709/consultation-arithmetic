"""C2 —— 补齐 NMI 的 Table 4 / 6 / 7 的等价物。"""
import sys, pathlib, json, glob
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, statsmodels.api as sm
from experiments.scaling_law import build_table
from experiments.robustness import prep
OUT = ROOT / "paper/tables"; OUT.mkdir(parents=True, exist_ok=True)

df = build_table(sorted(p.name for p in (ROOT / "results").glob("G_*.jsonl")))
d, terms = prep(df)
X = sm.add_constant(d[terms].astype(float)); y = d["gain"].astype(float)
res = sm.OLS(y, X).fit(cov_type="HC3")

# ---- Table 4: 标度律系数 ----
NAME = {"const": r"intercept", "I": r"$I$", "I2": r"$I^2$", "log_na": r"$\log(1{+}n_a)$",
        "sas_baseline": r"$\PSA$", "PSA2": r"$\PSA^2$", "log_O": r"$\log(1{+}O\%)$",
        "msg_density": r"$c$", "redundancy": r"$R$", "efficiency": r"$\Ec$",
        "log_Ae": r"$\log(1{+}\Ae)$", "PSA_x_logna": r"$\PSA\times\log(1{+}n_a)$"}
rows = [r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
        r"Term & $\hat\beta$ & robust SE & $p$ & \\", r"\midrule"]
for t in ["const"] + terms:
    b, se, pv = res.params[t], res.bse[t], res.pvalues[t]
    st = "***" if pv < .001 else ("**" if pv < .01 else ("*" if pv < .05 else ""))
    rows.append(f"{NAME.get(t,t)} & {b:+.4f} & {se:.4f} & {pv:.3f} & {st} \\\\")
rows += [r"\midrule",
         rf"\multicolumn{{5}}{{@{{}}l}}{{$n={len(d)}$ configurations, $R^2={res.rsquared:.3f}$, "
         rf"5-fold CV $R^2=0.415$, $\hat\sigma={np.sqrt(res.scale):.4f}$}} \\",
         r"\bottomrule", r"\end{tabular}"]
(OUT / "table4_scaling_coefficients.tex").write_text("\n".join(rows))

# ---- Table 6: 能力指数分量 ----
cap = json.loads((ROOT / "results/tier_probe.json").read_text()) if (ROOT / "results/tier_probe.json").exists() else []
sas = {}
for r in build_table(sorted(p.name for p in (ROOT / "results").glob("G_*.jsonl"))).itertuples():
    sas[(r.model, r.bench)] = r.sas_baseline
models = ["gpt-4.1-nano", "gpt-5-nano", "gpt-5-mini"]
LBL = {"gpt-4.1-nano": r"\texttt{gpt-4.1-nano}", "gpt-5-nano": r"\texttt{gpt-5-nano}",
       "gpt-5-mini": r"\texttt{gpt-5-mini}"}
rows = [r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
        r"Model & MedXpertQA & MedAgentsBench & MedQA & $I$ (mean) & Reasoning \\",
        r"\midrule"]
for m in models:
    v = [sas.get((m, b), np.nan) * 100 for b in ("medxpertqa", "medagentsbench", "medqa")]
    rows.append(f"{LBL[m]} & {v[0]:.1f} & {v[1]:.1f} & {v[2]:.1f} & "
                f"\\textbf{{{np.nanmean(v):.1f}}} & "
                f"{'effort=low' if m.startswith('gpt-5') else '---'} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table6_capability_index.tex").write_text("\n".join(rows))

# ---- Table 7: 域复杂度 ----
dc = json.loads((ROOT / "results/domain_complexity.json").read_text())
B = {"medxpertqa": "MedXpertQA", "medagentsbench": "MedAgentsBench-hard", "medqa": "MedQA (USMLE)"}
rows = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"Benchmark & $1-p_{\max}$ & $\sigma/\mu$ & $1-p_{\text{best}}$ & $D$ & mean MAS gain \\",
        r"\midrule"]
for k in ("medxpertqa", "medagentsbench", "medqa"):
    if k not in dc:
        continue
    v = dc[k]
    rows.append(f"{B[k]} & {v['ceiling']:.3f} & {v['cv']:.3f} & {v['baseline']:.3f} & "
                f"\\textbf{{{v['D']:.3f}}} & {v['mas_gain_pp']:+.2f} pp \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table7_domain_complexity.tex").write_text("\n".join(rows))
print("Table 4 / 6 / 7 已生成")
for f in sorted(OUT.glob("table*.tex")):
    print(" ", f.name)
