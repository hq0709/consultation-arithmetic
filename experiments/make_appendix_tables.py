"""C2 —— 补齐 NMI 的 Table 4 / 6 / 7 的等价物。"""
import sys, pathlib, json, glob
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, statsmodels.api as sm
from experiments.grid_files import main_grid
from experiments.scaling_law import build_table
from experiments.robustness import prep
OUT = ROOT / "paper/tables"; OUT.mkdir(parents=True, exist_ok=True)

df = build_table([pathlib.Path(f).name for f in main_grid()])
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
# 三个厂商全部由数据生成。此前只生成 OpenAI 三行、Google 两行是手工补的，
# 一重跑生成器就被覆盖掉了（2026-09-01 发生过一次）。
from experiments.refresh_capability import capability_from_results
CAP, ACC = capability_from_results()
VENDORS = [
    ("\\openai", "OpenAI --- the capability ladder the grid is run on",
     ["gpt-4.1-nano", "gpt-5-nano", "gpt-5-mini"]),
    ("\\gemini", "Google --- a second vendor, for the diversity ladder",
     ["gemini-3.5-flash-lite", "gemini-3.7-flash"]),
    ("\\claude", "Anthropic --- a third vendor, for the diversity ladder",
     ["claude-haiku-4-5-20251001", "claude-sonnet-5"]),
    ("\\deepseek", "DeepSeek --- a fourth vendor in the grid; Alibaba and Zhipu "
                 "contribute single-doctor baselines to the diversity ladder",
     ["deepseek-v4-flash", "deepseek-v4-pro"]),
    ("\\qwen", None, ["qwen3.8-flash", "qwen3.8-max"]),   # 无图标资源，用文字
    ("\\zhipu", None, ["glm-5.3-flash", "glm-5.3"]),
]
TEX = {"claude-haiku-4-5-20251001": r"\texttt{claude-haiku-4.5}",
       "claude-sonnet-5": r"\texttt{claude-sonnet-5}"}
BEN3 = ("medxpertqa", "medagentsbench", "medqa")
rows = [r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.02}",
        r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
        r"Model & MedXpertQA & MedAgentsBench & MedQA & $I$ (mean) & Setting \\",
        r"\midrule"]
for vi, (ven, head, ms) in enumerate(VENDORS):
    ms = [m for m in ms if m in CAP]
    if not ms:
        continue
    # head=None 表示与上一节共用标题（三家中国实验室合成一节），不再重复画标题行
    if head:
        if vi:
            rows.append(r"\midrule")
        rows += [r"\sectrow", rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{head}}}}} \\"]
    for m in ms:
        v = [100 * np.mean(ACC[m][b]) for b in BEN3]
        # 厂商图标前置到模型名，不再单开一列
        rows.append(f"{ven}~{TEX.get(m, chr(92)+'texttt{'+m+'}')} & "
                    f"{v[0]:.1f} & {v[1]:.1f} & {v[2]:.1f} & "
                    f"\\textbf{{{CAP[m]:.1f}}} & "
                    f"{'effort=low' if m.startswith('gpt-5') else '---'} \\\\")
rows += [r"\bottomrule",
         r"\multicolumn{6}{@{}l}{\scriptsize Capability-matched cross-vendor pairs:} \\",
         r"\multicolumn{6}{@{}l}{\scriptsize \texttt{gemini-3.5-flash-lite} ($50.3$) with \texttt{gpt-5-nano} ($50.8$), and} \\",
         r"\multicolumn{6}{@{}l}{\scriptsize \texttt{deepseek-v4-pro} ($59.7$) with \texttt{gpt-5-mini} ($59.2$) across the Pacific.} \\",
         r"\end{tabular}"]
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
