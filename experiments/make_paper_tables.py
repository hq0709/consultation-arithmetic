"""从结果 jsonl 自动生成论文的 LaTeX 表格与关键数字（避免手抄导致的数字漂移）。"""
from __future__ import annotations
import sys, pathlib, json, glob, collections, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from experiments.grid_files import main_grid
from experiments.analyze import load, wilson, mcnemar
from mechanisms.nmi_metrics import config_metrics, fit_turn_powerlaw, fit_msg_density_log
from panels.architectures import NMI_CLASS

TIER = {"gpt-4.1-nano": "T1", "gpt-5-nano": "T2", "gpt-5-mini": "T3"}
BENCH = {"medxpertqa": "MedXpertQA", "medqa": "MedQA", "medagentsbench": "MedAgentsBench-hard"}
ARCHLAB = {"independent": "Independent", "centralized": "Centralized",
           "discussion": "Decentralized", "tiered": "Hybrid", "debate": "Debate",
           "zeroshot": "SAS zero-shot", "cot": "SAS CoT", "sc": "Self-consistency"}


def main():
    ap = argparse.ArgumentParser()
    # 默认走 main_grid()：以前默认是 "G_*.jsonl"，只匹配最早的 9 个 OpenAI 文件，
    # 所以协调指标表在网格扩到 21 个文件之后仍然只覆盖三个模型（k=45）。
    ap.add_argument("--glob", default=None, help="覆盖主网格，用于对照臂/消融臂")
    ap.add_argument("--outdir", default="paper")
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "results" / a.glob))) if a.glob else main_grid()
    if not files:
        print("没有匹配的结果文件"); return
    rows = load(files)
    print(f"{len(rows)} episodes from {len(files)} files")
    out = pathlib.Path(ROOT / a.outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)

    # ---- Table 2: 主结果 accuracy x tier x bench x arch x N
    recs = []
    for (m, b, arch, N), v in cells.items():
        p, lo, hi = wilson(sum(x["correct"] for x in v), len(v))
        recs.append({"tier": TIER.get(m, m), "model": m, "bench": BENCH.get(b, b),
                     "arch": ARCHLAB.get(arch, arch), "arch_key": arch, "N": N, "n": len(v),
                     "acc": p * 100, "lo": lo * 100, "hi": hi * 100,
                     "usd_1k": sum(x["cost"]["usd"] for x in v) / len(v) * 1000,
                     "samples": sum(x["cost"]["samples"] for x in v) / len(v)})
    df = pd.DataFrame(recs).sort_values(["tier", "bench", "arch", "N"])
    df.to_csv(out / "tables/main_results.csv", index=False)

    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Tier & Architecture & $N$ & Accuracy (\%) & 95\% CI & USD/1k \\", r"\midrule"]
    for bench in df.bench.unique():
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{bench}}}}} \\")
        s = df[df.bench == bench]
        for _, r in s.iterrows():
            lines.append(f"{r.tier} & {r.arch} & {r.N} & {r.acc:.1f} & "
                         f"[{r.lo:.1f}, {r.hi:.1f}] & {r.usd_1k:.2f} \\\\")
        lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables/main_results.tex").write_text("\n".join(lines))

    # ---- 协调指标（NMI Table 5 对应）
    sasc = {k: v for k, v in cells.items() if k[2] == "cot"}
    mrecs = []
    for (m, b, arch, N), v in cells.items():
        if arch in ("cot", "zeroshot"):
            continue
        base = sasc.get((m, b, "cot", 1))
        if not base:
            continue
        mm = config_metrics(v, base)
        if mm:
            mm.update({"tier": TIER.get(m, m), "bench": BENCH.get(b, b), "N": N,
                       "nmi_class": NMI_CLASS.get(arch, arch), "arch": ARCHLAB.get(arch, arch)})
            mrecs.append(mm)
    md = pd.DataFrame(mrecs)
    if not md.empty:
        md.to_csv(out / "tables/coordination_metrics.csv", index=False)
        g = md.groupby("nmi_class").agg(
            n=("n", "size"), acc=("accuracy", "mean"), turns=("turns", "mean"),
            c=("msg_density", "mean"), R=("redundancy", "mean"),
            O=("overhead_pct", "mean"), Ec=("efficiency", "mean"),
            Ae=("error_amp", "mean"), absorb=("absorb", "mean")).round(3)
        tl = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
              r"NMI class & $k$ & Acc & $T$ & $c$ & $R$ & $O\%$ & $\Ec$ & $\Ae$ \\", r"\midrule"]
        for cls, r in g.iterrows():
            tl.append(f"{cls} & {int(r.n)} & {r.acc*100:.1f} & {r.turns:.1f} & {r.c:.3f} & "
                      f"{r.R:.3f} & {r.O:.0f} & {r.Ec:.2f} & {r.Ae:.2f} \\\\")
        tl += [r"\bottomrule", r"\end{tabular}"]
        (out / "tables/coordination_metrics.tex").write_text("\n".join(tl))
        print("\n=== 协调指标（按 NMI 架构类）===")
        print(g.to_string())

        pl = fit_turn_powerlaw([(r.n_agents, r.turns) for r in md.itertuples()])
        ml = fit_msg_density_log([(r.msg_density, r.accuracy) for r in md.itertuples()])
        facts = {"turn_powerlaw": pl, "msg_density_log": ml,
                 "n_configs": int(len(md)), "n_episodes": int(len(rows))}
        (out / "tables/key_facts.json").write_text(json.dumps(facts, indent=1))
        if pl:
            print(f"\n轮数幂律: T = {pl['a']:.2f}(n+0.5)^{pl['exponent']:.3f}, R²={pl['r2']:.3f}"
                  f"   [NMI: 2.72(n+0.5)^1.724, R²=0.974]")
        if ml:
            print(f"消息密度: S = {ml['intercept']:.3f} + {ml['slope']:.3f}ln(c), R²={ml['r2']:.3f}"
                  f"   [NMI: 0.73+0.28ln(c), R²=0.68]")

    print(f"\n表格写入 {out}/tables/")
    print(df.pivot_table(index=["tier", "bench"], columns=["arch", "N"],
                         values="acc").round(1).to_string()[:3000])


if __name__ == "__main__":
    main()
