"""Table 1（主结果）。以前这张表是手写死的 .tex，没有生成者 —— 网格从 3 个模型
扩到 9 个之后它一行没动，读者看到的主结果表只有 OpenAI 的三层。

现在按厂商分节、逐模型一行：单医生基线，四种架构各自最好的 (准确率 / n_a)，
以及最好架构相对单医生的增益。
"""
import sys, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.grid_files import load_main
from experiments.vizstyle import MODEL_ORDER, SHORT_MODEL, FAMILY, FAMILY_ORDER

BENCH = [("medxpertqa", "MedXpertQA", "expert-level reasoning, 10 options"),
         ("medagentsbench", "MedAgentsBench-hard", "items where strong models still fail"),
         ("medqa", "MedQA (USMLE)", "licensing-exam questions, 4 options")]
ARCH = [("independent", "Independ."), ("centralized", "Central."),
        ("discussion", "Decentr."), ("tiered", "Hybrid")]
MARK = {"openai": r"\openai", "google": r"\gemini",
        "anthropic": r"\claude", "deepseek": r"\deepseek"}


def main():
    rows = load_main()
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r["arch"], r["N"])].append(r)
    acc = {k: float(np.mean([x["correct"] for x in v])) * 100 for k, v in cells.items()}

    # 行格式与 Table 8（能力指数）一致：logo 内联在完整模型名之前，厂商用分节行分组。
    NAME = {"claude-haiku-4-5-20251001": "claude-haiku-4.5"}
    L = [r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.02}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"& Single & \multicolumn{4}{c}{Best multi-agent variant\ \ (accuracy \% / $n_a$)} & Gain \\",
         r"\cmidrule(lr){3-6}",
         r"Model & doctor & " + " & ".join(l for _, l in ARCH) + r" & best \\"]

    for bk, blab, bnote in BENCH:
        L += [r"\midrule", r"\sectrow",
              rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{blab}}} --- {bnote}}} \\"]
        for fam in FAMILY_ORDER:
            for m in [x for x in MODEL_ORDER if x in FAMILY[fam]["models"]]:
                base = acc.get((m, bk, "cot", 1))
                if base is None:
                    continue
                best, cellstr = None, []
                for ak, _ in ARCH:
                    cand = [(acc[(m, bk, ak, n)], n) for n in (1, 3, 5, 7, 9)
                            if (m, bk, ak, n) in acc]
                    if not cand:
                        cellstr.append("---"); continue
                    a, n = max(cand)
                    cellstr.append((a, n))
                    if best is None or a > best[0]:
                        best = (a, ak)
                vals = [c[0] for c in cellstr if c != "---"]
                top = max(vals) if vals else None
                txt = []
                for c in cellstr:
                    if c == "---":
                        txt.append("---"); continue
                    a, n = c
                    txt.append(rf"\textbf{{{a:.1f}}}\,/\,{n}" if a == top
                               else f"{a:.1f}\\,/\\,{n}")
                g = best[0] - base
                gs = (rf"\gp{{+{g:.1f}}}" if g >= 0 else rf"\gn{{{g:.1f}}}")
                if abs(g) >= 5:
                    gs = (rf"\gp{{\textbf{{+{g:.1f}}}}}" if g >= 0
                          else rf"\gn{{\textbf{{{g:.1f}}}}}")
                name = NAME.get(m, m).replace("_", r"\_")
                L.append(f"{MARK[fam]}~\\texttt{{{name}}} & {base:.1f} & "
                         + " & ".join(txt) + f" & {gs} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    p = ROOT / "paper/tables/main_results_compact.tex"
    p.write_text("\n".join(L) + "\n")
    print(f"写入 {p.relative_to(ROOT)}  ({sum(1 for x in L if x.startswith(chr(92)+'texttt') or ' & \\\\texttt{' in x)} 行模型)")


if __name__ == "__main__":
    main()
