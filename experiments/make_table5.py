"""从 results/diversity_ladder.json 生成论文 Table 5。

这张表此前是手写内联的。同类的手填数字已经出过三次漂移（vizstyle 的能力指数、
正文与附录对 I 的两种定义、这张表的 n=36 停留在加入 Anthropic 之前），所以改成生成式，
与附录 table6/table7 的 \\input 结构一致。
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

LABEL = {
    "One model, specialty prompts": "One model, different specialty prompts",
    "Distinct models, one family": "Different checkpoints, one training family",
    "Distinct families, one vendor": "Different families, one vendor",
}


def main():
    rows = json.loads((ROOT / "results/diversity_ladder.json").read_text())["ladder"]
    out = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
           r"Source of diversity & $n$ & $\varphi$ & $\varphi_{\max}$ & "
           r"$\varphi/\varphi_{\max}$ & $N_{\text{eff}}^{9}$ \\",
           r"\midrule", r"\sectrow"]
    for r in rows:
        src = r["source"]
        if src == "fully independent":
            continue
        cross = src.startswith("Distinct vendors")
        lab = LABEL.get(src, src)
        if cross:
            lab = lab.replace("Distinct vendors", r"Different \emph{vendors}")
            out.append(r"\winrow")
        n = "---" if r["n"] is None else str(r["n"])
        bold = (lambda v, d: rf"\textbf{{{v:.{d}f}}}") if cross else (lambda v, d: f"{v:.{d}f}")
        out.append(f"{lab} & {n} & {r['phi']:.3f} & {r['phi_max']:.3f} & "
                   f"{bold(r['phi_norm'], 3)} & {bold(r['neff9'], 2)} \\\\")
    out += [r"\midrule",
            r"\textit{fully independent} & --- & 0 & 1 & 0 & 9.00 \\",
            r"\bottomrule", r"\end{tabular}"]
    p = ROOT / "paper/tables/table5_diversity.tex"
    p.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n写入 {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
