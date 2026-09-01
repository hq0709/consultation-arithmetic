"""主网格的唯一定义。

以前每个分析脚本各写一遍 glob("results/G_*.jsonl")，有两个后果：
  1. 加新厂商时要改六处，漏一处就出现"论文里两个数字来自不同数据集"；
  2. 通配符会误吞同前缀的对照臂 —— 2026-08-31 通用内科对照以 G_T2_generic.jsonl
     落盘，被六个脚本静默算进主网格，12 个 cell 的准确率被改动最多 1.60pp。
所以主网格改为在这里显式列出，并且只收 27/27 满额的文件。

主网格 = 三家厂商 / 七个模型 / 三个 benchmark，全部 27 cell 同构（temp=0.7）：
  OpenAI     gpt-4.1-nano, gpt-5-nano, gpt-5-mini
  Google     gemini-3.5-flash-lite, gemini-3.7-flash
  Anthropic  claude-haiku-4.5, claude-sonnet-5
"""
import json, glob, pathlib, collections

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PREFIXES = ("G_", "CLA_", "GEM_", "OR_gemini-3.7-flash_medqa")
EXCLUDE = ("generic", "specialty", "heterogeneity", "pilot", "sanity", "_smoke")


def _cells_full(path, need=27, need_items=240):
    cells = collections.defaultdict(set)
    for l in open(path):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("status") != "ok":
            continue
        cells[(r.get("arch"), r.get("N"))].add(r.get("qid"))
    return sum(1 for q in cells.values() if len(q) >= need_items) >= need


def main_grid(verbose=False):
    """27/27 满额的主网格文件。不满额的一律不进 —— 半截网格会让某些 cell
    用 60 题的均值去和 250 题的比。"""
    out = []
    for p in sorted(glob.glob(str(ROOT / "results/*.jsonl"))):
        b = pathlib.Path(p).name
        if not b.startswith(MAIN_PREFIXES) or any(x in b for x in EXCLUDE):
            continue
        if _cells_full(p):
            out.append(p)
        elif verbose:
            print(f"  [不进主网格] {b}：未满 27 cell")
    return out


if __name__ == "__main__":
    fs = main_grid(verbose=True)
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.analyze import load
    rows = load(fs)
    MAS = {"independent", "centralized", "discussion", "tiered", "hybrid"}
    cfg = {(r["model"], r["bench"], r["arch"], r["N"]) for r in rows}
    print(f"\n主网格 {len(fs)} 个文件")
    print(f"  模型 {len({r['model'] for r in rows})} · benchmark {len({r['bench'] for r in rows})}")
    print(f"  {len(rows)} episodes · {len(cfg)} 配置 · 其中多智能体 {len([c for c in cfg if c[2] in MAS])}")
