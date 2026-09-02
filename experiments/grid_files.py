"""主网格的唯一定义。

以前每个分析脚本各写一遍 glob("results/G_*.jsonl")，有两个后果：
  1. 加新厂商时要改六处，漏一处就出现"论文里两个数字来自不同数据集"；
  2. 通配符会误吞同前缀的对照臂 —— 2026-08-31 通用内科对照以 G_T2_generic.jsonl
     落盘，被六个脚本静默算进主网格，12 个 cell 的准确率被改动最多 1.60pp。
所以主网格改为在这里显式列出，并且只收 27/27 满额的文件。

主网格 = 四家厂商 / 八个模型 / 三个 benchmark，全部 27 cell 同构（temp=0.7）：
  OpenAI     gpt-4.1-nano, gpt-5-nano, gpt-5-mini
  Google     gemini-3.5-flash-lite, gemini-3.7-flash
  Anthropic  claude-haiku-4.5, claude-sonnet-5
  DeepSeek   deepseek-v4-flash

deepseek-v4-pro 只有 MedXpertQA 一个 benchmark 跑满，且缺自洽性最高两档
（k=9/15 上模型推理失控，单次吐 20 万 reasoning token，2026-09-01 终止）。
它以显式豁免进主网格 —— 见 EXEMPT。不放松 27 的通用阈值：那样会让别的
半截文件静默混进来，而这正是 2026-08-31 通用内科对照污染主网格的原因。
"""
import json, glob, pathlib, collections, functools

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PREFIXES = ("G_", "CLA_", "GEM_", "OR_gemini-3.7-flash_medqa",
                 "OR_deepseek-v4-flash_")
EXCLUDE = ("generic", "specialty", "heterogeneity", "pilot", "sanity", "_smoke")

# 显式豁免：文件名 -> 要求的满额 cell 数。只写在这里，不改通用阈值。
EXEMPT = {
    # deepseek-v4-pro 不进主网格：只有 MedXpertQA 跑完了架构×N，另外两个
    # benchmark 停在 18/20 和 2/20（2026-09-02，OpenRouter 余额耗尽）。
    # 一个只有单 benchmark 的模型会在按 benchmark 分列的图里留下残缺的一行。
    # 它仍然参与多样性阶梯 —— 那里按模型对配对，不要求三个 benchmark 同构，
    # 而它正是与 gpt-5-mini 能力匹配的跨太平洋那一对。
}


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
        if _cells_full(p, need=EXEMPT.get(b, 27)):
            out.append(p)
        elif verbose:
            print(f"  [不进主网格] {b}：未满 27 cell")
    return out



@functools.lru_cache(maxsize=None)
def canonical_items(bench):
    """某个 benchmark 的官方 250 题（论文声明的评测集）。"""
    f = ROOT / f"data/{bench}_250.jsonl"
    return frozenset(json.loads(l)["qid"] for l in f.open() if l.strip())


def load_main(dedup=True):
    """主网格的 episode，**只保留官方 250 题**。

    为什么必须过滤：网格是分批跑出来的，早期有几批用 `medxpertqa_500.jsonl` 下的题，
    后来统一到 250 题。250 是 500 的严格前缀，所以数据本身没坏，但 11 个 cell 的均值
    落在 500 题的池子上、其余 cell 落在 250 题的池子上 —— 同一个 model x bench 里
    不同架构就不再是同题比较了（gpt-5-mini/MedXpertQA 的 independent N=1 因此差 4.0pp）。
    论文声明的是每个 benchmark 250 题，分析口径就必须是 250 题。
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.analyze import load
    rows = load(main_grid(), dedup=dedup)
    rows = [r for r in rows if r.get("qid") in canonical_items(r.get("bench"))]

    # 只保留满额的 (model, bench, arch, N)。网格是分批跑的，有些格子跑到一半
    # 就停了 —— deepseek-v4-pro 在 MedAgentsBench 上 discussion N=9 只有 175 题、
    # tiered N=9 一题没有。半截格子混进来，按架构取的均值就会拿 175 题的
    # 结果去和 250 题的比，而缺席的架构还会静默地不参与竞争。
    cnt = collections.Counter()
    for r in rows:
        cnt[(r["model"], r["bench"], r["arch"], r["N"])] += 1
    full = {k for k, v in cnt.items() if v >= 240}

    # 再要求 (model, bench) 的架构×N 覆盖完整。deepseek-v4-pro 的 MedQA 只跑到
    # 2/20、MedAgentsBench 18/20，混进来会让缺席的架构静默地不参与竞争 ——
    # 按架构取的均值就成了「谁跑得多谁说了算」。宁可少一个 cell，不要一个歪的。
    MAS_N = {(a, n) for a in ("independent", "centralized", "discussion", "tiered")
             for n in (1, 3, 5, 7, 9)}
    cov = collections.defaultdict(set)
    for m, b, a, n in full:
        cov[(m, b)].add((a, n))
    complete = {k for k, v in cov.items() if MAS_N <= v}
    return [r for r in rows
            if (r["model"], r["bench"], r["arch"], r["N"]) in full
            and (r["model"], r["bench"]) in complete]

if __name__ == "__main__":
    fs = main_grid(verbose=True)
    import sys
    sys.path.insert(0, str(ROOT))
    rows = load_main()   # 必须和分析走同一条路径，否则这里打印的规模不是论文的规模
    MAS = {"independent", "centralized", "discussion", "tiered", "hybrid"}
    cfg = {(r["model"], r["bench"], r["arch"], r["N"]) for r in rows}
    print(f"\n主网格 {len(fs)} 个文件（过滤后）")
    print(f"  模型 {len({r['model'] for r in rows})} · benchmark {len({r['bench'] for r in rows})}")
    print(f"  {len(rows)} episodes · {len(cfg)} 配置 · 其中多智能体 {len([c for c in cfg if c[2] in MAS])}")
