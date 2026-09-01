"""面板配置下（temp=0.7 + 专科提示）两条路是否给出同一个分布。

cot 探测在 temp=0.3 下拿到 100% 逐题一致，但那是最容易的条件。本文的核心量是
面板成员之间的误差相关性 phi，它在 temp=0.7 下测量；温度越高随机性越大，
两条路即使同模型也不该期待逐题相同。所以这里比的是分布量：准确率与组内 phi。

用 MedXpertQA（直连已 27/27 完整，且模型准确率 60% 左右，有足够的出错空间让
phi 可测），取 independent N=9 的 round-0 意见。
"""
import sys, os, json, pathlib, collections, itertools
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from panels.roles import role_system
from panels.base import render_question
from panels.architectures import parse_opinion
from experiments.analyze import load

N_ITEMS = int(os.environ.get("PROBE_N", "40"))
MODEL_OR = "google/gemini-3.7-flash"


def phi_of(agents):
    """agents: {agent_index: {qid: err01}} -> 两两相关的均值"""
    ph = []
    for i, j in itertools.combinations(sorted(agents), 2):
        ids = sorted(set(agents[i]) & set(agents[j]))
        if len(ids) < 20:
            continue
        x = np.array([agents[i][q] for q in ids], float)
        y = np.array([agents[j][q] for q in ids], float)
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        ph.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(ph)) if ph else float("nan")


def main():
    rows = [r for r in load([str(ROOT / "results/GEM_gemini-3.7-flash_medxpertqa.jsonl")])
            if r.get("arch") == "independent" and r.get("N") == 9]
    rows = rows[:N_ITEMS]
    items = {json.loads(l)["qid"]: json.loads(l)
             for l in open(ROOT / "data/medxpertqa_250.jsonl")}

    # 直连一侧：直接从已跑数据里取 round-0 的九个意见
    dx = collections.defaultdict(dict)
    dx_corr = []
    for r in rows:
        for i, a in enumerate(r["rounds"][0]):
            dx[i][r["qid"]] = int(a.get("answer") != r["gold"])
        dx_corr.append(r["correct"])

    cli = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                 base_url="https://openrouter.ai/api/v1")

    def ask(args):
        qid, idx, spec = args
        it = items[qid]
        u = (render_question(it) + "\n\nGive your independent opinion. Reply with JSON only: "
             '{"answer": "<option letter>", "confidence": <0-100>, '
             '"reason": "<at most 60 words>"}')
        r = cli.chat.completions.create(
            model=MODEL_OR, max_tokens=800, temperature=0.7,
            messages=[{"role": "system", "content": role_system(spec)},
                      {"role": "user", "content": u}])
        o = parse_opinion(r.choices[0].message.content or "", list(it["options"]), "probe")
        cost = getattr(r.usage, "cost", 0) or 0
        return qid, idx, o.answer, cost

    jobs = []
    for r in rows:
        for i, a in enumerate(r["rounds"][0]):
            spec = a["agent"].rsplit("#", 1)[0]
            jobs.append((r["qid"], i, spec))
    orr = collections.defaultdict(dict)
    votes = collections.defaultdict(list)
    total = 0.0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for qid, idx, ans, c in ex.map(ask, jobs):
            gold = next(x["gold"] for x in rows if x["qid"] == qid)
            orr[idx][qid] = int(ans != gold)
            votes[qid].append(ans)
            total += c

    or_acc = np.mean([collections.Counter(v).most_common(1)[0][0]
                      == next(x["gold"] for x in rows if x["qid"] == q)
                      for q, v in votes.items()])
    print("=" * 74)
    print(f"面板配置对比 (temp=0.7, 专科提示) —— MedXpertQA, {len(rows)} 题 x 9 个 agent")
    print("=" * 74)
    print(f"  直连       多数票准确率 {100*np.mean(dx_corr):5.1f}%   组内 phi = {phi_of(dx):.3f}")
    print(f"  OpenRouter 多数票准确率 {100*or_acc:5.1f}%   组内 phi = {phi_of(orr):.3f}")
    print(f"  本次探测花费 ${total:.4f}")


if __name__ == "__main__":
    main()
