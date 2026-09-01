"""OpenRouter 路由的 gemini-3.7-flash 与直连 Google API 是否可互换。

背景：直连的每日配额把 gemini-3.7-flash 的 MedQA 网格卡在 5/27。OpenRouter 有同名模型，
问题是能不能把两条路的数据混进同一个网格。同名不等于同配置：OpenRouter 那一路的
思考(reasoning)是强制开启、无法关闭的（reasoning.enabled=false 直接 400）。

判据是逐题一致率，不是总体准确率——两条路各自 95% 但错在不同题上，说明是两个不同的
采样分布，混用会污染误差相关性 phi（本文的核心量）。
"""
import sys, os, json, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from openai import OpenAI
from panels.roles import role_system
from panels.base import render_question
from panels.architectures import parse_opinion
from experiments.analyze import load

N_ITEMS = int(os.environ.get("PROBE_N", "60"))
MODEL_OR = "google/gemini-3.7-flash"
MODEL_DIRECT = "gemini-3.7-flash"


def main():
    items = [json.loads(l) for l in open(ROOT / "data/medqa_250.jsonl")][:N_ITEMS]
    direct = {r["qid"]: r for r in load([str(p) for p in (ROOT / "results").glob("*gemini-3.7-flash_medqa.jsonl")])
              if r.get("arch") == "cot" and r.get("N") == 1}
    cli = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                 base_url="https://openrouter.ai/api/v1")
    agree = both = 0
    or_correct = dx_correct = 0
    cost = 0.0
    rsn = []
    for it in items:
        qid = it["qid"]
        if qid not in direct:
            continue
        u = (render_question(it) + "\n\nThink through the case step by step (briefly), then give "
             'your final answer. Reply with JSON only: {"answer": "<option letter>", '
             '"confidence": <0-100>, "reason": "<your step-by-step reasoning, at most 150 words>"}')
        r = cli.chat.completions.create(
            model=MODEL_OR, max_tokens=800, temperature=0.3,
            messages=[{"role": "system", "content": role_system("internal medicine", generic=True)},
                      {"role": "user", "content": u}])
        txt = r.choices[0].message.content or ""
        o = parse_opinion(txt, list(it["options"]), "probe")
        ctd = getattr(r.usage, "completion_tokens_details", None)
        rsn.append(getattr(ctd, "reasoning_tokens", 0) or 0 if ctd else 0)
        cost += (getattr(r, "usage", None) and getattr(r.usage, "cost", 0)) or 0
        d = direct[qid]
        both += 1
        agree += int(o.answer == d["rounds"][0][0].get("answer"))
        or_correct += int(o.answer == d["gold"])
        dx_correct += d["correct"]
    print("=" * 70)
    print(f"OpenRouter vs 直连  ——  gemini-3.7-flash / MedQA / 单医生 CoT，同一批 {both} 题")
    print("=" * 70)
    print(f"  直连准确率        {100*dx_correct/both:.1f}%")
    print(f"  OpenRouter 准确率 {100*or_correct/both:.1f}%")
    print(f"  逐题答案一致率    {100*agree/both:.1f}%   <- 这才是能否混用的判据")
    print(f"  OpenRouter 思考 token  中位 {int(np.median(rsn))}  最大 {max(rsn)}"
          f"   (直连记录为 0)")
    print(f"  本次探测花费 ${cost:.4f}")


if __name__ == "__main__":
    main()
