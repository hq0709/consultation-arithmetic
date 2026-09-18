"""MedAgents (Tang et al., 2024) —— 照官方实现复现，不是照论文描述猜。

Prompt 逐字取自 github.com/gersteinlab/MedAgents 的 prompt_generator.py
（存档在 frameworks/_medagents_official_prompts.py），控制流取自同仓库 utils.py
的 fully_decode()，method="syn_verif"（论文的完整版本）。

五步：
  1 领域划分   —— 5 个问题领域 + 2 个选项领域（NUM_QD=5, NUM_OD=2）
  2 各自分析   —— 每个领域一位专家出分析
  3 汇总报告   —— 合成器把 7 份分析并成一份报告
  4 共识循环   —— 每位专家对报告投 YES/NO；任一 NO 则收集修订意见、改写报告、再投
                  终止：全体 YES，或用满 max_attempt_vote=3 轮
  5 最终作答   —— 从报告导出答案

第 4 步是本文关心的对象：它的终止条件是「没人反对」，而不是「报告更正确」。
"""
import sys, pathlib, re, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import chat
from frameworks._medagents_official_prompts import (
    NUM_QD, NUM_OD, get_question_domains_prompt, get_question_analysis_prompt,
    get_options_domains_prompt, get_options_analysis_prompt,
    get_synthesized_report_prompt, get_consensus_prompt,
    get_consensus_opinion_prompt, get_revision_prompt,
    get_final_answer_prompt_wsyn)

MAX_ATTEMPT_VOTE = 3        # run.py 的默认值


def domains_all(qdoms, odoms):
    seen, out = set(), []
    for d in list(qdoms) + list(odoms):
        if d not in seen:
            seen.add(d); out.append(d)
    return out


# 官方的 30/50 token 小预算是给非推理模型定的（原论文用 GPT-3.5/GPT-4）。
# 我们也用非推理模型跑，所以保留官方预算；只给一个很低的下限防止解析失败。
MIN_TOK = 120


def _ask(model, system, user, max_tokens, meter, tag):
    r = chat(model=model, system=system or "You are a helpful assistant.",
             messages=[{"role": "user", "content": user}],
             temperature=0.0, max_tokens=max(max_tokens, MIN_TOK), tag=tag)
    meter["calls"] += 1
    meter["usd"] = meter.get("usd", 0.0)
    return (r.get("text") or "").strip()


def _clean_report(question, options, raw):
    """官方 data_utils.cleansing_syn_report：报告开头必须带 Question + Options。
    漏了这一步，第 5 步的 get_final_answer_prompt_wsyn(report) 就没有选项可选，
    模型会回「请提供选项」—— 99 题里 84 题因此解析为空。"""
    tmp = raw.split("Total Analysis:")
    total = tmp[-1].strip() if len(tmp) > 1 else raw.strip()
    key = tmp[0].split("Key Knowledge:")[-1].strip() if "Key Knowledge" in tmp[0] else ""
    head = f"Question: {question} \nOptions: {options} \n"
    return head + (f"Key Knowledge: {key} \n" if key else "") + f"Total Analysis: {total} \n"


def _fields(raw, n, default="General Medicine"):
    part = raw.split(":")[-1].strip()
    fs = [x.strip() for x in part.split("|") if x.strip()]
    return (fs + [default] * n)[:n]


def run(item, model, meter=None):
    """返回 (pred, trace)。trace 里带共识循环的全过程，供后续测 phi/一致率。"""
    meter = meter if meter is not None else {"calls": 0}
    q = item.get("stem") or item["question"]
    opts = "\n".join(f"({k}) {v}" for k, v in item["options"].items())

    # --- 1 领域划分 ---
    sysm, pr = get_question_domains_prompt(q)
    qdoms = _fields(_ask(model, sysm, pr, 50, meter, "ma_qdom"), NUM_QD)
    sysm, pr = get_options_domains_prompt(q, opts)
    odoms = _fields(_ask(model, sysm, pr, 50, meter, "ma_odom"), NUM_OD)

    # --- 2 各自分析 ---
    qa = {}
    for d in qdoms:
        sysm, pr = get_question_analysis_prompt(q, d)
        qa[d] = _ask(model, sysm, pr, 300, meter, "ma_qana")
    oa = {}
    for d in odoms:
        sysm, pr = get_options_analysis_prompt(q, opts, d, qa)
        oa[d] = _ask(model, sysm, pr, 300, meter, "ma_oana")

    # --- 2b 每位专家的独立答案 ---
    # 官方流程里第 2 步产出的是领域「分析文本」而非选项，所以原实现没有可比的
    # 首轮意见。为了能对 MedAgents 测 phi / oracle / kappa，这里额外让每位专家
    # 在自己的分析基础上给出一个选项。这一步不进入官方控制流，不影响它的输出。
    first = []
    for d in domains_all(qdoms, odoms):
        txt = _ask(model, f"You are a medical expert specialized in the {d} domain.",
                   f"{q}\n\nOptions:\n{opts}\n\nBased on your expertise, respond only with "
                   f"the selected option's letter, in the format 'Option: [Letter]'.",
                   40, meter, "ma_indiv")
        a = None
        for pat in (r"Option\s*[:：]\s*\[?\(?\s*([A-J])\b", r"\(([A-J])\)"):
            a = next((m.group(1) for m in re.finditer(pat, txt)
                      if m.group(1) in item["options"]), None)
            if a: break
        first.append({"agent": d, "answer": a})

    # --- 3 汇总报告 ---
    qtxt = "\n".join(f"The {d} expert reports: {v}" for d, v in qa.items())
    otxt = "\n".join(f"The {d} expert reports: {v}" for d, v in oa.items())
    sysm, pr = get_synthesized_report_prompt(qtxt, otxt)
    report = _clean_report(q, opts, _ask(model, sysm, pr, 2000, meter, "ma_syn"))

    # --- 4 共识循环 ---
    domains = qdoms + odoms
    votes, rounds = [], 0
    hasno = True
    while rounds < MAX_ATTEMPT_VOTE and hasno:
        rounds += 1
        hasno = False
        this, advice = {}, {}
        for d in domains:
            voter, cp = get_consensus_prompt(d, report)
            v = _ask(model, voter, cp, 30, meter, f"ma_vote{rounds}")
            yes = "yes" in v.lower()
            this[d] = "yes" if yes else "no"
            if not yes:
                hasno = True
                advice[d] = _ask(model, voter, get_consensus_opinion_prompt(d, report),
                                 500, meter, f"ma_adv{rounds}")
        votes.append(this)
        if hasno:
            report = _clean_report(q, opts, _ask(model, "", get_revision_prompt(report, advice),
                                                  2000, meter, f"ma_rev{rounds}"))

    # --- 5 最终作答 ---
    ansprompt = get_final_answer_prompt_wsyn(report)
    final = _ask(model, "", ansprompt, 600, meter, "ma_final")
    # 官方 prompt 要求 "Option: X"，优先按它解析。退化到裸字母匹配会把正文里的
    # 大写 I（句首的 "I ..."）当成选项 —— MedXpertQA 有 A-J 十个选项，I 合法，
    # 所以校验拦不住：81 题里 65 题被误解析成 I。
    valid = set(item["options"])
    pred = None
    for pat in (r"Option\s*[:：]\s*\[?\(?\s*([A-J])\b",
                r"\(([A-J])\)",
                r"(?:answer|Answer)\s*(?:is)?\s*[:：]?\s*\(?([A-J])\b"):
        for m in re.finditer(pat, final):
            if m.group(1) in valid:
                pred = m.group(1); break
        if pred: break

    return pred, {"domains": domains, "votes": votes, "n_vote_rounds": rounds,
                  "reached_consensus": not hasno, "first_round": first,
                  "final_text": final[-300:],
                  "calls": meter["calls"]}
