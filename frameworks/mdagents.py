"""MDAgents (Kim et al., 2024) —— 照官方实现复现。

Prompt 与控制流取自 github.com/mitmedialab/MDAgents 的 utils.py
（determine_difficulty / process_basic_query / process_intermediate_query）。

流程：
  1 复杂度分诊 —— 一个 agent 把题分成 basic / intermediate / advanced
  2 basic      —— 单个 agent 直接作答
    intermediate —— 招募 5 名专家 → 各自初评 → 多轮自由对话 → 各自终评 → 多数票
  3 早停       —— 当一轮里没有任何 agent 表示还想发言时停止

论文关心的是第 3 条：循环的终止条件是「没人还想说话」，不是「答案更正确」。
官方默认 5 轮 × 5 turn；我们按论文报告的规模保留轮数，turn 降到 2（成本 ~4 倍差），
这是与原实现的唯一偏离，在正文中写明。
"""
import sys, pathlib, re, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import chat

NUM_AGENTS = 5
NUM_ROUNDS = 5
NUM_TURNS = 2        # 官方 5；见模块注释

DIFF_SYS = ("You are a medical expert who conducts initial assessment and your job is to "
            "decide the difficulty/complexity of the medical query.")
DIFF_TMPL = ("Now, given the medical query as below, you need to decide the difficulty/complexity "
             "of it:\n{q}.\n\nPlease indicate the difficulty/complexity of the medical query among "
             "below options:\n1) basic: a single medical agent can output an answer.\n"
             "2) intermediate: number of medical experts with different expertise should dicuss "
             "and make final decision.\n3) advanced: multiple teams of clinicians from different "
             "departments need to collaborate with each other to make final decision.")
RECRUIT_SYS = ("You are an experienced medical expert who recruits a group of experts with diverse "
               "identity and ask them to discuss and solve the given medical query.")
RECRUIT_TMPL = ("Question: {q}\nYou can recruit {n} experts in different medical expertise. "
                "Considering the medical question and the options for the answer, what kind of "
                "experts will you recruit to better make an accurate answer?\n\n"
                "Please answer in the format '1. Pediatrician - Specializes in ...' one per line, "
                "and do not include your reason.")


# 官方的 30/50 token 小预算是给非推理模型定的（原论文用 GPT-3.5/GPT-4）。
# 我们也用非推理模型跑，所以保留官方预算；只给一个很低的下限防止解析失败。
MIN_TOK = 120


def _ask(model, system, user, max_tokens, meter, tag):
    r = chat(model=model, system=system or "You are a helpful medical agent.",
             messages=[{"role": "user", "content": user}],
             temperature=0.0, max_tokens=max(max_tokens, MIN_TOK), tag=tag)
    meter["calls"] += 1
    meter["usd"] = meter.get("usd", 0.0)
    return (r.get("text") or "").strip()


def _letter(txt, valid):
    """按显式格式解析。裸字符扫描会把正文里的大写 I / A 当成选项 ——
    MedXpertQA 有 A-J 十个选项，这些字母都合法，所以「在 valid 里」拦不住。"""
    txt = txt or ""
    for pat in (r"Answer\s*[:：]\s*\(?\[?\s*([A-J])\b",
                r"\(([A-J])\)",
                r"Option\s*[:：]\s*\[?\(?\s*([A-J])\b",
                r"(?:answer|Answer)\s*(?:is)?\s*[:：]?\s*\(?([A-J])\b",
                r"^\s*([A-J])\s*[.):]"):
        for m in re.finditer(pat, txt):
            if m.group(1) in valid:
                return m.group(1)
    return None


def run(item, model, meter=None):
    meter = meter if meter is not None else {"calls": 0}
    q = item.get("stem") or item["question"]
    valid = set(item["options"])
    opts = "\n".join(f"({k}) {v}" for k, v in item["options"].items())
    qfull = f"{q}\n{opts}"

    # --- 1 复杂度分诊 ---
    d = _ask(model, DIFF_SYS, DIFF_TMPL.format(q=qfull), 120, meter, "md_diff").lower()
    if "basic" in d or "1)" in d: level = "basic"
    elif "advanced" in d or "3)" in d: level = "advanced"
    else: level = "intermediate"

    # --- 2a basic：单人 ---
    if level == "basic":
        txt = _ask(model, "You are a helpful medical agent.",
                   f"Question: {qfull}\n\nAnswer: ", 500, meter, "md_basic")
        return _letter(txt, valid), {"level": level, "n_agents": 1, "rounds": 0,
                                     "opinions": [], "calls": meter["calls"]}

    # --- 2b intermediate / advanced：招募 → 初评 → 对话 → 终评 → 多数票 ---
    rec = _ask(model, RECRUIT_SYS, RECRUIT_TMPL.format(q=qfull, n=NUM_AGENTS),
               400, meter, "md_recruit")
    roles = []
    for line in rec.split("\n"):
        m = re.match(r'\s*\d+\.\s*([^-\n]+?)\s*[-–]', line)
        if m: roles.append(m.group(1).strip())
    roles = (roles or ["general practitioner"] * NUM_AGENTS)[:NUM_AGENTS]
    while len(roles) < NUM_AGENTS: roles.append("general practitioner")

    def sysmsg(r):
        return f"You are a {r}. Your job is to collaborate with other medical experts in a team."

    # 初评（= 我们的 round-0，供 phi / oracle 测量）
    first = {}
    for r in roles:
        t = _ask(model, sysmsg(r), f"Question: {qfull}\n\nYour answer should be like below "
                 f"format.\n\nAnswer: ", 400, meter, "md_init")
        first[r] = (_letter(t, valid), t)

    assessment = "".join(f"({r}): {v[1][:400]}\n" for r, v in first.items())
    cur = dict(first)
    rounds_run = 0
    for n in range(NUM_ROUNDS):
        rounds_run = n + 1
        num_yes = 0
        for r in roles:
            want = _ask(model, sysmsg(r), "Given the opinions from other medical experts in your "
                        f"team, please indicate whether you want to talk to any expert (yes/no)\n\n"
                        f"Opinions:\n{assessment}", 20, meter, f"md_part{n}")
            if "yes" in want.lower(): num_yes += 1
        if num_yes == 0:
            break                                   # 官方早停：没人还想发言
        newops = {}
        for r in roles:
            t = _ask(model, sysmsg(r), "Now that you've interacted with other medical experts, "
                     "remind your expertise and the comments from other experts and make your "
                     f"final answer to the given question:\n{qfull}\n"
                     "Respond with the option letter only, as 'Answer: X'.\nAnswer: ", 400,
                     meter, f"md_fin{n}")
            newops[r] = (_letter(t, valid), t)
        cur = newops
        assessment = "".join(f"({r}): {v[1][:400]}\n" for r, v in cur.items())

    votes = [v[0] for v in cur.values() if v[0]]
    pred = collections.Counter(votes).most_common(1)[0][0] if votes else None
    return pred, {"level": level, "n_agents": len(roles), "rounds": rounds_run,
                  "roles": roles,
                  "first_round": [{"agent": r, "answer": v[0], "raw": v[1][:200]}
                                  for r, v in first.items()],
                  "final_round": [{"agent": r, "answer": v[0], "raw": v[1][:200]}
                                  for r, v in cur.items()],
                  "calls": meter["calls"]}
