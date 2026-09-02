"""The four consultation architectures + single-doctor baselines.

Cost-critical invariant (NESTED CALL SHARING): every agent's round-0 opinion is keyed on
(item, roster[i], seed_base, i) ONLY -- never on N or on the architecture. So the round-0
calls of a panel of size N are a superset of those for N' < N, and architectures (1)(2)(3)(4)
all reuse the same round-0 calls from cache. Do not put N or arch into a round-0 cache tag.
"""
from __future__ import annotations
import json, re, collections, random, threading
from dataclasses import dataclass, field

from common.llm import chat, PRICING, is_reasoning
from panels.base import (Opinion, user_prompt, render_question, parse_opinion, majority,
                         conf_weighted, has_majority, entropy, unanimous, ANSWER_JSON)
from panels.roles import route, role_system

# Claude 5 系列自适应思考常开，思考与输出共用 max_tokens：MedXpertQA 的长题目下 400 会被
# 思考吃光、返回空串（2026-08-31 实测 sonnet-5 有 28/250 空输出，out_tok 正好顶到 400）。
# haiku-4.5 不属于该系列，实测空答率 0.06--1.31%，维持 400 以保住已付费的缓存。
MAX_TOK_OPINION = 400
MAX_TOK_THINKING = 2000
THINKING_MODELS = ("claude-sonnet-5", "claude-opus-5", "claude-fable-5", "gemini-3.1-pro")


def _thinks(model: str) -> bool:
    """思考与输出共用 max_tokens 的模型，必须给更大的预算，否则返回空串。

    走 OpenRouter 的一律算在内：那条路把思考 token 计入 max_tokens，而直连
    Google 不计（实测 gemini-3.7-flash 直连每样本可见输出 57 tok，
    经 OpenRouter 同一提示 out=251 其中思考 210）。同样的 400 在两条路上
    语义不同，按路由判定才不会踩空输出。
    """
    from common.llm import is_openrouter
    return model.startswith(THINKING_MODELS) or is_openrouter(model)


# 逐模型的思考预算。2000 对 glm / qwen 这类不够：实测 glm-5.3-flash 单次思考
# 就用到 1420 token，可见输出只剩 98，再多一点就被挤没变成空串。
# 这几个模型是新加的、没有既有缓存，调大不损失任何东西；
# 已在跑或已跑完的模型不动，否则会作废它们的缓存与进度。
THINK_BUDGET = {
    # max_tokens 是封顶不是预付：不用不计费。所以在模型允许的范围内往高了给，
    # 让思考永远挤不掉可见输出 —— 一次截断要重试，等于时间和钱双输。
    # 各模型最大输出上限实测：deepseek 384k、qwen/glm 131k，32000 远在其内，
    # 且是实测最大思考量(3781)的 8 倍。
    "glm-5.3": 32000, "glm-5.3-flash": 32000,
    "qwen3.8-flash": 32000, "qwen3.8-max": 32000,
    "deepseek-v4-flash": 32000,
    # deepseek-v4-pro 的思考量是重尾的：均值 8.2k，p95 33.7k，实测最大 225,919。
    # 32000 会截掉约 5% 的调用，每一次截断都要重试且大概率再截 —— 2026-09-01
    # 的全网格就是这样卡在最后两格上的。max_tokens 是封顶不是预付，用不到不计费，
    # 所以直接开到覆盖整条尾巴（模型允许 384k 输出）。
    "deepseek-v4-pro": 262144,
}


def opinion_budget(model: str) -> int:
    if model in THINK_BUDGET:
        return THINK_BUDGET[model]
    return MAX_TOK_THINKING if _thinks(model) else MAX_TOK_OPINION


def baseline_budget(model: str, default: int) -> int:
    """零样本 / CoT 基线的预算。

    这两条原来直接写 MAX_TOK_THINKING，绕过了 THINK_BUDGET —— 结果 qwen3.8-flash
    每次都从 2000 起步、被思考顶满、再加倍重试，一条数据付两三次钱。
    基线和面板意见必须走同一套预算表。
    """
    if model in THINK_BUDGET:
        return THINK_BUDGET[model]
    return MAX_TOK_THINKING if _thinks(model) else default


def round_budget(model: str) -> int:
    if model in THINK_BUDGET:
        return THINK_BUDGET[model]
    return MAX_TOK_THINKING if _thinks(model) else MAX_TOK_ROUND


MAX_TOK_ROUND = 400
SC_CHUNK = 8            # OpenAI hard limit: n <= 8 per request
SC_POOL = 40            # self-consistency pool = 5 chunks; any k <= pool is a prefix (cache-shared)
DISCUSSION_ROUNDS = 3
DEBATE_ROUNDS = 2
DEFAULT_THETA = 70.0    # tiered-referral confidence threshold (calibrated on dev slice)


class Meter:
    """Per-episode NOMINAL cost/token accounting (counts cached calls at list price too --
    the paper's economics axis must not be discounted by our cache)."""

    def __init__(self):
        self.calls = 0; self.samples = 0
        self.inp = 0; self.out = 0; self.rsn = 0; self.usd = 0.0
        self._lock = threading.Lock()          # R1 #8: pmap calls add() from many threads

    def add(self, r, n_charged=None, out_frac=1.0):
        n = len(r["texts"]) if n_charged is None else n_charged
        inp = r["input_tokens"]; out = int(round(r["output_tokens"] * out_frac))
        pi, po = PRICING.get(r["model"], (0.0, 0.0))
        with self._lock:
            self.calls += 1; self.samples += n
            self.inp += inp; self.out += out; self.rsn += r.get("reasoning_tokens", 0)
            self.usd += inp / 1e6 * pi + out / 1e6 * po

    def asdict(self):
        return {"calls": self.calls, "samples": self.samples, "in_tok": self.inp,
                "out_tok": self.out, "rsn_tok": self.rsn, "usd": round(self.usd, 6)}


def _temp(model, t):
    return None if is_reasoning(model) else t


def _ask(meter, model, system, user, seed, temp, effort, tag, max_tokens=MAX_TOK_OPINION,
         sbase=None):
    # R1 #4, round 2. Reasoning models reject the `seed` argument, so `_ask` nulls it BEFORE it
    # reaches the cache key -- which silently made every extra seed a cached clone of seed 1
    # (verified: 1000/1000 identical predictions across seeds 1-3 on gpt-5-nano). Encode the
    # seed in the tag instead. Only for sbase != 1, so the existing seed-1 cache stays valid.
    if is_reasoning(model) and sbase not in (None, 1):
        tag = f"{tag}|s{sbase}"
    r = chat(model, [{"role": "user", "content": user}], system=system,
             temperature=_temp(model, temp), seed=None if is_reasoning(model) else seed,
             max_tokens=max_tokens, effort=effort, json_mode=True, tag=tag)
    meter.add(r)
    return r["text"]


# ---------------------------------------------------------------- round-0 opinions (shared)
def agent_model(cfg, i):
    """异质性 panel：cfg['models'] 给出每个 agent 位次的模型；否则全用 cfg['model']。
    cfg['orch_model'] 单独指定 Centralized/Hybrid 的 orchestrator（NMI Fig 4 的关键操纵）。"""
    ms = cfg.get("models")
    return ms[i % len(ms)] if ms else cfg["model"]


def orch_model(cfg):
    return cfg.get("orch_model") or cfg["model"]


def round0_opinions(item, roster, n, model, seed_base, temp, effort, meter,
                    generic_roles=False, cfg=None) -> list[Opinion]:
    valid = list(item["options"])
    up = user_prompt(item)

    def one(i):
        sysmsg = role_system(roster[i], generic=generic_roles)
        m_i = agent_model(cfg, i) if cfg else model
        # 异质 panel 的 agent 模型不同 -> tag 必须带模型，否则会串到同质 panel 的缓存
        tg = ("op0g" if generic_roles else "op0") + ("" if m_i == model else f"|{m_i}")
        # 通用角色下每个 agent 的系统提示与用户提示完全相同，agent 索引只体现在 seed 上；
        # 而推理模型会在进缓存键之前丢掉 seed（见 _ask），于是 N 个 agent 命中同一条缓存、
        # 返回同一个答案，"零多样性"就成了构造出来的假象。把索引写进 tag。
        # 专科角色不需要这一条（roster[i] 不同 -> 系统提示不同 -> 缓存键已不同），
        # 因此保持原 tag 以复用既有缓存。
        if generic_roles:
            tg = f"{tg}|a{i}"
        txt = _ask(meter, m_i, sysmsg, up, seed_base * 100 + i, temp, effort,
                   tag=tg, sbase=seed_base, max_tokens=opinion_budget(m_i))
        return parse_opinion(txt, valid, agent=f"{'generalist' if generic_roles else roster[i]}#{i}", rnd=0)

    from common.llm import pmap
    return pmap(one, list(range(n)))


PEER_CHAR_BUDGET = 900   # R1 #11: total peer context is FIXED, independent of N, so the N
                         # effect cannot be read as a context-length effect.


def _peer_order(qid: str, rnd: int, n: int) -> list[int]:
    """R1 #12: rotate the order peers are listed in, per (item, round), so 'first speaker'
    is decorrelated from 'top-ranked specialty'. Deterministic given the item."""
    idx = list(range(n))
    random.Random(f"{qid}|{rnd}").shuffle(idx)
    return idx


def _peer_block(ops, order=None, budget=PEER_CHAR_BUDGET):
    order = order if order is not None else list(range(len(ops)))
    per = max(50, budget // max(1, len(order)))          # balanced truncation across peers
    lines = []
    for i in order:
        o = ops[i]
        lines.append(f"- {o.agent}: answer ({o.answer}), confidence {int(o.confidence)}. "
                     f"{o.reason[:per]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- (1) Independent panel
def arch_independent(item, cfg, meter, roster=None):
    valid = list(item["options"])
    roster = roster or route(item, cfg.get("use_router", True))
    ops = round0_opinions(item, roster, cfg["N"], cfg["model"], cfg["seed"], cfg["temp"],
                          cfg.get("effort"), meter, cfg.get("generic_roles", False), cfg=cfg)
    pred, tie = majority(ops, valid)
    return {"pred": pred, "pred_cw": conf_weighted(ops, valid), "tie": tie,
            "rounds": [[o.__dict__ for o in ops]], "n_rounds": 1,
            "entropy0": entropy(ops), "roster": roster[:cfg["N"]]}


# ---------------------------------------------------------------- (2) Panel discussion
def arch_discussion(item, cfg, meter, roster=None, ops0=None, max_rounds=DISCUSSION_ROUNDS):
    valid = list(item["options"])
    roster = roster or route(item, cfg.get("use_router", True))
    ops = ops0 or round0_opinions(item, roster, cfg["N"], cfg["model"], cfg["seed"],
                                  cfg["temp"], cfg.get("effort"), meter,
                                  cfg.get("generic_roles", False), cfg=cfg)
    hist = [[o.__dict__ for o in ops]]
    orders = []
    q = render_question(item)
    from common.llm import pmap
    for rnd in range(1, max_rounds + 1):
        if unanimous(ops, valid):        # R1 #10: true unanimity only
            break
        order = _peer_order(item["qid"], rnd, len(ops))
        orders.append(order)
        peers = _peer_block(ops, order)
        cur = ops

        def one(i):
            o = cur[i]
            u = (f"{q}\n\nThe panel's current opinions (round {rnd - 1}):\n{peers}\n\n"
                 f"You previously answered ({o.answer}). Weigh your colleagues' reasoning. "
                 f"State your answer for round {rnd} -- keep it or revise it, whichever the "
                 f"evidence supports.\n\n{ANSWER_JSON}")
            m_i = agent_model(cfg, i)
            txt = _ask(meter, m_i, role_system(roster[i], cfg.get("generic_roles", False)),
                       u, cfg["seed"] * 100 + i, cfg["temp"], cfg.get("effort"),
                       tag=f"disc{rnd}n{cfg['N']}v2" + ("" if m_i == cfg["model"] else f"|{m_i}"),
                       max_tokens=round_budget(cfg["model"]), sbase=cfg["seed"])
            return parse_opinion(txt, valid, agent=cur[i].agent, rnd=rnd)

        ops = pmap(one, list(range(cfg["N"])))
        hist.append([o.__dict__ for o in ops])
    pred, tie = majority(ops, valid)
    return {"pred": pred, "pred_cw": conf_weighted(ops, valid), "tie": tie,
            "rounds": hist, "n_rounds": len(hist), "peer_orders": orders,
            "entropy0": entropy([Opinion(**d) for d in hist[0]]), "entropy_final": entropy(ops),
            "roster": roster[:cfg["N"]]}


# ---------------------------------------------------------------- (3) Tiered referral
def arch_tiered(item, cfg, meter, roster=None, theta=None):
    theta = cfg.get("theta", DEFAULT_THETA) if theta is None else theta
    valid = list(item["options"])
    roster = roster or route(item, cfg.get("use_router", True))
    # 分诊的全科医生：漏传 max_tokens 会吃默认的 400，推理模型的思考直接把它吃光
    # 而输出为空 —— 2026-09-02 deepseek-v4-pro 在这里触发了 98 次加倍重试。
    gen_txt = _ask(meter, cfg["model"], role_system("internal medicine", generic=True),
                   user_prompt(item), cfg["seed"] * 100 + 90, 0.3, cfg.get("effort"), tag="gen",
                   max_tokens=opinion_budget(cfg["model"]), sbase=cfg["seed"])
    gen = parse_opinion(gen_txt, valid, agent="generalist", rnd=0)
    out = {"generalist": gen.__dict__, "referred": False, "escalated": False,
           "theta": theta, "roster": roster[:cfg["N"]], "n_rounds": 1}
    if gen.confidence >= theta:
        out.update({"pred": gen.answer, "pred_cw": gen.answer, "tie": False,
                    "rounds": [[gen.__dict__]]})
        return out
    out["referred"] = True
    ops = round0_opinions(item, roster, cfg["N"], cfg["model"], cfg["seed"], cfg["temp"],
                          cfg.get("effort"), meter, cfg.get("generic_roles", False), cfg=cfg)
    if has_majority(ops, valid):
        pred, tie = majority(ops, valid)
        out.update({"pred": pred, "pred_cw": conf_weighted(ops, valid), "tie": tie,
                    "rounds": [[o.__dict__ for o in ops]], "n_rounds": 1,
                    "entropy0": entropy(ops)})
        return out
    out["escalated"] = True
    d = arch_discussion(item, cfg, meter, roster=roster, ops0=ops)
    d.update({k: out[k] for k in ("generalist", "referred", "escalated")})
    return d


# ---------------------------------------------------------------- (C) Centralized MAS
# NMI 2026 (arXiv:2512.08296) formalisation: A = {a_orch, a_1..a_n}, C = star{(a_orch,a_i)},
# Omega = hierarchical, cost O(r*n*k). The orchestrator REVIEWS sub-agent outputs before
# aggregating -- the "validation bottleneck" that NMI credits with containing error
# amplification to 4.4x versus Independent's 17.2x. Clinically: the attending physician.
ORCH_SYS = ("You are the attending physician running a diagnostic panel. You do not answer from "
            "your own recall alone: you task specialists, then you audit what they return "
            "before you commit to an answer. You are accountable for the final decision.")

MAX_ORCH_ROUNDS = 2   # NMI allows up to 5; medical MCQA converges far sooner (measured)


def arch_centralized(item, cfg, meter, roster=None, rounds=MAX_ORCH_ROUNDS):
    valid = list(item["options"])
    N = cfg["N"]
    roster = roster or route(item, cfg.get("use_router", True))
    q = render_question(item)
    # star topology: sub-agents answer to the orchestrator only, never to each other.
    # These are the SAME round-0 calls as (1) -- cache-shared, so Centralized costs only
    # the orchestrator turns on top of Independent.
    ops = round0_opinions(item, roster, N, cfg["model"], cfg["seed"], cfg["temp"],
                          cfg.get("effort"), meter, cfg.get("generic_roles", False), cfg=cfg)
    hist = [[o.__dict__ for o in ops]]
    orch_turns = 0
    verdict = None
    for r in range(1, rounds + 1):
        order = _peer_order(item["qid"], 50 + r, len(ops))
        review = _ask(meter, orch_model(cfg), ORCH_SYS,
                      f"{q}\n\nYour specialists reported:\n{_peer_block(ops, order)}\n\n"
                      "Audit these opinions BEFORE aggregating. Name the single specialist "
                      "opinion most likely to be wrong and why. Then either commit to the "
                      "panel's answer, or state which specialty you need to re-task.\n\n"
                      'Reply with JSON only: {"answer": "<option letter>", '
                      '"confidence": <0-100>, "reason": "<at most 45 words>", '
                      '"retask": "<specialty to re-task, or empty to commit>"}',
                      cfg["seed"] * 100 + 80 + r, 0.3, cfg.get("effort"),
                      tag=f"orch{r}n{N}" + ("" if orch_model(cfg) == cfg["model"] else f"|{orch_model(cfg)}"),
                      max_tokens=round_budget(cfg["model"]), sbase=cfg["seed"])
        orch_turns += 1
        verdict = parse_opinion(review, valid, agent="orchestrator", rnd=r)
        hist.append([verdict.__dict__])
        retask = ""
        try:
            m = re.search(r"\{.*\}", review, re.S)
            retask = str(json.loads(m.group(0)).get("retask", "")).strip() if m else ""
        except Exception:
            retask = ""
        if not retask or r == rounds:
            break
        # re-task the named specialist: one sub-agent redoes its opinion with the
        # orchestrator's critique in hand. Star topology preserved (no peer edges).
        tgt = 0
        for i, sp in enumerate(roster[:N]):
            if sp.lower() in retask.lower() or retask.lower() in sp.lower():
                tgt = i; break
        txt = _ask(meter, cfg["model"], role_system(roster[tgt], cfg.get("generic_roles", False)),
                   f"{q}\n\nThe attending physician has audited the panel and raised this "
                   f"concern about your specialty's read:\n\"{verdict.reason}\"\n\n"
                   f"Reconsider and answer again.\n\n{ANSWER_JSON}",
                   cfg["seed"] * 100 + tgt, cfg["temp"], cfg.get("effort"),
                   tag=f"retask{r}n{N}", max_tokens=round_budget(cfg["model"]), sbase=cfg["seed"])
        ops = list(ops)
        ops[tgt] = parse_opinion(txt, valid, agent=ops[tgt].agent, rnd=r)
        hist.append([o.__dict__ for o in ops])
    return {"pred": verdict.answer if verdict else None,
            "pred_cw": verdict.answer if verdict else None, "tie": False,
            "rounds": hist, "n_rounds": len(hist), "orch_turns": orch_turns,
            "entropy0": entropy([Opinion(**d) for d in hist[0]]),
            "roster": roster[:N], "orchestrator": verdict.__dict__ if verdict else None}


# ---------------------------------------------------------------- (4) Debate
DISSENT_SYS_SUFFIX = (" You are the panel's assigned devil's advocate: your duty is to build the "
                      "strongest possible case for the most defensible ALTERNATIVE to the "
                      "consensus, and to name the reasoning error the majority is most likely "
                      "making. Commit to that alternative unless it is clearly untenable.")


def arch_debate(item, cfg, meter, roster=None, rounds=DEBATE_ROUNDS):
    valid = list(item["options"])
    N = cfg["N"]
    if N < 2:
        # R1 #3: "debate" with one agent is not debate. Refuse rather than contaminate the
        # N=1 point; analyses take the N=1 debate point from architecture (1).
        raise ValueError("architecture 'debate' requires N >= 2")
    roster = roster or route(item, cfg.get("use_router", True))
    from common.llm import pmap
    n_prop = max(1, N - 1)
    ops = round0_opinions(item, roster, n_prop, cfg["model"], cfg["seed"], cfg["temp"],
                          cfg.get("effort"), meter, cfg.get("generic_roles", False))
    if N > 1:
        di = N - 1
        d_txt = _ask(meter, cfg["model"],
                     role_system(roster[di], cfg.get("generic_roles", False)) + DISSENT_SYS_SUFFIX,
                     (f"{render_question(item)}\n\nThe panel's initial opinions:\n"
                      f"{_peer_block(ops, _peer_order(item['qid'], 0, len(ops)))}\n\n"
                      f"Argue the strongest alternative.\n\n{ANSWER_JSON}"),
                     cfg["seed"] * 100 + di, cfg["temp"], cfg.get("effort"), tag="dissent0",
                     sbase=cfg["seed"])
        ops = ops + [parse_opinion(d_txt, valid, agent=f"dissenter({roster[di]})#{di}", rnd=0)]
    hist = [[o.__dict__ for o in ops]]
    orders = []
    q = render_question(item)
    for rnd in range(1, rounds + 1):
        order = _peer_order(item["qid"], rnd, len(ops))
        orders.append(order)
        peers = _peer_block(ops, order)
        cur = ops

        def one(i):
            is_diss = i == N - 1 and N > 1
            sysmsg = role_system(roster[i], cfg.get("generic_roles", False)) + (
                DISSENT_SYS_SUFFIX if is_diss else "")
            u = (f"{q}\n\nDebate transcript (round {rnd - 1}):\n{peers}\n\n"
                 f"Rebut the arguments you disagree with and state your answer for round {rnd}.\n\n{ANSWER_JSON}")
            txt = _ask(meter, cfg["model"], sysmsg, u, cfg["seed"] * 100 + i, cfg["temp"],
                       cfg.get("effort"), tag=f"deb{rnd}n{N}v2", max_tokens=round_budget(cfg["model"]),
                       sbase=cfg["seed"])
            return parse_opinion(txt, valid, agent=cur[i].agent, rnd=rnd)

        ops = pmap(one, list(range(len(cur))))
        hist.append([o.__dict__ for o in ops])
    # moderator synthesis
    mod = _ask(meter, cfg["model"],
               "You are the moderating attending physician. Weigh the debate and decide.",
               f"{q}\n\nFinal debate positions:\n"
               f"{_peer_block(ops, _peer_order(item['qid'], 99, len(ops)))}\n\n"
               f"Give the panel's final answer.\n\n{ANSWER_JSON}",
               cfg["seed"] * 100 + 99, 0.3, cfg.get("effort"), tag=f"mod n{N}v2",
               max_tokens=round_budget(cfg["model"]),
               sbase=cfg["seed"])
    m = parse_opinion(mod, valid, agent="moderator", rnd=rounds + 1)
    return {"pred": m.answer, "pred_cw": m.answer, "tie": False, "peer_orders": orders,
            "rounds": hist + [[m.__dict__]], "n_rounds": len(hist) + 1,
            "entropy0": entropy([Opinion(**d) for d in hist[0]]),
            "entropy_final": entropy(ops), "roster": roster[:N], "moderator": m.__dict__}


# ---------------------------------------------------------------- baselines
def base_zeroshot(item, cfg, meter):
    valid = list(item["options"])
    u = render_question(item) + '\n\nAnswer immediately, with no explanation. Reply with JSON only: {"answer": "<option letter>"}'
    txt = _ask(meter, cfg["model"], role_system("internal medicine", generic=True), u,
               cfg["seed"] * 100 + 0, 0.3, cfg.get("effort"), tag="zs", max_tokens=baseline_budget(cfg["model"], 30),
               sbase=cfg["seed"])
    o = parse_opinion(txt, valid, "single-zs")
    return {"pred": o.answer, "pred_cw": o.answer, "tie": False, "rounds": [[o.__dict__]], "n_rounds": 1}


def base_cot(item, cfg, meter):
    valid = list(item["options"])
    u = (render_question(item) + "\n\nThink through the case step by step (briefly), then give "
         'your final answer. Reply with JSON only: {"answer": "<option letter>", '
         '"confidence": <0-100>, "reason": "<your step-by-step reasoning, at most 150 words>"}')
    txt = _ask(meter, cfg["model"], role_system("internal medicine", generic=True), u,
               cfg["seed"] * 100 + 1, 0.3, cfg.get("effort"), tag="cot", max_tokens=baseline_budget(cfg["model"], 800),
               sbase=cfg["seed"])
    o = parse_opinion(txt, valid, "single-cot")
    return {"pred": o.answer, "pred_cw": o.answer, "tie": False, "rounds": [[o.__dict__]], "n_rounds": 1}


def base_selfconsistency(item, cfg, meter, k):
    """Budget-matched control. The sample pool is built from cached n=8 chunks (OpenAI caps n at
    8), so any k is a prefix of the same samples. Nominal cost charged = ceil(k/8) prompts +
    k completions -- exactly what an efficient SC implementation would actually pay."""
    valid = list(item["options"])
    u = (render_question(item) + "\n\nThink through the case step by step (briefly), then give "
         'your final answer. Reply with JSON only: {"answer": "<option letter>", '
         '"confidence": <0-100>, "reason": "<at most 45 words>"}')
    texts = []
    n_chunks = (min(k, SC_POOL) + SC_CHUNK - 1) // SC_CHUNK
    for c in range(n_chunks):
        r = chat(cfg["model"], [{"role": "user", "content": u}],
                 system=role_system("internal medicine", generic=True),
                 temperature=_temp(cfg["model"], 0.7),
                 seed=None if is_reasoning(cfg["model"]) else cfg["seed"] * 1000 + c,
                 n=SC_CHUNK, max_tokens=opinion_budget(cfg["model"]), effort=cfg.get("effort"),
                 json_mode=True,
                 # R1 #4: seed MUST be in the cache tag, else "3 seeds" of SC are one
                 # cached sample set replayed three times.
                 tag=f"sc{c}s{cfg['seed']}")
        take = min(SC_CHUNK, k - len(texts))
        meter.add(r, n_charged=take, out_frac=take / max(1, len(r["texts"])))
        texts.extend(r["texts"][:take])
    ops = [parse_opinion(t, valid, f"sc#{i}") for i, t in enumerate(texts[:k])]
    pred, tie = majority(ops, valid)
    return {"pred": pred, "pred_cw": conf_weighted(ops, valid), "tie": tie,
            "rounds": [[o.__dict__ for o in ops]], "n_rounds": 1, "k": k,
            "entropy0": entropy(ops)}


# NMI 2026 canonical taxonomy -> our clinical instantiations.
ARCHES = {"independent": arch_independent,      # NMI Independent   (C = empty set)
          "centralized": arch_centralized,      # NMI Centralized   (star, attending physician)
          "discussion": arch_discussion,        # NMI Decentralized (fully connected, consensus)
          "tiered": arch_tiered,                # NMI Hybrid        (star + peer edges)
          "debate": arch_debate}                # medicine-specific extra
NMI_CLASS = {"zeroshot": "SAS", "cot": "SAS", "sc": "SAS-repeated",
             "independent": "Independent", "centralized": "Centralized",
             "discussion": "Decentralized", "tiered": "Hybrid", "debate": "Decentralized+mod"}


def run_episode(item, cfg):
    """cfg: {arch, N, model, seed, temp, effort?, use_router?, generic_roles?, k?}"""
    meter = Meter()
    a = cfg["arch"]
    if a in ARCHES:
        res = ARCHES[a](item, cfg, meter)
    elif a == "zeroshot":
        res = base_zeroshot(item, cfg, meter)
    elif a == "cot":
        res = base_cot(item, cfg, meter)
    elif a == "sc":
        res = base_selfconsistency(item, cfg, meter, cfg["k"])
    else:
        raise ValueError(f"unknown arch {a}")
    gold = item["answer"]
    res.update({"qid": item["qid"], "bench": item["bench"], "arch": a, "N": cfg.get("N", 1),
                "model": cfg["model"], "seed": cfg["seed"], "temp": cfg["temp"],
                "effort": cfg.get("effort"), "gold": gold,
                "correct": int(res.get("pred") == gold),
                "correct_cw": int(res.get("pred_cw") == gold),
                "abstain": int(res.get("pred") is None), "cost": meter.asdict()})
    return res
