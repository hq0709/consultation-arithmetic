"""NMI 2026 (arXiv:2512.08296) coordination metrics, computed per configuration.

Every definition below is stated explicitly because NMI gives the metric NAMES and their
headline values but not always closed forms. Where we had to choose a form we say so, and
we keep NMI's own definition wherever it is unambiguous (Absorb, O%, turn power law).
"""
from __future__ import annotations
import math, collections, itertools


# ---------------------------------------------------------------- per-episode primitives
def turns(ep) -> int:
    """T: reasoning-response exchanges. One LLM call = one turn (NMI's unit)."""
    return ep.get("cost", {}).get("calls", 0)


def messages(ep) -> int:
    """Number of peer-opinion exposures: how many other agents' opinions were placed in
    some agent's context. This is the operational reading of NMI's message count on a
    communication graph C -- Independent has C = empty set and therefore 0 messages."""
    arch = ep.get("arch")
    rounds = ep.get("rounds", [])
    N = ep.get("N", 1)
    if arch in ("independent", "zeroshot", "cot", "sc"):
        return 0                      # C = empty set
    if arch == "centralized":         # star: only the orchestrator reads the panel
        return sum(N for r in rounds[1:] if r)
    # decentralized / hybrid / debate: every agent reads every other agent, each round
    n_disc = max(0, len(rounds) - 1)
    return n_disc * N * max(0, N - 1)


def msg_density(ep) -> float:
    """c: messages per reasoning turn (NMI plateaus at c* = 0.39)."""
    t = turns(ep)
    return messages(ep) / t if t else 0.0


def redundancy(ep) -> float:
    """R: mean pairwise rationale overlap among round-0 opinions (Jaccard over tokens).
    High R = the panel is saying the same thing = little independent information."""
    r0 = (ep.get("rounds") or [[]])[0]
    toks = [set(str(o.get("reason", "")).lower().split()) for o in r0]
    js = [len(a & b) / len(a | b) for a, b in itertools.combinations(toks, 2) if a | b]
    return sum(js) / len(js) if js else 0.0


def n_agents(ep) -> int:
    a = ep.get("arch")
    if a in ("zeroshot", "cot"):
        return 1
    if a == "sc":
        return ep.get("N", 1)
    if a == "centralized":
        return ep.get("N", 1) + 1        # + orchestrator
    if a == "tiered":
        return ep.get("N", 1) + 1        # + generalist
    return ep.get("N", 1)


# ---------------------------------------------------------------- per-configuration
def config_metrics(eps, sas_eps):
    """eps: episodes of one (arch, N, model, bench) cell. sas_eps: the SAS baseline cell
    (single CoT) on the SAME items, for the overhead and absorption denominators."""
    if not eps:
        return None
    n = len(eps)
    acc = sum(e["correct"] for e in eps) / n
    tok = sum(e["cost"]["in_tok"] + e["cost"]["out_tok"] for e in eps) / n
    usd = sum(e["cost"]["usd"] for e in eps) / n
    T = sum(turns(e) for e in eps) / n
    c = sum(msg_density(e) for e in eps) / n
    R = sum(redundancy(e) for e in eps) / n

    sas = {e["qid"]: e for e in sas_eps}
    shared = [e for e in eps if e["qid"] in sas]
    sas_tok = (sum(sas[e["qid"]]["cost"]["in_tok"] + sas[e["qid"]]["cost"]["out_tok"]
                   for e in shared) / len(shared)) if shared else 0.0
    sas_err = (sum(1 - sas[e["qid"]]["correct"] for e in shared) / len(shared)) if shared else 0.0
    mas_err = (sum(1 - e["correct"] for e in shared) / len(shared)) if shared else 0.0

    # O%: NMI's coordination overhead, token-based, relative to the single-agent baseline.
    O = (tok - sas_tok) / sas_tok * 100 if sas_tok else 0.0
    # E_c: NMI describes efficiency as a success/overhead ratio.
    E_c = acc / (tok / sas_tok) if sas_tok else acc
    # Absorb: NMI's exact definition (E_SAS - E_MAS)/E_SAS.
    absorb = (sas_err - mas_err) / sas_err if sas_err else 0.0

    # A_e: NMI describes error amplification as individual-agent errors reaching the final
    # output "without inter-agent verification". We measure exactly that failure: the panel
    # HELD the right answer and still emitted a wrong one, normalised by how often a
    # single agent is wrong. A_e > 1 = the system destroys information it already had.
    held_and_lost = 0; had_correct = 0; agent_wrong = 0; agent_tot = 0
    for e in eps:
        r0 = (e.get("rounds") or [[]])[0]
        answers = [o.get("answer") for o in r0]
        gold = e.get("gold")
        for a in answers:
            agent_tot += 1; agent_wrong += int(a != gold)
        if gold in answers:
            had_correct += 1
            held_and_lost += int(e.get("pred") != gold)
    p_agent_wrong = agent_wrong / agent_tot if agent_tot else 0.0
    p_lost = held_and_lost / had_correct if had_correct else 0.0
    A_e = p_lost / p_agent_wrong if p_agent_wrong else 0.0

    return {"n": n, "accuracy": acc, "tokens": tok, "usd": usd, "turns": T,
            "msg_density": c, "redundancy": R, "overhead_pct": O, "efficiency": E_c,
            "absorb": absorb, "error_amp": A_e, "p_held_and_lost": p_lost,
            "p_agent_wrong": p_agent_wrong, "n_agents": n_agents(eps[0]),
            "sas_baseline": 1 - sas_err}


def fit_turn_powerlaw(points):
    """NMI: T = 2.72*(n+0.5)^1.724, R^2=0.974. Fit log T = log a + b*log(n+0.5)."""
    pts = [(n, t) for n, t in points if n > 0 and t > 0]
    if len(pts) < 3:
        return None
    xs = [math.log(n + 0.5) for n, _ in pts]; ys = [math.log(t) for _, t in pts]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sxy / sxx if sxx else 0.0
    a = math.exp(my - b * mx)
    ss_res = sum((y - (math.log(a) + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"a": a, "exponent": b, "r2": 1 - ss_res / ss_tot if ss_tot else 0.0, "n_points": len(pts)}


def fit_msg_density_log(points):
    """NMI: S = 0.73 + 0.28*ln(c), R^2=0.68, plateau c* = 0.39."""
    pts = [(c, s) for c, s in points if c > 0]
    if len(pts) < 3:
        return None
    xs = [math.log(c) for c, _ in pts]; ys = [s for _, s in pts]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"intercept": a, "slope": b, "r2": 1 - ss_res / ss_tot if ss_tot else 0.0,
            "n_points": len(pts)}
