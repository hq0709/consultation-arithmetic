"""Shared episode primitives: prompt rendering, answer parsing, voting."""
from __future__ import annotations
import json, re, collections
from dataclasses import dataclass, field, asdict

ANSWER_JSON = ('Reply with JSON only, no other text:\n'
               '{"answer": "<option letter>", "confidence": <integer 0-100>, '
               '"reason": "<at most 45 words>"}')


def render_question(item) -> str:
    opts = "\n".join(f"({k}) {v}" for k, v in item["options"].items())
    return f"Clinical question:\n{item['stem']}\n\nOptions:\n{opts}"


def user_prompt(item) -> str:
    return render_question(item) + "\n\n" + ANSWER_JSON


@dataclass
class Opinion:
    agent: str
    answer: str | None
    confidence: float
    reason: str
    round: int = 0
    raw: str = ""


def parse_opinion(text: str, valid: list[str], agent: str, rnd: int = 0) -> Opinion:
    ans, conf, reason = None, 50.0, ""
    try:
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group(0)) if m else {}
        a = str(d.get("answer", "")).strip()
        m2 = re.search(r"[A-Za-z]", a)
        if m2:
            ans = m2.group(0).upper()
        try:
            conf = float(d.get("confidence", 50))
        except Exception:
            conf = 50.0
        reason = str(d.get("reason", ""))[:400]
    except Exception:
        pass
    if ans not in valid:                      # regex fallbacks on free text
        ans = None
        # R1 #9: NO bare single-letter fallback -- on 10-option MCQ a stray "A" in prose
        # would be scored as an answer. Only explicit cues or parenthesised letters count.
        for pat in (r"(?:answer|option|choice|final)\W{0,4}(?:is|:|=)?\W{0,4}\(?\b([A-J])\b\)?",
                    r"\(([A-J])\)"):
            for m in re.finditer(pat, text, re.I):
                cand = m.group(1).upper()
                if cand in valid:
                    ans = cand; break
            if ans:
                break
    conf = max(0.0, min(100.0, conf))
    return Opinion(agent=agent, answer=ans, confidence=conf, reason=reason, round=rnd, raw=text[:1500])


def unanimous(ops: list[Opinion], valid: list[str]) -> bool:
    """R1 #10: true unanimity -- EVERY agent gave the SAME VALID answer.
    [A, None, None] is not unanimity, it is two parse failures."""
    a = [o.answer for o in ops]
    return bool(a) and all(x in valid for x in a) and len(set(a)) == 1


def majority(ops: list[Opinion], valid: list[str]) -> tuple[str | None, bool]:
    """Majority vote; ties broken by mean confidence, then by option order (deterministic).
    Returns (answer, was_tie)."""
    votes = [o.answer for o in ops if o.answer in valid]
    if not votes:
        return None, False
    c = collections.Counter(votes)
    top = max(c.values())
    winners = [a for a, n in c.items() if n == top]
    if len(winners) == 1:
        return winners[0], False
    means = {w: sum(o.confidence for o in ops if o.answer == w) / max(1, sum(1 for o in ops if o.answer == w))
             for w in winners}
    # deterministic: highest mean confidence, ties broken by option order (never iteration order)
    return max(sorted(winners), key=lambda w: means[w]), True


def conf_weighted(ops: list[Opinion], valid: list[str]) -> str | None:
    s: dict[str, float] = {}
    for o in ops:
        if o.answer in valid:
            s[o.answer] = s.get(o.answer, 0.0) + o.confidence / 100.0
    return max(s, key=s.get) if s else None


def has_majority(ops: list[Opinion], valid: list[str]) -> bool:
    votes = [o.answer for o in ops if o.answer in valid]
    if not votes:
        return False
    c = collections.Counter(votes)
    return max(c.values()) > len(ops) / 2.0


def entropy(ops: list[Opinion]) -> float:
    import math
    votes = [o.answer for o in ops if o.answer]
    if not votes:
        return 0.0
    c = collections.Counter(votes); n = len(votes)
    return -sum((v / n) * math.log2(v / n) for v in c.values())
