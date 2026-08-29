"""Mechanism metrics: opinion-diversity trajectory, first-speaker conformity, error correlation."""
from __future__ import annotations
import json, math, collections, itertools


def _answers(round_ops):
    return [o.get("answer") for o in round_ops]


def diversity_trajectory(ep):
    """Per round: answer-distribution entropy, unique-answer count, mean pairwise rationale Jaccard."""
    out = []
    for r, ops in enumerate(ep.get("rounds", [])):
        ans = [a for a in _answers(ops) if a]
        c = collections.Counter(ans); n = len(ans)
        ent = -sum((v / n) * math.log2(v / n) for v in c.values()) if n else 0.0
        toks = [set(str(o.get("reason", "")).lower().split()) for o in ops]
        js = [len(a & b) / len(a | b) for a, b in itertools.combinations(toks, 2) if a | b]
        out.append({"round": r, "n": len(ops), "entropy": ent, "unique": len(c),
                    "jaccard": sum(js) / len(js) if js else None,
                    "agree_frac": (max(c.values()) / n) if n else 0.0})
    return out


def first_speaker_conformity(ep):
    """P(final = agent#0's round-0 answer), and whether agent#0 was right.
    'First speaker' = the first-listed panelist, i.e. the top-ranked specialty."""
    rounds = ep.get("rounds", [])
    if not rounds or not rounds[0]:
        return None
    first = rounds[0][0].get("answer")
    return {"first": first, "final": ep.get("pred"), "gold": ep.get("gold"),
            "final_eq_first": int(ep.get("pred") == first) if first else None,
            "first_correct": int(first == ep.get("gold")) if first else None,
            "n": ep.get("N")}


def error_correlation(eps):
    """Pairwise agent error-correlation matrix (phi) over round-0 opinions, by agent slot."""
    per_slot = collections.defaultdict(dict)   # slot -> qid -> wrong(0/1)
    for ep in eps:
        rounds = ep.get("rounds", [])
        if not rounds:
            continue
        for i, o in enumerate(rounds[0]):
            per_slot[i][ep["qid"]] = int(o.get("answer") != ep.get("gold"))
    slots = sorted(per_slot)
    mat = {}
    for i, j in itertools.combinations(slots, 2):
        qs = set(per_slot[i]) & set(per_slot[j])
        if len(qs) < 10:
            continue
        x = [per_slot[i][q] for q in qs]; y = [per_slot[j][q] for q in qs]
        n = len(qs); mx = sum(x) / n; my = sum(y) / n
        num = sum((a - mx) * (b - my) for a, b in zip(x, y))
        den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
        mat[f"{i}-{j}"] = num / den if den else 0.0
    vals = list(mat.values())
    return {"pairs": mat, "mean_phi": sum(vals) / len(vals) if vals else None,
            "n_pairs": len(vals)}
