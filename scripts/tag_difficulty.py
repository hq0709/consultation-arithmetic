"""Difficulty tagging (plan §3.5): pass rate of 3 mid-tier models at k=5 -> easy/medium/hard."""
import sys, pathlib, json, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common.llm import chat, pmap, LEDGER, is_reasoning
from panels.base import render_question, parse_opinion
from panels.roles import role_system

TAGGERS = [("gpt-4.1-nano", None), ("gpt-4o-mini", None), ("gpt-5-nano", "low")]
K = 5

ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="results/difficulty_tags.json")
a = ap.parse_args()

items = [json.loads(l) for l in open(ROOT / a.items)][: a.limit]
u_suffix = ('\n\nAnswer immediately. Reply with JSON only: {"answer": "<option letter>"}')


def tag(it):
    valid = list(it["options"]); hits = 0; tot = 0
    for m, eff in TAGGERS:
        r = chat(m, [{"role": "user", "content": render_question(it) + u_suffix}],
                 system=role_system("internal medicine", generic=True),
                 temperature=None if is_reasoning(m) else 0.7, n=K, max_tokens=30,
                 effort=eff, json_mode=True, tag="difftag")
        for t in r["texts"]:
            o = parse_opinion(t, valid, "tag")
            hits += int(o.answer == it["answer"]); tot += 1
    p = hits / max(1, tot)
    return {"qid": it["qid"], "pass_rate": p,
            "difficulty": "easy" if p >= 0.8 else ("hard" if p <= 0.2 else "medium")}


res = pmap(tag, items, workers=8)
(ROOT / a.out).write_text(json.dumps({r["qid"]: r for r in res}, indent=1))
import collections
print(collections.Counter(r["difficulty"] for r in res))
print(LEDGER.report())
