"""Build unified, deterministic, NESTED item files for all benchmarks.

Nesting matters for cost: every sample is a prefix of the next larger one, so a pilot
run's LLM calls are cache hits when the full run happens later.
Schema: {qid, bench, stem, options{letter:text}, answer, meta{}}
"""
import json, pathlib, random, collections, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
LOCAL = ROOT / "data_local"
MAB = pathlib.Path("/home/myid/hj67104/night_shift/external/MedAgentsBench/data")


def write(name, rows):
    # R1 #2: the primary endpoint is exact match against these keys -- never ship a file
    # whose gold is not an option key.
    for r in rows:
        assert r["answer"] in r["options"], f"{name}: gold {r['answer']!r} not in options for {r['qid']}"
    p = OUT / name
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {p.name}: {len(rows)} items")
    return p


def stratified_shuffle(rows, keyfn, seed=42):
    """Deterministic interleave across strata -> any prefix is ~stratum-balanced."""
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[keyfn(r)].append(r)
    rng = random.Random(seed)
    for k in buckets:
        rng.shuffle(buckets[k])
    order = sorted(buckets, key=lambda k: (-len(buckets[k]), str(k)))
    out, i = [], 0
    while any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                out.append(buckets[k].pop())
        i += 1
    return out


def build_medxpertqa():
    src = LOCAL / "medxpertqa/Text/test.jsonl"
    rows = [json.loads(l) for l in src.open()]
    items = []
    for r in rows:
        stem = r["question"].split("\nAnswer Choices:")[0].strip()
        items.append({"qid": f"mxq/{r['id']}", "bench": "medxpertqa", "stem": stem,
                      "options": r["options"], "answer": r["label"],
                      "meta": {"body_system": r["body_system"], "task": r["medical_task"],
                               "qtype": r["question_type"]}})
    items = stratified_shuffle(items, lambda r: (r["meta"]["body_system"], r["meta"]["task"]))
    write("medxpertqa_1000.jsonl", items[:1000])
    write("medxpertqa_pilot200.jsonl", items[:200])
    # dev slice for threshold calibration -- DISJOINT from every evaluation sample
    write("medxpertqa_dev100.jsonl", items[1000:1100])


def build_medqa():
    src = LOCAL / "medqa/phrases_no_exclude_test.jsonl"
    rows = [json.loads(l) for l in src.open()]
    items = []
    for i, r in enumerate(rows):
        opts = r["options"]
        ans = [k for k, v in opts.items() if v == r["answer"]]
        if not ans:
            continue
        items.append({"qid": f"mq/{i}", "bench": "medqa", "stem": r["question"].strip(),
                      "options": opts, "answer": ans[0], "meta": {}})
    rng = random.Random(42); rng.shuffle(items)
    write("medqa_500.jsonl", items[:500])
    write("medqa_pilot200.jsonl", items[:200])
    write("medqa_dev100.jsonl", items[500:600])


def build_medagentsbench():
    subsets = [d.name for d in sorted(MAB.iterdir()) if d.is_dir() and d.name != "medqa_5options"]
    items = []
    for s in subsets:
        f = MAB / s / "test_hard.jsonl"
        if not f.exists():
            continue
        for j, line in enumerate(f.open()):
            r = json.loads(line)
            opts = r.get("options") or {}
            # R1 #2: normalise gold to an option KEY. answer_idx may be a letter ("D"), a
            # numeric index (0 is falsy -> must not use `or`), or absent.
            ans = None
            for field in ("answer_idx", "answer_letter"):
                v = r.get(field)
                if v is None:
                    continue
                v = str(v).strip()
                if v in opts:
                    ans = v; break
                if v.isdigit():
                    keys = list(opts)
                    if int(v) < len(keys):
                        ans = keys[int(v)]; break
            if ans is None:
                cand = [k for k, v in opts.items() if v == r.get("answer")]
                ans = cand[0] if cand else None
            if not ans or not opts:
                continue
            items.append({"qid": f"mab/{s}/{j}", "bench": "medagentsbench",
                          "stem": r["question"].strip(), "options": opts, "answer": ans,
                          "meta": {"subset": s}})
    items = stratified_shuffle(items, lambda r: r["meta"]["subset"])
    write("medagentsbench_hard.jsonl", items)
    write("medagentsbench_pilot200.jsonl", items[:200])


if __name__ == "__main__":
    print("building datasets ->", OUT)
    build_medxpertqa(); build_medqa(); build_medagentsbench()
