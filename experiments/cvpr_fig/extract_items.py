"""Per-item data for the extra figure candidates (results/cvpr_items.json).

  python experiments/cvpr_fig/extract_items.py

From the raw episodes (results/IMG2_*.jsonl, read through experiments/imaging_v2_analysis.py):
- hist: for every model and data set, the number of correct agents (0-9) on each item in the
  independent panel of 9 with original images (initial answers)
- occlusion: SLAKE-target, per item and proprietary model, the share of agents whose answer changed
  under target and under control occlusion (independent panel of 9), and every agent's answer letter
  in the three image conditions
- pairs: SLAKE-pairs, per pair and proprietary model, the single agent's answers on the two images
- reports: SLAKE and VQA-RAD, per item, slot-0 reports of the three perception models and the mean
  agent accuracy of each reasoning model with each perception model's reports and with the image,
  with every agent's answer letter
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import imaging_v2_analysis as ia                                    # noqa: E402

API = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001"]
MODELS = API + ["qwen3.8-flash"]
rows = ia.load()
C = ia.cells(rows)
items = {}
for f in ("vqarad_closed_250", "slake_closed_250", "mxqa_mm_rad_250", "slake_target_250", "slake_pairs_200"):
    for line in open(ROOT / "data" / f"{f}.jsonl"):
        it = json.loads(line)
        items[it["qid"]] = it


def g(m, s, cond, arch, n, mode, per=None, sh=None):
    return {e["qid"]: e for e in C.get((m, s, cond, arch, n, mode, None, per, sh), [])}


out = {"hist": {}, "occlusion": {}, "pairs": {}, "reports": {}}
for m in MODELS:
    for s in ("vqarad", "slake", "mxqa_mm_rad"):
        E = g(m, s, "panel", "independent", 9, "real")
        if E:
            out["hist"][f"{m}|{s}"] = [int(sum(o.get("answer") == e["gold"] for o in e["rounds"][0])) for e in E.values()
                                       if len(e["rounds"][0]) == 9]

for m in API:
    R, T, K = (g(m, "slake_target", "panel", "independent", 9, md) for md in ("real", "occ_target", "occ_control"))
    for q in sorted(set(R) & set(T) & set(K)):
        r0 = [o.get("answer") for o in R[q]["rounds"][0]]
        ch = lambda E: sum(a != b for a, b in zip(r0, [o.get("answer") for o in E[q]["rounds"][0]])) / len(r0)
        d = out["occlusion"].setdefault(q, {"stem": items[q]["stem"], "gold": items[q]["options"][items[q]["answer"]],
                                            "gold_letter": R[q]["gold"], "options": items[q]["options"],
                                            "image": items[q]["images"][0], "target_box": items[q]["target_box"],
                                            "control_box": items[q]["control_box"], "meta": items[q]["meta"], "models": {}})
        d["models"][m] = {"target": ch(T), "control": ch(K),
                          "correct_real": sum(a == R[q]["gold"] for a in r0) / len(r0),
                          "answers": {k: [o.get("answer") for o in E[q]["rounds"][0]]
                                      for k, E in (("real", R), ("occ_target", T), ("occ_control", K))}}

for m in API:
    S = g(m, "slake_pairs", "single", "cot", 1, "real")
    for q, e in S.items():
        if not q.endswith("-a") or q[:-1] + "b" not in S:
            continue
        b = S[q[:-1] + "b"]
        ia_, ib = items[q], items[q[:-1] + "b"]
        txt = lambda it, ep: it["options"].get(ep["rounds"][0][0].get("answer"), "?")
        d = out["pairs"].setdefault(q[:-2], {"stem": ia_["stem"], "a": {"image": ia_["images"][0], "gold": ia_["options"][ia_["answer"]]},
                                             "b": {"image": ib["images"][0], "gold": ib["options"][ib["answer"]]}, "models": {}})
        d["models"][m] = {"a": txt(ia_, e), "b": txt(ib, b)}

for s in ("vqarad", "slake"):
    D = {r: g(r, s, "generic", "independent", 9, "real") for r in API}
    for r in API:
        for p in API:
            E = g(r, s, "report", "report", 9, "real", per=p, sh=False)
            for q, e in E.items():
                d = out["reports"].setdefault(q, {"set": s, "stem": items[q]["stem"], "gold": items[q]["options"][items[q]["answer"]],
                                                  "gold_letter": e["gold"], "image": items[q]["images"][0], "report": {},
                                                  "acc": {}, "answers": {}})
                if e.get("reports"):
                    d["report"][p] = e["reports"][0]
                d["acc"][f"{r}|{p}"] = sum(o.get("answer") == e["gold"] for o in e["rounds"][0]) / len(e["rounds"][0])
                d["answers"][f"{r}|{p}"] = [o.get("answer") for o in e["rounds"][0]]
        for q, e in D[r].items():
            if q in out["reports"]:
                out["reports"][q]["acc"][f"{r}|image"] = sum(o.get("answer") == e["gold"] for o in e["rounds"][0]) / len(e["rounds"][0])
                out["reports"][q]["answers"][f"{r}|image"] = [o.get("answer") for o in e["rounds"][0]]

p = ROOT / "results" / "cvpr_items.json"
p.write_text(json.dumps(out))
print(p, {k: len(v) for k, v in out.items()})
