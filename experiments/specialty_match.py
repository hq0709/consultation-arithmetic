"""专科匹配到底有没有用？—— 第一根柱子的临床版本。

多学科会诊的整个前提是：把对的专科请到桌上。本实验直接检验它。
每道 MedXpertQA 题带 body_system 标注，路由器按相关性排序专科。
于是 N 越小，面板越可能**只**包含最相关的专科；N 越大，越会掺入不相关的。

若"请对专科"重要，那么：
  (a) 面板包含本病例对应专科时，应当表现更好；
  (b) 用**通用内科医生**替换专科角色，表现应当下降。
两条都能直接测。
"""
import sys, pathlib, glob, json, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.analyze import load, mcnemar

# body_system -> 与之对口的专科（路由器词表里的名称）
SYS2SPEC = {
    "Nervous": {"neurology", "neurosurgery"},
    "Cardiovascular": {"cardiology", "cardiothoracic surgery"},
    "Respiratory": {"pulmonology", "critical care medicine"},
    "Digestive": {"gastroenterology", "hepatology", "general surgery"},
    "Renal": {"nephrology", "urology"}, "Urinary": {"nephrology", "urology"},
    "Endocrine": {"endocrinology"}, "Reproductive": {"obstetrics and gynecology", "urology"},
    "Musculoskeletal": {"orthopedics", "rheumatology"},
    "Integumentary": {"dermatology"}, "Skin": {"dermatology"},
    "Lymphatic": {"hematology-oncology", "infectious disease"},
    "Immune": {"immunology", "rheumatology", "infectious disease"},
    "Blood": {"hematology-oncology"},
    "Other": set(),
}


def main():
    items = {}
    for l in (ROOT / "data/medxpertqa_250.jsonl").open():
        r = json.loads(l)
        items[r["qid"]] = r.get("meta", {})

    rows = load(sorted(glob.glob(str(ROOT / "results/G_*.jsonl"))))
    print("=" * 74)
    print("1. 面板里有没有对口专科，成绩有差别吗？")
    print("=" * 74)
    per = collections.defaultdict(lambda: {"hit": [], "miss": []})
    sysct = collections.Counter()
    for r in rows:
        if r["bench"] != "medxpertqa" or r["arch"] not in ("independent", "centralized",
                                                           "discussion") or r["N"] < 3:
            continue
        if r.get("status") == "error":
            continue
        meta = items.get(r["qid"], {})
        want = SYS2SPEC.get(meta.get("body_system"), set())
        if not want:
            continue
        f0 = (r.get("rounds") or [[]])[0]
        specs = {str(o.get("agent", "")).split("#")[0].strip().lower() for o in f0}
        if not specs:
            continue
        sysct[meta["body_system"]] += 1
        per[(r["model"], r["N"])]["hit" if specs & want else "miss"].append(r["correct"])

    print(f"{'model':16s}{'N':>3s}{'含对口专科':>12s}{'不含':>12s}{'差':>8s}")
    dif = []
    for (m, N), d in sorted(per.items()):
        if len(d["hit"]) < 40 or len(d["miss"]) < 40:
            continue
        h = np.mean(d["hit"]) * 100; s = np.mean(d["miss"]) * 100
        dif.append(h - s)
        print(f"  {m:14s}{N:3d}{h:9.1f}% (n={len(d['hit']):4d}){s:8.1f}% (n={len(d['miss']):4d})"
              f"{h - s:+8.1f}")
    if dif:
        print(f"\n  平均差异 {np.mean(dif):+.2f}pp   (若专科匹配重要，这里应显著为正)")

    print("\n" + "=" * 74)
    print("2. 用通用内科医生替换专科角色，成绩会掉吗？")
    print("=" * 74)
    gen = [f for f in glob.glob(str(ROOT / "results/*.jsonl")) if "generic" in f.lower()]
    print(f"  找到通用内科对照文件: {[pathlib.Path(f).name for f in gen] or '无'}")
    if gen:
        g = load(gen)
        gc = collections.defaultdict(list)
        for r in g:
            if r.get("status") != "error":
                gc[(r["model"], r["bench"], r["arch"], r["N"])].append(r["correct"])
        sc = collections.defaultdict(list)
        for r in rows:
            if r.get("status") != "error":
                sc[(r["model"], r["bench"], r["arch"], r["N"])].append(r["correct"])
        print(f"\n{'cell':40s}{'专科角色':>10s}{'通用内科':>10s}{'差':>8s}")
        d2 = []
        for k in sorted(gc):
            if k not in sc or len(gc[k]) < 40:
                continue
            a = np.mean(sc[k]) * 100; b = np.mean(gc[k]) * 100
            d2.append(a - b)
            print(f"  {k[0][:14]:15s}{k[1][:12]:13s}{k[2][:10]:11s}N={k[3]:<2d}{a:8.1f}%{b:9.1f}%{a-b:+8.1f}")
        if d2:
            print(f"\n  平均差异 {np.mean(d2):+.2f}pp  (若专科身份重要，这里应显著为正)")


if __name__ == "__main__":
    main()
