"""Imaging pilot: do AI reading panels give independent second reads?

Reads results/IMG_<model>_<bench>.jsonl (scripts/run_imaging_pilot.sh) and reports, per model x
benchmark x image mode:
  accuracy of the single reader (cot) and of Independent / Decentralized panels at N = 3, 9;
  phi (first-round error correlation, the paper's definition: experiments/independence.py)
  and N_eff = N / (1 + (N - 1) phi) at N = 9; phi after discussion;
  the oracle at N = 9 (any first-round reader correct) and headroom over the single reader;
  unanimity and the error rate among unanimous answers (Independent first round, Decentralized
  last round);
  the blank-image ablation: accuracy and phi with the image replaced by a same-size grey image,
  and the share of single-reader answers that change when the image is removed.
95% intervals are percentile bootstrap over items (2,000 resamples, seed 0).
Writes results/imaging_pilot.json.
"""
import collections, glob, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.independence import phi
from experiments.refresh_stale import unanimous

B = 2000
rng = np.random.default_rng(0)


def load():
    rows, seen = [], set()
    for p in sorted(glob.glob(str(ROOT / "results/IMG_*_*.jsonl"))):
        if "smoke" in p:
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("status") != "ok":
                continue
            k = (r["qid"], r.get("cfg_hash"))
            if k in seen:
                continue
            seen.add(k); rows.append(r)
    return rows


def ci(fn, eps, b=B):
    """Point estimate and percentile bootstrap CI of fn over items."""
    est = fn(eps)
    if est is None:
        return None
    idx = np.arange(len(eps))
    boots = []
    for _ in range(b):
        s = [eps[i] for i in rng.choice(idx, len(idx))]
        v = fn(s)
        if v is not None:
            boots.append(v)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [round(float(est), 4), round(float(lo), 4), round(float(hi), 4)]


def acc(eps):
    return float(np.mean([e["correct"] for e in eps])) * 100


def oracle(eps):
    return float(np.mean([any(o.get("answer") == e["gold"] for o in e["rounds"][0]) for e in eps])) * 100


def una_stats(eps):
    u = [(unanimous(e), e["correct"]) for e in eps]
    u = [(a, c) for a, c in u if a is not None]
    una = [c for a, c in u if a]
    return {"unanimity": len(una) / len(u) * 100, "wrong_given_unanimous": (1 - np.mean(una)) * 100 if una else None}


def paired(a, b):
    """Paired items present in both episode lists, keyed by qid."""
    da = {e["qid"]: e for e in a}; db = {e["qid"]: e for e in b}
    qs = sorted(set(da) & set(db))
    return [(da[q], db[q]) for q in qs]


def main():
    rows = load()
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model"], r["bench"], r.get("image_mode", "real"), r["arch"], r["N"])].append(r)
    out = {}
    for (m, b) in sorted({(k[0], k[1]) for k in cells}):
        g = lambda mode, a, n: cells.get((m, b, mode, a, n), [])
        res = {"n_items": len(g("real", "cot", 1))}
        for mode in ("real", "blank"):
            d = {}
            for a, n in (("cot", 1), ("independent", 3), ("independent", 9), ("discussion", 3), ("discussion", 9)):
                eps = g(mode, a, n)
                if eps:
                    d[f"acc_{a}_{n}"] = ci(acc, eps)
            ind9 = g(mode, "independent", 9)
            if ind9:
                d["phi0_9"] = ci(lambda e: phi(e, 0), ind9, b=500)
                d["neff_9"] = round(9 / (1 + 8 * d["phi0_9"][0]), 3)
                d["oracle_9"] = ci(oracle, ind9)
                d["headroom_9"] = round(d["oracle_9"][0] - d["acc_cot_1"][0], 2) if "acc_cot_1" in d else None
                d["unanimity_ind9"] = una_stats(ind9)
            dis9 = g(mode, "discussion", 9)
            if dis9:
                d["phi_last_disc9"] = ci(lambda e: phi(e, -1), dis9, b=500)
                d["unanimity_disc9"] = una_stats(dis9)
            res[mode] = d
        pr = paired(g("real", "cot", 1), g("blank", "cot", 1))
        if pr:
            res["ablation"] = {
                "single_reader_acc_drop_pp": ci(lambda s: float(np.mean([x["correct"] - y["correct"] for x, y in s])) * 100, pr),
                "single_reader_answer_changed_pct": ci(lambda s: float(np.mean([x["pred"] != y["pred"] for x, y in s])) * 100, pr),
            }
        out[f"{m}|{b}"] = res
    (ROOT / "results/imaging_pilot.json").write_text(json.dumps(out, indent=1))
    for k, res in out.items():
        print(f"\n== {k}  (items {res['n_items']})")
        for mode in ("real", "blank"):
            d = res.get(mode, {})
            if not d:
                continue
            f = lambda x: f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]" if x else "—"
            print(f"  {mode:5s} single {f(d.get('acc_cot_1'))} | Ind3 {f(d.get('acc_independent_3'))} | Ind9 {f(d.get('acc_independent_9'))}"
                  f" | Dec3 {f(d.get('acc_discussion_3'))} | Dec9 {f(d.get('acc_discussion_9'))}")
            if "phi0_9" in d:
                p = d["phi0_9"]
                print(f"        phi0 {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]  N_eff(9) {d['neff_9']:.2f}  oracle {f(d['oracle_9'])}"
                      f"  headroom {d['headroom_9']:+.1f} pp  unanimity(Ind9) {d['unanimity_ind9']['unanimity']:.1f}%"
                      f" wrong|unanimous {d['unanimity_ind9']['wrong_given_unanimous'] or 0:.1f}%")
            if "phi_last_disc9" in d:
                p = d["phi_last_disc9"]; u = d["unanimity_disc9"]
                print(f"        after discussion: phi {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]  unanimity {u['unanimity']:.1f}%"
                      f"  wrong|unanimous {u['wrong_given_unanimous'] or 0:.1f}%")
        if "ablation" in res:
            a = res["ablation"]
            print(f"  ablation: single-reader accuracy drop {a['single_reader_acc_drop_pp'][0]:+.1f} pp "
                  f"[{a['single_reader_acc_drop_pp'][1]:+.1f}, {a['single_reader_acc_drop_pp'][2]:+.1f}];"
                  f" answers changed {a['single_reader_answer_changed_pct'][0]:.1f}%")
    print("\nwrote results/imaging_pilot.json")


if __name__ == "__main__":
    main()
