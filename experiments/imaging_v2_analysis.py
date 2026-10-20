"""Protocol-v2 analysis (docs/IMAGING_PROTOCOL.md sections 6-7, 11).

Rows: results/IMG2_<model>_<set>.jsonl. Each row's cfg_hash is mapped back to its condition by
recomputing the hash of every planned configuration (same function as experiments/run_grid.py).
All intervals are item-level percentile bootstraps (2000 resamples, seed 0) unless stated.

  python experiments/imaging_v2_analysis.py A      # experiment A -> results/imaging_v2_A.json
"""
import collections, glob, hashlib, itertools, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np

MODELS = ["gpt-5-mini", "gemini-3.7-flash", "claude-haiku-4-5-20251001", "qwen3.8-flash"]
# NMI-style capability ladder (scripts/run_imaging_nmi.sh): within-family rungs added to the four models
LADDER = ["gpt-5-nano", "gpt-5", "gemini-3.5-flash-lite", "claude-sonnet-5", "qwen3.8-max-0902"]
SHORT = {"gpt-5-mini": "GPT-5 mini", "gemini-3.7-flash": "Gemini 3.7 Flash",
         "claude-haiku-4-5-20251001": "Claude Haiku 4.5", "qwen3.8-flash": "Qwen 3.8 Flash"}
SETS = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM",
        "slake_target": "SLAKE-target", "slake_pairs": "SLAKE-pairs"}
RENDER = ",,,gamma0.7,gamma0.7,gamma0.7,clahe,clahe,clahe".split(",")
B = 2000


# ---------------------------------------------------------------- condition labels
def cfg_hash(c):
    keys = ("arch", "N", "k", "model", "seed", "temp", "effort", "theta", "use_router", "generic_roles")
    d = {k: c.get(k) for k in keys}
    for k in ("image_mode", "render", "memory", "perceiver", "shared_report"):
        if c.get(k) not in (None, False):
            d[k] = c[k]
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def planned(m):
    """-> {cfg_hash: label dict} for every configuration scripts/run_imaging_v2_full.sh runs."""
    base = {"model": m, "seed": 1, "temp": 0.7, "effort": "low" if m.startswith("gpt-5") else None,
            "use_router": True, "generic_roles": False}
    out = {}

    def add(lab, **kw):
        c = dict(base, **kw)
        out[cfg_hash(c)] = dict(lab, model=m, arch=c["arch"], N=c["N"], image_mode=c.get("image_mode"))
    for mode in ("real", "blank", "res256", "res128", "occ_target", "occ_control"):
        add({"cond": "single", "roles": "generic"}, arch="cot", N=1, image_mode=mode)
        for n in (1, 3, 9):
            add({"cond": "panel", "roles": "specialty"}, arch="independent", N=n, image_mode=mode)
    for n in (1, 3, 9):
        add({"cond": "discussion", "roles": "specialty"}, arch="discussion", N=n, image_mode="real")
    for n in (3, 9):            # NMI Centralized (orchestrator audits, star) and Hybrid (tiered referral)
        add({"cond": "centralized", "roles": "specialty"}, arch="centralized", N=n, image_mode="real")
        add({"cond": "hybrid", "roles": "specialty"}, arch="tiered", N=n, image_mode="real")
    add({"cond": "generic", "roles": "generic"}, arch="independent", N=9, generic_roles=True, image_mode="real")
    add({"cond": "render", "roles": "generic"}, arch="independent", N=9, generic_roles=True, image_mode="real",
        render=RENDER)
    for mem in ("shared-image", "private-image", "shared-text", "private-text"):
        add({"cond": "memory", "memory": mem, "roles": "generic"}, arch="independent", N=9, generic_roles=True,
            image_mode="real", memory=mem)
    for mem in ("shared-image", "shared-text"):
        add({"cond": "memory-single", "memory": mem, "roles": "generic"}, arch="cot", N=1, image_mode="real",
            memory=mem)
    for p in MODELS[:3]:
        add({"cond": "report", "perceiver": p, "shared": False, "roles": "generic"}, arch="report", N=9,
            image_mode="real", perceiver=p, shared_report=False)
        add({"cond": "report", "perceiver": p, "shared": True, "roles": "generic"}, arch="report", N=9,
            image_mode="real", perceiver=p, shared_report=True)
    return out


LABELS = {h: lab for m in MODELS + LADDER for h, lab in planned(m).items()}


PAIR_OPTS = {json.loads(l)["qid"]: json.loads(l)["options"] for l in open(ROOT / "data/slake_pairs_200.jsonl")}


def bench_of(r):
    b = r["bench"]
    return "mxqa_mm_rad" if b.startswith("mxqa") else b


def load():
    rows, seen, unknown = [], set(), collections.Counter()
    for p in sorted(glob.glob(str(ROOT / "results/IMG2_*.jsonl"))):
        for l in open(p):
            r = json.loads(l)
            if r.get("status") != "ok":
                continue
            k = (r["qid"], r["cfg_hash"], r["model"])
            if k in seen:
                continue
            seen.add(k)
            lab = LABELS.get(r["cfg_hash"])
            if lab is None:
                unknown[pathlib.Path(p).stem] += 1
                continue
            r["_lab"] = lab; r["_set"] = bench_of(r)
            rows.append(r)
    if unknown:
        print("!! rows with unplanned cfg_hash:", dict(unknown))
    return rows


def cells(rows):
    c = collections.defaultdict(list)
    for r in rows:
        L = r["_lab"]
        key = (r["model"], r["_set"], L["cond"], L["arch"], L["N"], L.get("image_mode"),
               L.get("memory"), L.get("perceiver"), L.get("shared"))
        c[key].append(r)
    return c


# ---------------------------------------------------------------- statistics
def boot(fn, units, b=B):
    """fn(list of units) -> float; percentile bootstrap over units. -> [est, lo, hi]. Each call draws
    from its own generator (seed 0), so an interval does not depend on which analyses ran before it."""
    rng = np.random.default_rng(0)
    if not units:
        return None
    est = fn(units)
    if est is None:
        return None
    vals = []
    for _ in range(b):
        s = [units[i] for i in rng.integers(0, len(units), len(units))]
        v = fn(s)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(est), 4), round(float(lo), 4), round(float(hi), 4)]


def acc(eps):
    return float(np.mean([e["correct"] for e in eps])) * 100 if eps else None


def err_matrix(eps, rnd=0):
    """items x slots matrix of error indicators (1 = wrong) for one panel condition."""
    mats = []
    for e in eps:
        r = (e.get("rounds") or [[]])[rnd]
        mats.append([int(o.get("answer") != e["gold"]) for o in r])
    n = min(len(x) for x in mats) if mats else 0
    return np.array([x[:n] for x in mats], float)


def phi_from_matrix(E):
    """mean pairwise Pearson correlation of error indicators across items (experiments/independence.py)."""
    vals = []
    for i, j in itertools.combinations(range(E.shape[1]), 2):
        x, y = E[:, i], E[:, j]
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(vals)) if vals else None


def phi(eps, rnd=0):
    if not eps:
        return None
    return phi_from_matrix(err_matrix(eps, rnd))


def phi_answered(eps, rnd=0):
    """Sensitivity version of phi: each reader pair uses only the items both readers answered
    (an abstention is then neither an error nor a correct answer)."""
    if not eps:
        return None
    A = [[o.get("answer") for o in (e.get("rounds") or [[]])[rnd]] for e in eps]
    n = min(len(a) for a in A)
    gold = [e["gold"] for e in eps]
    vals = []
    for i, j in itertools.combinations(range(n), 2):
        keep = [k for k, a in enumerate(A) if a[i] is not None and a[j] is not None]
        if len(keep) < 30:
            continue
        x = np.array([A[k][i] != gold[k] for k in keep], float); y = np.array([A[k][j] != gold[k] for k in keep], float)
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(vals)) if vals else None


def abstention(eps, rnd=0):
    ops = [o for e in eps for o in (e.get("rounds") or [[]])[rnd]]
    return sum(1 for o in ops if not o.get("answer")) / max(1, len(ops)) * 100


def unanimity(eps, rnd=-1):
    u = []
    for e in eps:
        r = (e.get("rounds") or [[]])[rnd]
        a = [o.get("answer") for o in r]
        if len(a) >= 2:
            u.append((len(set(a)) == 1 and a[0] is not None, e["correct"]))
    una = [c for x, c in u if x]
    return {"unanimity": len(una) / len(u) * 100 if u else None,
            "wrong_given_unanimous": (1 - float(np.mean(una))) * 100 if una else None}


def reader_acc(eps, rnd=0):
    """Mean accuracy of the individual readers (round rnd), percent."""
    return float(np.mean([o.get("answer") == e["gold"] for e in eps for o in (e.get("rounds") or [[]])[rnd]])) * 100 if eps else None


def by_qid(eps):
    return {e["qid"]: e for e in eps}


def paired_diff(a, b):
    """Accuracy of a minus accuracy of b on the items both answered, percentage points, item bootstrap."""
    A, Bq = by_qid(a), by_qid(b)
    qs = sorted(set(A) & set(Bq))
    return boot(lambda u: float(np.mean(u)) * 100, [A[q]["correct"] - Bq[q]["correct"] for q in qs]) if qs else None


def boot_ratio(num, den, b=B):
    """Ratio of sums over items (x100), percentile bootstrap over items. num, den: arrays, or tuples of
    two arrays each for a difference of two ratios. -> [est, lo, hi]."""
    rng = np.random.default_rng(0)
    pair = isinstance(num, tuple)
    nums, dens = (num, den) if pair else ((num,), (den,))
    if any(d.sum() == 0 for d in dens):
        return None
    n = len(dens[0])
    idx = rng.integers(0, n, (b, n))
    est = [x.sum() / d.sum() * 100 for x, d in zip(nums, dens)]
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = [x[idx].sum(1) / d[idx].sum(1) * 100 for x, d in zip(nums, dens)]
    e, v = (est[0] - est[1], bs[0] - bs[1]) if pair else (est[0], bs[0])
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(e), 4), round(float(lo), 4), round(float(hi), 4)]


def paired_reader_vs(panel, single):
    """Mean accuracy of the panel's readers minus the single reader's accuracy, same items (pp)."""
    P, S = by_qid(panel), by_qid(single)
    qs = sorted(set(P) & set(S))
    return boot(lambda u: float(np.mean(u)) * 100,
                [float(np.mean([o.get("answer") == P[q]["gold"] for o in P[q]["rounds"][0]])) - S[q]["correct"] for q in qs]) if qs else None


def aggregation_gain(panel):
    """Majority-vote accuracy minus the mean accuracy of the same readers (pp, item bootstrap)."""
    return boot(lambda u: float(np.mean(u)) * 100,
                [e["correct"] - float(np.mean([o.get("answer") == e["gold"] for o in e["rounds"][0]])) for e in panel]) if panel else None


def panel_arithmetic(C):
    """Every 9-reader panel on yes/no items: observed majority accuracy against the beta-binomial
    prediction from the readers' mean accuracy and their error correlation (intraclass correlation phi)."""
    from scipy.stats import betabinom
    out = []
    for k, eps in C.items():
        m, s, cond, arch, n, mode, mem, per, sh = k
        if n != 9 or arch not in ("independent", "report") or s == "mxqa_mm_rad" or len(eps) < 150:
            continue
        R = np.array([[o.get("answer") == e["gold"] for o in e["rounds"][0]] for e in eps if len(e["rounds"][0]) == 9], float)
        pbar, ph = float(R.mean()), phi_from_matrix(1 - R)
        if ph is None or ph <= 0:
            continue
        a_, b_ = pbar * (1 - ph) / ph, (1 - pbar) * (1 - ph) / ph
        out.append({"model": m, "set": s, "cond": cond, "image_mode": mode, "memory": mem, "perceiver": per, "shared": sh,
                    "reader_mean": pbar * 100, "phi": ph, "majority": float(np.mean(R.sum(1) >= 5)) * 100,
                    "predicted": float(1 - betabinom.cdf(4, 9, a_, b_)) * 100})
    d = np.array([c["majority"] - c["predicted"] for c in out])
    return {"n_cells": len(out), "observed_minus_predicted_mean": float(d.mean()), "observed_minus_predicted_sd": float(d.std()),
            "observed_minus_predicted_max_abs": float(np.abs(d).max()), "cells": out}


def paired_reader_diff(a, b):
    """Mean reader accuracy of a minus that of b on the items both answered (pp, item bootstrap)."""
    A, Bq = by_qid(a), by_qid(b)
    qs = sorted(set(A) & set(Bq))
    ra = lambda e: float(np.mean([o.get("answer") == e["gold"] for o in e["rounds"][0]]))
    return boot(lambda u: float(np.mean(u)) * 100, [ra(A[q]) - ra(Bq[q]) for q in qs]) if qs else None


def conformity(dis, blank):
    """Readers correct in round 0 who end the discussion wrong, split by whether their round-0
    answer differed from the same reader's answer on the gray image (image-dependent) or not
    (prior-consistent); the same split within the round-0 majority and minority; and the reverse,
    readers wrong in round 0 who end right. Ratios of reader counts, item bootstrap."""
    D, Bk = by_qid(dis), by_qid(blank)
    rows = []                                      # per item: Counter over (group, start correct, side, end correct)
    for q in sorted(set(D) & set(Bk)):
        e = D[q]
        r0, rf, b0 = e["rounds"][0], e["rounds"][-1], Bk[q]["rounds"][0]
        maj = collections.Counter(o.get("answer") for o in r0).most_common(1)[0][0]
        c = collections.Counter()
        for o0, of, ob in zip(r0, rf, b0):
            grp = "image_dependent" if o0.get("answer") != ob.get("answer") else "prior_consistent"
            c[(grp, o0.get("answer") == e["gold"], o0.get("answer") == maj, of.get("answer") == e["gold"])] += 1
        rows.append(c)
    if not rows:
        return None

    def arr(fn):
        return np.array([sum(v for k, v in c.items() if fn(*k)) for c in rows], float)
    out = {"n_items": len(rows)}
    G = ("image_dependent", "prior_consistent")
    for start, lab in ((True, "right_to_wrong"), (False, "wrong_to_right")):
        num = {g: arr(lambda g_, c0, mj, cf, g=g: g_ == g and c0 == start and cf != start) for g in G}
        den = {g: arr(lambda g_, c0, mj, cf, g=g: g_ == g and c0 == start) for g in G}
        for g in G:
            out[f"{lab}_{g}"] = boot_ratio(num[g], den[g])
            out[f"n_{lab.split('_')[0]}_{g}"] = int(den[g].sum())
        out[f"{lab}_difference"] = boot_ratio((num[G[0]], num[G[1]]), (den[G[0]], den[G[1]]))
    # within the round-0 majority and within the minority: does an image-dependent correct answer
    # fail because it is in the minority, or also when it is in the majority?
    for side, lab in ((True, "majority"), (False, "minority")):
        num = {g: arr(lambda g_, c0, mj, cf, g=g: g_ == g and c0 and mj == side and not cf) for g in G}
        den = {g: arr(lambda g_, c0, mj, cf, g=g: g_ == g and c0 and mj == side) for g in G}
        for g in G:
            out[f"right_to_wrong_{lab}_{g}"] = boot_ratio(num[g], den[g])
            out[f"n_right_{lab}_{g}"] = int(den[g].sum())
        out[f"right_to_wrong_{lab}_difference"] = boot_ratio((num[G[0]], num[G[1]]), (den[G[0]], den[G[1]]))
    for side, lab in ((True, "majority"), (False, "minority")):        # pooled over the two groups
        num = arr(lambda g_, c0, mj, cf, side=side: c0 and mj == side and not cf)
        den = arr(lambda g_, c0, mj, cf, side=side: c0 and mj == side)
        out[f"right_to_wrong_{lab}"] = boot_ratio(num, den)
        out[f"n_right_{lab}"] = int(den.sum())
    for g in G:
        out[f"{g}_correct_in_minority_pct"] = boot_ratio(arr(lambda g_, c0, mj, cf, g=g: g_ == g and c0 and not mj),
                                                        arr(lambda g_, c0, mj, cf, g=g: g_ == g and c0))
    return out


# ---------------------------------------------------------------- experiment A
def experiment_A(rows):
    C = cells(rows)
    out = {}
    for m in MODELS:
        for s in ("vqarad", "slake", "mxqa_mm_rad"):
            g = lambda cond, arch, n, mode: C.get((m, s, cond, arch, n, mode, None, None, None), [])
            if not g("single", "cot", 1, "real"):
                continue
            d = {"n_items": len(g("single", "cot", 1, "real"))}
            # A0 / A1: accuracy and phi across evidence levels
            for mode in ("real", "res256", "res128", "blank"):
                e = {"single": boot(acc, g("single", "cot", 1, mode))}
                for n in (1, 3, 9):
                    e[f"panel{n}"] = boot(acc, g("panel", "independent", n, mode))
                p9 = g("panel", "independent", 9, mode)
                if p9:
                    e["phi"] = boot(lambda x: phi(x, 0), p9, b=500)
                    e["neff"] = round(9 / (1 + 8 * e["phi"][0]), 3) if e["phi"] else None
                    e["unanimity_round0"] = unanimity(p9, 0)
                    e["abstention_pct"] = abstention(p9, 0)
                    if e["abstention_pct"] > 0.5:
                        e["phi_answered"] = boot(lambda x: phi_answered(x, 0), p9, b=300)
                d[mode] = e
            dis9 = g("discussion", "discussion", 9, "real")
            if dis9:
                d["discussion"] = {"acc3": boot(acc, g("discussion", "discussion", 3, "real")), "acc9": boot(acc, dis9),
                                   "phi_last": boot(lambda x: phi(x, -1), dis9, b=500), "unanimity_final": unanimity(dis9, -1)}
                d["discussion"]["conformity"] = conformity(dis9, g("panel", "independent", 9, "blank"))
            # paired differences (same items): panels against the single reader, image against gray image
            d["paired"] = {
                "panel9_minus_single": paired_diff(g("panel", "independent", 9, "real"), g("single", "cot", 1, "real")),
                "disc9_minus_single": paired_diff(dis9, g("single", "cot", 1, "real")),
                "disc9_minus_panel9": paired_diff(dis9, g("panel", "independent", 9, "real")),
                "single_real_minus_blank": paired_diff(g("single", "cot", 1, "real"), g("single", "cot", 1, "blank")),
                "panel9_real_minus_blank": paired_diff(g("panel", "independent", 9, "real"), g("panel", "independent", 9, "blank")),
                # the panel's gain over the single reader = its readers' own accuracy + aggregation by majority
                "reader_mean_minus_single": paired_reader_vs(g("panel", "independent", 9, "real"), g("single", "cot", 1, "real")),
                "panel9_minus_reader_mean": aggregation_gain(g("panel", "independent", 9, "real"))}
            d["real"]["reader_mean"] = boot(reader_acc, g("panel", "independent", 9, "real"))
            # prior decomposition: the same reader slot with the real and the blank image
            R, Bk = by_qid(g("panel", "independent", 9, "real")), by_qid(g("panel", "independent", 9, "blank"))
            qs = sorted(set(R) & set(Bk))
            if qs:
                tab = collections.Counter(); cons = []
                for q in qs:
                    gold = R[q]["gold"]
                    ra = [o.get("answer") for o in R[q]["rounds"][0]]; ba = [o.get("answer") for o in Bk[q]["rounds"][0]]
                    for a, b_ in zip(ra, ba):
                        tab[(b_ == gold, a == gold)] += 1
                    if sum(a != gold for a in ra) >= 7:
                        mr = collections.Counter(ra).most_common(1)[0][0]; mb = collections.Counter(ba).most_common(1)[0][0]
                        cons.append(int(mr == mb))
                tot = sum(tab.values())
                d["prior_decomposition"] = {
                    "prior_right_stays_right": tab[(True, True)] / tot * 100,
                    "image_corrects_prior": tab[(False, True)] / tot * 100,
                    "image_misleads_prior": tab[(True, False)] / tot * 100,
                    "prior_wrong_stays_wrong": tab[(False, False)] / tot * 100,
                    "consensus_errors": len(cons),
                    "consensus_errors_matching_blank_majority_pct": float(np.mean(cons)) * 100 if cons else None,
                    "consensus_errors_matching_blank_majority": boot(lambda u: float(np.mean(u)) * 100, cons) if cons else None}
            out[f"{m}|{s}"] = d
    out["panel_arithmetic"] = panel_arithmetic(C)
    # A2: region occlusion (SLAKE-target)
    for m in MODELS:
        g = lambda cond, arch, n, mode: C.get((m, "slake_target", cond, arch, n, mode, None, None, None), [])
        R = by_qid(g("panel", "independent", 9, "real"))
        if not R:
            continue
        res = {}
        for lvl, arch, n, cond in (("single", "cot", 1, "single"), ("panel9", "independent", 9, "panel")):
            base = by_qid(g(cond, arch, n, "real"))
            for occ in ("occ_target", "occ_control"):
                O = by_qid(g(cond, arch, n, occ))
                qs = sorted(set(base) & set(O))
                if not qs:
                    continue
                if lvl == "panel9":       # reader-level flips, averaged within item
                    units = [float(np.mean([a.get("answer") != b_.get("answer") for a, b_ in
                                            zip(base[q]["rounds"][0], O[q]["rounds"][0])])) for q in qs]
                    maj = [int(base[q]["pred"] != O[q]["pred"]) for q in qs]
                    res[f"{lvl}_{occ}_reader_flip"] = boot(lambda u: float(np.mean(u)) * 100, units)
                    res[f"{lvl}_{occ}_majority_flip"] = boot(lambda u: float(np.mean(u)) * 100, maj)
                else:
                    units = [int(base[q]["pred"] != O[q]["pred"]) for q in qs]
                    res[f"{lvl}_{occ}_flip"] = boot(lambda u: float(np.mean(u)) * 100, units)
            # region dependence = target flip - control flip (paired by item)
            T, Cc, base = by_qid(g(cond, arch, n, "occ_target")), by_qid(g(cond, arch, n, "occ_control")), by_qid(g(cond, arch, n, "real"))
            qs = sorted(set(T) & set(Cc) & set(base))
            if qs:
                if lvl == "panel9":
                    u = [float(np.mean([a.get("answer") != t.get("answer") for a, t in zip(base[q]["rounds"][0], T[q]["rounds"][0])]))
                         - float(np.mean([a.get("answer") != c.get("answer") for a, c in zip(base[q]["rounds"][0], Cc[q]["rounds"][0])])) for q in qs]
                else:
                    u = [int(base[q]["pred"] != T[q]["pred"]) - int(base[q]["pred"] != Cc[q]["pred"]) for q in qs]
                res[f"{lvl}_region_dependence"] = boot(lambda x: float(np.mean(x)) * 100, u)
        out[f"{m}|slake_target"] = res
    # A3: same question, different image, opposite answers (SLAKE-pairs)
    for m in MODELS:
        g = lambda cond, arch, n: C.get((m, "slake_pairs", cond, arch, n, "real", None, None, None), [])
        P = by_qid(g("panel", "independent", 9)); S = by_qid(g("single", "cot", 1))
        if not P:
            continue
        pairs = collections.defaultdict(dict)
        for q in P:
            pid, side = q.split("/")[1].split("-")
            pairs[pid][side] = q
        full = [v for v in pairs.values() if "a" in v and "b" in v]
        res = {"n_pairs": len(full)}
        # option order is randomized per item, so the two answers are compared as option text, not letters
        txt = lambda q, letter: PAIR_OPTS[q].get(letter) if letter else None
        for lvl, src in (("single", S), ("panel9", P)):
            units = [(src[v["a"]], src[v["b"]]) for v in full if v["a"] in src and v["b"] in src]
            res[f"{lvl}_both_correct"] = boot(lambda u: float(np.mean([a["correct"] and b_["correct"] for a, b_ in u])) * 100, units)
            res[f"{lvl}_answers_differ"] = boot(lambda u: float(np.mean([txt(a["qid"], a["pred"]) != txt(b_["qid"], b_["pred"])
                                                                          for a, b_ in u])) * 100, units)
        # reader level: each slot answers both images
        units = [(P[v["a"]], P[v["b"]]) for v in full if v["a"] in P and v["b"] in P]
        res["reader_answers_differ"] = boot(lambda u: float(np.mean([np.mean([txt(a["qid"], x.get("answer")) != txt(b_["qid"], y.get("answer"))
                                                                              for x, y in zip(a["rounds"][0], b_["rounds"][0])])
                                                                     for a, b_ in u])) * 100, units)
        out[f"{m}|slake_pairs"] = res
    return out


# ---------------------------------------------------------------- experiment B
# Target terms for report coverage on SLAKE (locked 2026-10-19, before any report-pathway data):
# the organ or finding a question names, and the words a findings report would use for it.
TERMS = {
    "liver": ["liver", "hepatic"], "lung": ["lung", "pulmonary", "lobe"], "heart": ["heart", "cardiac", "cardio"],
    "kidney": ["kidney", "renal"], "spleen": ["spleen", "splenic"], "bladder": ["bladder"],
    "colon": ["colon", "colonic", "large bowel"], "stomach": ["stomach", "gastric"], "rectum": ["rectum", "rectal"],
    "esophagus": ["esophagus", "oesophagus", "esophageal"], "duodenum": ["duodenum", "duodenal"],
    "small bowel": ["small bowel", "small intestine", "jejun", "ileum", "ileal"], "spinal cord": ["spinal cord", "cord"],
    "trachea": ["trachea", "tracheal"], "larynx": ["larynx", "laryngeal"], "parotid": ["parotid"],
    "mandible": ["mandible", "mandibular"], "temporal lobe": ["temporal"], "brain stem": ["brainstem", "brain stem", "pons", "medulla"],
    "femoral head": ["femoral head", "femur"], "eye": ["eye", "orbit", "globe"], "tooth": ["tooth", "teeth", "dental"],
    "brain": ["brain", "cerebral", "cerebrum", "intracranial"],
    "abnormal": ["mass", "lesion", "nodule", "tumor", "tumour", "opacity", "effusion", "consolidation", "edema", "oedema",
                 "enlarg", "cardiomegaly", "pneumothorax", "atelectasis", "infiltrat", "abnormal", "hyperdense", "hypodense",
                 "hyperintens", "hypointens", "enhancing", "metasta", "cancer", "carcinoma"],
    "cancer": ["mass", "lesion", "tumor", "tumour", "nodule", "cancer", "carcinoma", "neoplasm", "metasta"],
    "effusion": ["effusion"], "pneumothorax": ["pneumothorax"], "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "pneumonia": ["pneumonia", "consolidation", "opacit", "infiltrat"], "nodule": ["nodule", "nodular", "mass"],
    "mass": ["mass", "lesion", "tumor", "tumour"], "atelectasis": ["atelectasis", "atelectatic"], "edema": ["edema", "oedema"],
}


def question_terms(q):
    """Terms the question asks about (organs and findings), in TERMS keys."""
    ql = q.lower().replace("brian", "brain")
    keys = [k for k in TERMS if k not in ("abnormal",) and k in ql]
    if any(w in ql for w in ("abnormal", "healthy", "normal", "disease", "lesion")):
        keys.append("abnormal")
    return keys


def covers(report, keys):
    rl = report.lower()
    return all(any(w in rl for w in TERMS[k]) for k in keys) if keys else None


def experiment_B(rows):
    C = cells(rows)
    out = {}
    items = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "data/slake_closed_250.jsonl")}
    for s in ("vqarad", "slake", "mxqa_mm_rad"):
        mat = {}
        for r_ in MODELS[:3]:
            direct = C.get((r_, s, "generic", "independent", 9, "real", None, None, None), [])
            row = {"direct": {"panel_acc": boot(acc, direct), "reader_acc": boot(reader_acc, direct),
                              "phi": boot(lambda x: phi(x, 0), direct, b=500)} if direct else None}
            for p_ in MODELS[:3]:
                for shared in (False, True):
                    eps = C.get((r_, s, "report", "report", 9, "real", None, p_, shared), [])
                    if not eps:
                        continue
                    ent = {"panel_acc": boot(acc, eps), "reader_acc": boot(reader_acc, eps),
                           "panel_acc_minus_image": paired_diff(eps, direct) if direct else None,
                           "reader_acc_minus_image": paired_reader_diff(eps, direct) if direct else None,
                           "phi": boot(lambda x: phi(x, 0), eps, b=500)}
                    if s == "slake":
                        wrong = [(e, i) for e in eps for i, o in enumerate(e["rounds"][0]) if o.get("answer") != e["gold"]]
                        cov = [covers(e["reports"][0 if shared else i], question_terms(items[e["qid"]]["stem"])) for e, i in wrong]
                        cov = [c for c in cov if c is not None]
                        allc = [covers(e["reports"][0 if shared else i], question_terms(items[e["qid"]]["stem"]))
                                for e in eps for i in range(len(e["rounds"][0]))]
                        allc = [c for c in allc if c is not None]
                        ent["errors_report_misses_target_pct"] = (1 - float(np.mean(cov))) * 100 if cov else None
                        ent["all_reports_miss_target_pct"] = (1 - float(np.mean(allc))) * 100 if allc else None
                    row[f"{p_}|{'shared' if shared else 'own'}"] = ent
            mat[r_] = row
        out[s] = mat
    return out


# ---------------------------------------------------------------- experiment C: independence budget
def reader_slots(C, s):
    """-> (qids, E items x slots error matrix, attrs list). One column per reader slot.
    perceiver: the model that looked at the image; reasoner: the model that chose the answer;
    channel: what the reasoner read, the image itself or a findings report of it."""
    cols, attrs = [], []

    def add(eps, attr_fn, cell, slots=range(9)):
        if not eps:
            return
        q = by_qid(eps)
        for i in slots:
            cols.append({k: int(v["rounds"][0][i].get("answer") != v["gold"]) for k, v in q.items() if len(v["rounds"][0]) > i})
            attrs.append(dict(attr_fn(i), cell=cell))
    for m in MODELS:
        img = dict(perceiver=m, reasoner=m, channel="image", render="orig", role="generic", memory="none", report=None)
        add(C.get((m, s, "panel", "independent", 9, "real", None, None, None)), lambda i, img=img: dict(img, role=f"spec{i}"),
            "specialty")
        add(C.get((m, s, "generic", "independent", 9, "real", None, None, None)), lambda i, img=img: dict(img), "generic")
        add(C.get((m, s, "render", "independent", 9, "real", None, None, None)),
            lambda i, img=img: dict(img, render=RENDER[i] or "orig"), "render",
            slots=range(3, 9))                      # slots 0-2 are the generic panel's own calls (cached)
        for mem in ("shared-image", "private-image", "shared-text", "private-text"):
            add(C.get((m, s, "memory", "independent", 9, "real", mem, None, None)),
                lambda i, img=img, mem=mem: dict(img, memory=f"{mem.split('-')[1]}:{0 if mem.startswith('shared') else i}"),
                f"memory:{mem}")
        for p_ in MODELS[:3]:
            for shared in (False, True):
                add(C.get((m, s, "report", "report", 9, "real", None, p_, shared)),
                    lambda i, m=m, p_=p_, shared=shared: dict(perceiver=p_, reasoner=m, channel="report", render="orig",
                                                              role="generic", memory="none",
                                                              report=f"{p_}:{0 if shared else i}"),
                    f"report:{p_}:{'shared' if shared else 'own'}")
    qids = sorted(set.intersection(*[set(c) for c in cols])) if cols else []
    E = np.array([[c[q] for c in cols] for q in qids], float)
    return qids, E, attrs


COMPONENTS = ["different perceiver", "image vs report", "different rendering", "different role",
              "different reference cases", "different report"]


def pair_design(attrs):
    """One row per reader pair. Columns: one intercept per reasoner pair (for a same-model pair, the
    phi of two readers that share everything; for a cross-model pair, the phi at the same perceiver),
    then one indicator per component the two readers do not share. Coding fixed on 2026-10-19,
    before any report-pathway data (docs/IMAGING_PROTOCOL.md section 12)."""
    P, X, pair_models = [], [], []
    for i, j in itertools.combinations(range(len(attrs)), 2):
        a, b = attrs[i], attrs[j]
        P.append((i, j))
        pair_models.append("~".join(sorted((a["reasoner"], b["reasoner"]))))
        X.append([a["perceiver"] != b["perceiver"],                # who looked at the image
                  a["channel"] != b["channel"],                    # one read the image, the other a report
                  a["render"] != b["render"],
                  a["role"] != b["role"],
                  a["memory"] != b["memory"],
                  a["channel"] == b["channel"] == "report" and a["report"] != b["report"]])
    keys = sorted(set(pair_models))
    F = np.array([[pm == k for k in keys] for pm in pair_models], float)
    return P, np.hstack([F, np.array(X, float)]), keys, pair_models


def pair_phi(E, P):
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(E.T)
    return np.array([R[i, j] for i, j in P])


def budget_fit(y, X):
    ok = np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta


def reasoner_effect(beta, keys):
    """Cross-model intercept minus the mean of the two same-model intercepts, per model pair and
    averaged over pairs: the change in phi when the two readers' answers come from different models."""
    a = dict(zip(keys, beta))
    out = {}
    for k in keys:
        m1, m2 = k.split("~")
        if m1 != m2 and f"{m1}~{m1}" in a and f"{m2}~{m2}" in a:
            out[k] = a[k] - (a[f"{m1}~{m1}"] + a[f"{m2}~{m2}"]) / 2
    if out:
        out["mean"] = float(np.mean(list(out.values())))
    return out


def pattern_means(y, X, pair_models, k0):
    """Mean pairwise phi for each reasoner pair and combination of non-shared components: the raw
    cells the regression summarizes."""
    out, keys = {}, collections.defaultdict(list)
    for r, (pm, row) in enumerate(zip(pair_models, X[:, k0:])):
        keys[pm + "|" + ("+".join(t for t, v in zip(COMPONENTS, row) if v) or "sampling only")].append(r)
    for k, rs in keys.items():
        v = y[rs]; v = v[np.isfinite(v)]
        out[k] = (float(np.mean(v)) if len(v) else None, len(v))
    return out


def contrast_groups(attrs, P):
    """Matched pair groups for the within-model contrasts: (group, model or model pair) -> pair indices."""
    G = collections.defaultdict(list)
    own = lambda c: c.startswith("report:") and c.endswith(":own")
    for k, (i, j) in enumerate(P):
        a, b = attrs[i], attrs[j]
        ca, cb = a["cell"], b["cell"]
        if a["reasoner"] == b["reasoner"]:
            r = a["reasoner"]
            if ca == cb == "generic":
                G[("generic", r)].append(k)
            if ca == cb == "specialty":
                G[("specialty", r)].append(k)
            if {ca, cb} <= {"generic", "render"} and a["render"] != b["render"]:
                G[("render", r)].append(k)
            if ca == cb and ca in ("memory:shared-image", "memory:shared-text"):
                G[("memory_shared", r)].append(k)
            if ca == cb and ca in ("memory:private-image", "memory:private-text"):
                G[("memory_private", r)].append(k)
            if ca == cb == f"report:{r}:shared":
                G[("report_shared", r)].append(k)
            if ca == cb == f"report:{r}:own":
                G[("report_own_self", r)].append(k)
            if ca == cb and own(ca):
                G[("report_own", r)].append(k)
            if ca != cb and own(ca) and own(cb):
                G[("two_perceivers", r)].append(k)
        else:
            pm = "~".join(sorted((a["reasoner"], b["reasoner"])))
            if ca == cb == "generic":
                G[("two_models", pm)].append(k)
            if a["channel"] == b["channel"] == "report" and a["report"] == b["report"]:
                G[("same_report_two_reasoners", pm)].append(k)
    return G


def contrasts(y, G):
    """Within-model contrasts of mean pairwise phi (no regression). -> {name: {key: value}}."""
    mu = {}
    for g, ks in G.items():
        v = y[ks]; v = v[np.isfinite(v)]
        if len(v):
            mu[g] = float(np.mean(v))
    out = collections.defaultdict(dict)
    models = sorted({k for (_, k) in mu if "~" not in k})
    pairs = sorted({k for (_, k) in mu if "~" in k})
    for r in models:
        g = lambda name: mu.get((name, r))
        for name, a, b_ in (("role", "specialty", "generic"), ("rendering", "render", "generic"),
                            ("reference cases", "memory_private", "memory_shared"),
                            ("report", "report_own_self", "report_shared"), ("perceiver", "two_perceivers", "report_own")):
            if g(a) is not None and g(b_) is not None:
                out[name][r] = g(a) - g(b_)
        if g("generic") is not None:
            out["sampling only"][r] = g("generic")
    for pm in pairs:
        r1, r2 = pm.split("~")
        for name, a, b_ in (("reasoner", "same_report_two_reasoners", "report_shared"), ("model", "two_models", "generic")):
            if (a, pm) in mu and (b_, r1) in mu and (b_, r2) in mu:
                out[name][pm] = mu[(a, pm)] - (mu[(b_, r1)] + mu[(b_, r2)]) / 2
    for name, d in out.items():
        if d and name != "sampling only":
            d["mean"] = float(np.mean([v for k, v in d.items() if k != "mean"]))
    return out


def experiment_C(rows, b=300):
    C = cells(rows)
    out = {}
    for s in ("vqarad", "slake", "mxqa_mm_rad"):
        qids, E, attrs = reader_slots(C, s)
        if not attrs:
            continue
        keep = E.std(0) > 1e-9
        E, attrs = E[:, keep], [a for a, k in zip(attrs, keep) if k]
        P, X, keys, pm = pair_design(attrs)
        k0 = len(keys)
        y = pair_phi(E, P)
        beta, pats = budget_fit(y, X), pattern_means(y, X, pm, k0)
        reff = reasoner_effect(beta[:k0], keys)
        G = contrast_groups(attrs, P)
        con, cboots = contrasts(y, G), collections.defaultdict(list)
        boots, rboots, pboots = [], collections.defaultdict(list), collections.defaultdict(list)
        rng_c = np.random.default_rng(0)
        for _ in range(b):
            yb = pair_phi(E[rng_c.integers(0, len(qids), len(qids))], P)
            bb = budget_fit(yb, X)
            boots.append(bb)
            for k, v in reasoner_effect(bb[:k0], keys).items():
                rboots[k].append(v)
            for k, (v, _n) in pattern_means(yb, X, pm, k0).items():
                if v is not None:
                    pboots[k].append(v)
            for name, d in contrasts(yb, G).items():
                for k, v in d.items():
                    cboots[(name, k)].append(v)
        boots = np.array(boots)
        ci = lambda v, bs: [round(float(v), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        out[s] = {"n_items": len(qids), "n_slots": len(attrs), "n_pairs": len(P),
                  "intercepts": {k: ci(beta[i], boots[:, i]) for i, k in enumerate(keys)},
                  "different_reasoner": {k: ci(v, rboots[k]) for k, v in reff.items()},
                  "terms": {t: ci(beta[k0 + k], boots[:, k0 + k]) for k, t in enumerate(COMPONENTS)},
                  "contrasts": {name: {k: ci(v, cboots[(name, k)]) for k, v in d.items() if cboots[(name, k)]}
                                for name, d in con.items()},
                  "contrast_n_pairs": {f"{g}|{k}": len(ks) for (g, k), ks in sorted(G.items())},
                  "patterns": {k: {"phi": ci(v, pboots[k]) if v is not None and pboots[k] else None, "n_pairs": n}
                               for k, (v, n) in sorted(pats.items())}}
        # the discussion transcript, separately: first-round vs final-round phi of Decentralized N = 9
        dis = {}
        for m in MODELS:
            eps = C.get((m, s, "discussion", "discussion", 9, "real", None, None, None))
            if eps:
                dis[m] = {"phi_round0": boot(lambda x: phi(x, 0), eps, b=300), "phi_final": boot(lambda x: phi(x, -1), eps, b=300)}
        out[s]["discussion"] = dis
    return out


# ---------------------------------------------------------------- memory accuracy (C4)
def experiment_memory(rows):
    C = cells(rows)
    out = {}
    for m in MODELS[:3]:
        for s in ("vqarad", "slake"):
            d = {"single_none": boot(acc, C.get((m, s, "single", "cot", 1, "real", None, None, None), [])),
                 "panel_none": boot(acc, C.get((m, s, "generic", "independent", 9, "real", None, None, None), [])),
                 "phi_none": boot(lambda x: phi(x, 0), C.get((m, s, "generic", "independent", 9, "real", None, None, None), []), b=500)}
            for mem in ("shared-image", "shared-text"):
                d[f"single_{mem}"] = boot(acc, C.get((m, s, "memory-single", "cot", 1, "real", mem, None, None), []))
            for mem in ("shared-image", "private-image", "shared-text", "private-text"):
                eps = C.get((m, s, "memory", "independent", 9, "real", mem, None, None), [])
                d[f"panel_{mem}"] = boot(acc, eps)
                d[f"reader_{mem}"] = boot(lambda x: float(np.mean([o.get("answer") == e["gold"] for e in x for o in e["rounds"][0]])) * 100, eps)
                d[f"phi_{mem}"] = boot(lambda x: phi(x, 0), eps, b=500)
            out[f"{m}|{s}"] = d
    return out


# ---------------------------------------------------------------- experiment D: open-weight mechanism
OW_ENCODER = {"medgemma-4b": "siglip", "huatuogpt-vision-7b": "clip-l14-336", "qwen3-vl-8b": "qwen3-vit",
              "qwen2-vl-7b": "qwen2-vit", "llava-1.5-7b": "clip-l14-336"}


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _reliance(pa, pb, opts):
    """Binary items: |delta log-odds of 'Yes'|; multi-option items: KL(pa || pb)."""
    yes = [k for k, v in opts.items() if v.lower() == "yes"]
    if yes and len(opts) == 2:
        return abs(_logit(pa[yes[0]]) - _logit(pb[yes[0]]))
    a = np.array([max(pa[k], 1e-9) for k in opts]); b_ = np.array([max(pb[k], 1e-9) for k in opts])
    return float(np.sum(a * np.log(a / b_)))


def experiment_D(rows=None):
    items = {}
    for f in ("vqarad_closed_250", "slake_closed_250", "mxqa_mm_rad_250", "slake_target_250", "slake_pairs_200"):
        for l in open(ROOT / f"data/{f}.jsonl"):
            it = json.loads(l); items[it["qid"]] = it
    ow = collections.defaultdict(dict)                        # (model, qid) -> {condition: record}
    for p_ in glob.glob(str(ROOT / "results/OW_*.jsonl")):
        for l in open(p_):
            r = json.loads(l); ow[(r["model"], r["qid"])][r["condition"]] = r
    models = sorted({k[0] for k in ow})
    out = {"models": models, "encoder": {m: OW_ENCODER.get(m) for m in models}}
    setof = lambda q: bench_of(items[q])
    # accuracy per condition, visual reliance, region reliance
    for m in models:
        for s in ("vqarad", "slake", "mxqa_mm_rad", "slake_target"):
            qs = [q for (mm, q) in ow if mm == m and setof(q) == s]
            if not qs:
                continue
            d = {}
            for cond in ("real", "blank", "res256", "res128", "gamma0.7", "clahe", "occ_target", "occ_control"):
                v = [ow[(m, q)][cond]["correct"] for q in qs if cond in ow[(m, q)]]
                if v:
                    d[f"acc_{cond}"] = boot(lambda u: float(np.mean(u)) * 100, v)
            vr = [_reliance(ow[(m, q)]["real"]["probs"], ow[(m, q)]["blank"]["probs"], items[q]["options"])
                  for q in qs if "real" in ow[(m, q)] and "blank" in ow[(m, q)]]
            if vr:
                d["visual_reliance_median"] = float(np.median(vr))
                d["visual_reliance"] = boot(lambda u: float(np.mean(u)), vr)
            if s == "slake_target":
                rr = [_reliance(ow[(m, q)]["real"]["probs"], ow[(m, q)]["occ_target"]["probs"], items[q]["options"])
                      - _reliance(ow[(m, q)]["real"]["probs"], ow[(m, q)]["occ_control"]["probs"], items[q]["options"])
                      for q in qs if {"real", "occ_target", "occ_control"} <= set(ow[(m, q)])]
                if rr:
                    d["region_reliance"] = boot(lambda u: float(np.mean(u)), rr)
                    d["region_reliance_positive_pct"] = float(np.mean([x > 0 for x in rr])) * 100
                # the API arm's measure: share of answers that change under each occlusion
                trip = [(ow[(m, q)]["real"]["pred"], ow[(m, q)]["occ_target"]["pred"], ow[(m, q)]["occ_control"]["pred"])
                        for q in qs if {"real", "occ_target", "occ_control"} <= set(ow[(m, q)])]
                if trip:
                    d["flip_target"] = boot(lambda u: float(np.mean([a != b_ for a, b_, _ in u])) * 100, trip)
                    d["flip_control"] = boot(lambda u: float(np.mean([a != c for a, _, c in u])) * 100, trip)
                    d["region_dependence_flip"] = boot(lambda u: float(np.mean([(a != b_) - (a != c) for a, b_, c in u])) * 100, trip)
            out[f"{m}|{s}"] = d
    # SLAKE-pairs: the same question on two images with opposite reference answers
    for m in models:
        P = {q: ow[(mm, q)]["real"] for (mm, q) in ow if mm == m and setof(q) == "slake_pairs" and "real" in ow[(mm, q)]}
        pairs = collections.defaultdict(dict)
        for q in P:
            pid, side = q.split("/")[1].split("-")
            pairs[pid][side] = q
        units = [(P[v["a"]], P[v["b"]]) for v in pairs.values() if "a" in v and "b" in v]
        if units:
            out[f"{m}|slake_pairs"] = {
                "n_pairs": len(units),
                "both_correct": boot(lambda u: float(np.mean([a["correct"] and b_["correct"] for a, b_ in u])) * 100, units),
                "answers_differ": boot(lambda u: float(np.mean([items[a["qid"]]["options"][a["pred"]] !=
                                                                 items[b_["qid"]]["options"][b_["pred"]] for a, b_ in u])) * 100, units)}
    # error correlation between models, by shared vision encoder
    pairs = {}
    for s in ("vqarad", "slake", "mxqa_mm_rad"):
        for a, b_ in itertools.combinations(models, 2):
            qs = sorted({q for (mm, q) in ow if mm == a and setof(q) == s and "real" in ow[(a, q)]}
                        & {q for (mm, q) in ow if mm == b_ and setof(q) == s and "real" in ow[(b_, q)]})
            if len(qs) < 50:
                continue
            x = np.array([1 - ow[(a, q)]["real"]["correct"] for q in qs], float)
            y = np.array([1 - ow[(b_, q)]["real"]["correct"] for q in qs], float)
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            pa, pb = x.mean(), y.mean()          # phi_max for unequal error rates
            pmax = np.sqrt(min(pa, pb) * (1 - max(pa, pb)) / (max(pa, pb) * (1 - min(pa, pb))))
            pairs[f"{s}|{a}|{b_}"] = {"phi": round(r, 4), "phi_over_max": round(r / pmax, 4) if pmax > 0 else None,
                                      "shared_encoder": OW_ENCODER.get(a) == OW_ENCODER.get(b_)}
    out["cross_model_pairs"] = pairs
    # link to the API panels: are API consensus errors the items open models read least?
    if rows is not None:
        C = cells(rows)
        link = {}
        for s in ("vqarad", "slake", "mxqa_mm_rad"):
            cons = collections.defaultdict(list)
            for m in MODELS:
                for e in C.get((m, s, "panel", "independent", 9, "real", None, None, None), []):
                    cons[e["qid"]].append(int(sum(o.get("answer") != e["gold"] for o in e["rounds"][0]) >= 7))
            vr = {}
            for q in cons:
                v = [_reliance(ow[(m, q)]["real"]["probs"], ow[(m, q)]["blank"]["probs"], items[q]["options"])
                     for m in models if (m, q) in ow and {"real", "blank"} <= set(ow[(m, q)])]
                if v:
                    vr[q] = float(np.mean(v))
            qs = sorted(set(cons) & set(vr))
            if len(qs) < 30:
                continue
            from scipy.stats import spearmanr
            cfrac = [float(np.mean(cons[q])) for q in qs]
            rho = spearmanr(cfrac, [vr[q] for q in qs]).statistic
            e_, n_ = [vr[q] for q in qs if np.mean(cons[q]) > 0], [vr[q] for q in qs if np.mean(cons[q]) == 0]

            def rho_of(u):
                x, y = [a for a, _ in u], [b_ for _, b_ in u]
                if len(set(x)) < 2 or len(set(y)) < 2:
                    return None
                return float(spearmanr(x, y).statistic)
            link[s] = {"spearman_consensus_vs_reliance": round(float(rho), 4),
                       "spearman_ci": boot(rho_of, list(zip(cfrac, [vr[q] for q in qs]))), "n_items": len(qs),
                       "reliance_on_consensus_error_items": float(np.median(e_)) if e_ else None,
                       "reliance_on_other_items": float(np.median(n_)) if n_ else None}
        out["api_link"] = link
    return out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "A"
    rows = load()
    print(f"{len(rows)} labelled episodes")
    fn = {"A": experiment_A, "B": experiment_B, "C": experiment_C, "M": experiment_memory, "D": experiment_D}[what]
    res = fn(rows)
    (ROOT / f"results/imaging_v2_{what}.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, json.dumps(v)[:600])


if __name__ == "__main__":
    main()
