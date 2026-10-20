"""Per-system quantities for the NMI-style displays of the CVPR paper (results/cvpr_nmi.json).

  python experiments/cvpr_fig/extract_nmi.py

Architectures follow Kim et al. (NMI 2026): SAS (cot), Independent, Centralized, Decentralized
(discussion) and Hybrid (tiered referral), original images, on VQA-RAD, SLAKE and MedXpertQA-MM.
Models: the four models of the study plus the within-family capability ladder
(imaging_v2_analysis.LADDER). Cells with fewer than MIN_ITEMS items are left out (runs in progress).

- arch:    accuracy with item-bootstrap 95% CI for each architecture at N = 3 and 9, and the paired
           gain over SAS (pp)
- scaling: majority accuracy of the first k agents of the Independent MAS of 9 (k = 1, 3, 5, 7, 9);
           on yes/no sets also the binomial (independent agents) and beta-binomial (observed phi)
           predictions and N_eff(k) = k / (1 + (k - 1) phi)
- cost:    nominal input, output and reasoning tokens and USD per item (cached calls priced too)
- debate:  per round of the Decentralized MAS of 9, majority accuracy, unanimity and phi
"""
import collections
import json
import pathlib
import sys

import numpy as np
from scipy.stats import betabinom, binom

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import imaging_v2_analysis as ia                                    # noqa: E402

SETS = ["vqarad", "slake", "mxqa_mm_rad"]
MODELS = ia.MODELS + ia.LADDER
ARCH = {"SAS": ("single", "cot", 1), "Independent": ("panel", "independent", None),
        "Centralized": ("centralized", "centralized", None), "Decentralized": ("discussion", "discussion", None),
        "Hybrid": ("hybrid", "tiered", None)}
MIN_ITEMS = 240


def plurality(answers, rank):
    """Most frequent non-empty answer; ties go to the answer given first (deterministic)."""
    c = collections.Counter(a for a in answers if a)
    if not c:
        return None
    top = max(c.values())
    return next(a for a in rank if a and c[a] == top)


def main():
    rows = ia.load()
    C = ia.cells(rows)
    get = lambda m, s, cond, arch, n: C.get((m, s, cond, arch, n, "real", None, None, None), [])
    out = {"arch": {}, "scaling": {}, "cost": {}, "debate": {}, "models": MODELS, "min_items": MIN_ITEMS}
    for m in MODELS:
        for s in SETS:
            sas = get(m, s, "single", "cot", 1)
            if len(sas) < MIN_ITEMS:
                continue
            key = f"{m}|{s}"
            A = out["arch"][key] = {"n_items": len(sas), "SAS": ia.boot(ia.acc, sas)}
            for name, (cond, arch, _) in ARCH.items():
                if name == "SAS":
                    continue
                for n in (3, 9):
                    eps = get(m, s, cond, arch, n)
                    if len(eps) < MIN_ITEMS:
                        continue
                    A[f"{name}|{n}"] = ia.boot(ia.acc, eps)
                    A[f"{name}|{n}|gain"] = ia.paired_diff(eps, sas)
            # cost per item (nominal) for every architecture at N = 9 and SAS
            Cst = out["cost"][key] = {}
            for name, (cond, arch, n0) in ARCH.items():
                eps = get(m, s, cond, arch, n0 or 9)
                if len(eps) < MIN_ITEMS:
                    continue
                c = [e.get("cost") or {} for e in eps]
                Cst[name] = {k: float(np.mean([x.get(k, 0) for x in c])) for k in ("calls", "in_tok", "out_tok", "rsn_tok", "usd")}
                Cst[name]["acc"] = ia.acc(eps)
            # agent scaling from the Independent MAS of 9
            ind = [e for e in get(m, s, "panel", "independent", 9) if len(e["rounds"][0]) == 9]
            if len(ind) >= MIN_ITEMS:
                R = np.array([[o.get("answer") == e["gold"] for o in e["rounds"][0]] for e in ind], float)
                ph = ia.phi_from_matrix(1 - R)
                S = out["scaling"][key] = {"phi": ph, "mean_agent": float(R.mean()) * 100, "k": [], "observed": [],
                                           "binomial": [], "betabinomial": [], "neff": []}
                for k in (1, 3, 5, 7, 9):
                    if s == "mxqa_mm_rad":
                        obs = np.mean([plurality([o.get("answer") for o in e["rounds"][0][:k]],
                                                 [o.get("answer") for o in e["rounds"][0][:k]]) == e["gold"] for e in ind])
                    else:
                        obs = np.mean(R[:, :k].sum(1) >= (k + 1) / 2)
                    pbar = float(R[:, :k].mean())
                    S["k"].append(k); S["observed"].append(float(obs) * 100)
                    S["neff"].append(k / (1 + (k - 1) * ph) if ph else None)
                    if s != "mxqa_mm_rad" and ph and ph > 0:
                        a_, b_ = pbar * (1 - ph) / ph, (1 - pbar) * (1 - ph) / ph
                        thr = (k + 1) // 2
                        S["binomial"].append(float(1 - binom.cdf(thr - 1, k, pbar)) * 100)
                        S["betabinomial"].append(float(1 - betabinom.cdf(thr - 1, k, a_, b_)) * 100)
            # debate dynamics, Decentralized MAS of 9
            dis = [e for e in get(m, s, "discussion", "discussion", 9) if e.get("rounds") and len(e["rounds"][0]) == 9]
            if len(dis) >= MIN_ITEMS:
                Dd = out["debate"][key] = {"round": [], "accuracy": [], "unanimity": [], "phi": [], "mean_agent": []}
                for r in range(4):
                    # a debate that stopped early keeps its last round
                    ans = [[o.get("answer") for o in e["rounds"][min(r, len(e["rounds"]) - 1)]] for e in dis]
                    gold = [e["gold"] for e in dis]
                    maj = [plurality(a, a) == g for a, g in zip(ans, gold)]
                    E = np.array([[x != g for x in a] for a, g in zip(ans, gold)], float)
                    Dd["round"].append(r); Dd["accuracy"].append(float(np.mean(maj)) * 100)
                    Dd["unanimity"].append(float(np.mean([len(set(a)) == 1 and a[0] is not None for a in ans])) * 100)
                    Dd["phi"].append(ia.phi_from_matrix(E)); Dd["mean_agent"].append(float(1 - E.mean()) * 100)
    p = ROOT / "results" / "cvpr_nmi.json"
    p.write_text(json.dumps(out))
    print(p, {k: len(v) for k, v in out.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
