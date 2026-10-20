"""Shared data access for the round-3 candidates S2a/S2b (capability vs multi-agent gain) and
S5c/S5d (language prior).

S2: results/cvpr_nmi.json (experiments/cvpr_fig/extract_nmi.py), key "arch": per "model|set" the SAS
accuracy [est, lo, hi], "<Arch>|<N>" MAS accuracy and "<Arch>|<N>|gain" paired gain over SAS (pp,
item bootstrap) for Independent, Centralized, Decentralized and Hybrid. Whatever architectures and
models are present are drawn; cells still running are absent from the JSON.
S5: results/imaging_v2_A.json, per "model|set": single-agent accuracy with the original ("real") and
the blank image, item-bootstrap 95% CIs.

Model styles: cf_style.MODEL for the four models of the study. Capability-ladder models get a
provisional style keyed by name (LADDER_STYLE, the same values as the S1 candidates): the family hue,
lighter for the smaller rung and darker for the larger one, with the family marker.
"""
import json
import pathlib

import numpy as np

from cf_style import MODEL, PAL                                      # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[3]
SETS = ["vqarad", "slake", "mxqa_mm_rad"]
SET_LABEL = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM"}
SET_MARKER = {"vqarad": "o", "slake": "s", "mxqa_mm_rad": "^"}
CHANCE = {"vqarad": 50.0, "slake": 50.0, "mxqa_mm_rad": 20.0}
ARCHS = ["Independent", "Centralized", "Decentralized", "Hybrid"]
ARCH_MARKER = {"Independent": "o", "Centralized": "D", "Decentralized": "s", "Hybrid": "^"}
LADDER_STYLE = {
    "gpt-5-nano":            {"label": "GPT-5 nano",            "color": "#8F97CC", "marker": "o"},
    "gpt-5":                 {"label": "GPT-5",                 "color": "#262C60", "marker": "o"},
    "gemini-3.5-flash-lite": {"label": "Gemini 3.5 Flash-Lite", "color": "#8ED5CA", "marker": "s"},
    "claude-sonnet-5":       {"label": "Claude Sonnet 5",       "color": "#7E5309", "marker": "D"},
    "qwen3.8-max-0902":      {"label": "Qwen 3.8 Max",          "color": "#73203D", "marker": "^"},
}
ORDER = ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gemini-3.5-flash-lite", "gemini-3.7-flash",
         "claude-haiku-4-5-20251001", "claude-sonnet-5", "qwen3.8-flash", "qwen3.8-max-0902"]


def style(m):
    return MODEL.get(m) or LADDER_STYLE.get(m) or {"label": m, "color": PAL["neutral_mid"], "marker": "o"}


def ordered(models):
    models = set(models)
    return [m for m in ORDER if m in models] + sorted(models - set(ORDER))


def gain_points(n=9):
    """[(model, set, arch, sas_est, gain [est, lo, hi])] for every gain present at N = n."""
    A = json.loads((ROOT / "results" / "cvpr_nmi.json").read_text())["arch"]
    pts = []
    for key, cell in A.items():
        m, s = key.split("|")
        for a in ARCHS:
            g = cell.get(f"{a}|{n}|gain")
            if g is not None and cell.get("SAS"):
                pts.append((m, s, a, cell["SAS"][0], g))
    return pts


def ols_slope_ci(pts, b=2000, seed=0):
    """OLS slope of gain on SAS accuracy; percentile CI from resampling systems (model x set), which
    keeps every architecture of a sampled system together. -> (slope, intercept, lo, hi) or None."""
    if len(pts) < 3:
        return None
    x = np.array([p[3] for p in pts]); y = np.array([p[4][0] for p in pts])
    if np.ptp(x) < 1e-9:
        return None
    slope, icpt = np.polyfit(x, y, 1)
    systems = sorted({(p[0], p[1]) for p in pts})
    idx = {s: [i for i, p in enumerate(pts) if (p[0], p[1]) == s] for s in systems}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        take = [i for j in rng.integers(0, len(systems), len(systems)) for i in idx[systems[j]]]
        xs, ys = x[take], y[take]
        if np.ptp(xs) > 1e-9:
            vals.append(np.polyfit(xs, ys, 1)[0])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(slope), float(icpt), float(lo), float(hi)


def prior_points():
    """[(model, set, original [est, lo, hi], blank [est, lo, hi])] from experiment A."""
    A = json.loads((ROOT / "results" / "imaging_v2_A.json").read_text())
    pts = []
    for key, cell in A.items():
        if "|" not in key:
            continue
        m, s = key.split("|")
        if s in SETS and cell.get("real", {}).get("single") and cell.get("blank", {}).get("single"):
            pts.append((m, s, cell["real"]["single"], cell["blank"]["single"]))
    return pts


def fmt(v, nd=2):
    return f"{v:.{nd}f}".replace("-", "−")
