"""Shared data access for the agent-scaling candidates S1a-S1c (round 3).

Reads results/cvpr_nmi.json (experiments/cvpr_fig/extract_nmi.py), key "scaling": for every
"model|set" with a complete Independent MAS of 9 on original images, the majority accuracy of the
first k agents (k = 1, 3, 5, 7, 9), and on the yes/no sets the binomial (independent agents with the
same mean accuracy) and beta-binomial (observed error correlation phi) predictions, N_eff(k) and phi.

Model styles: cf_style.MODEL for the four models of the study. Capability-ladder models that appear
later in the same JSON get a provisional style keyed by model name (LADDER_STYLE): the family's hue,
lighter for the smaller rung and darker for the larger one, with the family marker. Replace it with
cf_style entries once the ladder is part of the shared style module.
"""
import json
import pathlib

from cf_style import MODEL, PAL                                      # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCALING = json.loads((ROOT / "results" / "cvpr_nmi.json").read_text())["scaling"]
SETS = ["vqarad", "slake", "mxqa_mm_rad"]
SET_LABEL = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM"}
YESNO = ["vqarad", "slake"]
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


def models():
    """Models present in the scaling block, in family and capability order."""
    present = {k.split("|")[0] for k in SCALING}
    return [m for m in ORDER if m in present] + sorted(present - set(ORDER))


def cell(m, s):
    return SCALING.get(f"{m}|{s}")
