"""Shared data access and styles for the round-3 candidates (cost-accuracy, debate dynamics).

Data: results/cvpr_nmi.json (experiments/cvpr_fig/extract_nmi.py). Only systems present in the file
are drawn, so the same scripts pick up Centralized, Hybrid and the capability-ladder models once their
runs finish.

Model styles: the four study models keep cf_style.MODEL (colour and marker). The ladder models take
their family's marker and hue, lighter for the lower rung and darker for the higher rung, so a family
reads as one hue ordered by capability:
  OpenAI    gpt-5-nano (light) < gpt-5-mini (MODEL) < gpt-5 (dark)        circle
  Google    gemini-3.5-flash-lite (light) < gemini-3.7-flash (MODEL)       square
  Anthropic claude-haiku-4-5 (MODEL) < claude-sonnet-5 (dark)              diamond
  Qwen      qwen3.8-flash (MODEL) < qwen3.8-max-0902 (dark)                triangle
"""
import json
import pathlib

from cf_style import MODEL

ROOT = pathlib.Path(__file__).resolve().parents[3]
NMI = json.loads((ROOT / "results" / "cvpr_nmi.json").read_text())

LADDER_STYLE = {
    "gpt-5-nano":            {"label": "GPT-5 nano",            "color": "#9AA2D2", "marker": "o"},
    "gpt-5":                 {"label": "GPT-5",                 "color": "#262D5E", "marker": "o"},
    "gemini-3.5-flash-lite": {"label": "Gemini 3.5 Flash-Lite", "color": "#93D6CC", "marker": "s"},
    "claude-sonnet-5":       {"label": "Claude Sonnet 5",       "color": "#7E5410", "marker": "D"},
    "qwen3.8-max-0902":      {"label": "Qwen 3.8 Max",          "color": "#74233D", "marker": "^"},
}
STYLE = dict(MODEL)
STYLE.update(LADDER_STYLE)
# family order, then capability rung
ORDER_ALL = ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gemini-3.5-flash-lite", "gemini-3.7-flash",
             "claude-haiku-4-5-20251001", "claude-sonnet-5", "qwen3.8-flash", "qwen3.8-max-0902"]
ARCHS = ["SAS", "Independent", "Centralized", "Decentralized", "Hybrid"]
SETS = ["vqarad", "slake", "mxqa_mm_rad"]
SET_LABEL = {"vqarad": "VQA-RAD", "slake": "SLAKE", "mxqa_mm_rad": "MedXpertQA-MM"}


def present(block):
    """Models with at least one system in NMI[block], in family / capability order."""
    return [m for m in ORDER_ALL if any(k.startswith(m + "|") for k in NMI[block])]
