"""Specialty roster + role prompts + the cheap router that ranks specialties per item."""
from __future__ import annotations
import json, re
from common.llm import chat

# Fixed 17-specialty clinical roster (superset covering MedXpertQA's 12 body systems).
ROSTER = [
    "internal medicine", "cardiology", "neurology", "pulmonology", "gastroenterology",
    "nephrology", "endocrinology", "infectious disease", "hematology-oncology",
    "rheumatology", "general surgery", "orthopedic surgery", "obstetrics and gynecology",
    "pediatrics", "psychiatry", "dermatology", "emergency medicine",
]
_NORM = {s.lower(): s for s in ROSTER}

ROUTER_MODEL = "gpt-4.1-nano"

# Deterministic fallback ordering when the router fails / is disabled.
_BY_SYSTEM = {
    "Nervous": ["neurology", "psychiatry", "internal medicine", "emergency medicine", "general surgery"],
    "Skeletal": ["orthopedic surgery", "rheumatology", "internal medicine", "general surgery", "emergency medicine"],
    "Cardiovascular": ["cardiology", "internal medicine", "emergency medicine", "pulmonology", "nephrology"],
    "Digestive": ["gastroenterology", "general surgery", "internal medicine", "infectious disease", "hematology-oncology"],
    "Reproductive": ["obstetrics and gynecology", "internal medicine", "endocrinology", "general surgery", "infectious disease"],
    "Respiratory": ["pulmonology", "internal medicine", "infectious disease", "emergency medicine", "cardiology"],
    "Muscular": ["rheumatology", "neurology", "orthopedic surgery", "internal medicine", "emergency medicine"],
    "Endocrine": ["endocrinology", "internal medicine", "nephrology", "general surgery", "pediatrics"],
    "Urinary": ["nephrology", "internal medicine", "general surgery", "infectious disease", "emergency medicine"],
    "Lymphatic": ["hematology-oncology", "infectious disease", "internal medicine", "rheumatology", "general surgery"],
    "Integumentary": ["dermatology", "internal medicine", "infectious disease", "rheumatology", "general surgery"],
}
_DEFAULT_TAIL = ["internal medicine", "emergency medicine", "cardiology", "neurology",
                 "infectious disease", "general surgery", "pulmonology", "gastroenterology",
                 "hematology-oncology", "endocrinology", "nephrology", "pediatrics",
                 "psychiatry", "rheumatology", "obstetrics and gynecology",
                 "orthopedic surgery", "dermatology"]

ROUTER_SYS = (
    "You triage clinical exam questions to a diagnostic panel. Given a question, rank the 9 most "
    "relevant medical specialties, most relevant first, choosing ONLY from this list:\n"
    + ", ".join(ROSTER) +
    '\nReply with JSON only: {"specialties": ["...", ... 9 items ...]}'
)


def _fallback(item) -> list[str]:
    seed = _BY_SYSTEM.get(item.get("meta", {}).get("body_system"), [])
    out = list(seed)
    for s in _DEFAULT_TAIL:
        if s not in out:
            out.append(s)
    return out[:9]


def route(item, use_router=True, model=ROUTER_MODEL) -> list[str]:
    """Ordered roster of 9 specialties for this item. Cached; a panel of size N = first N."""
    if not use_router:
        return _fallback(item)
    q = item["stem"][:2500]
    try:
        r = chat(model, [{"role": "user", "content": f"Question:\n{q}"}], system=ROUTER_SYS,
                 temperature=0.0, seed=7, max_tokens=200, json_mode=True, tag="router")
        got = json.loads(r["text"]).get("specialties", [])
    except Exception:
        got = []
    out = []
    for s in got:
        s2 = _NORM.get(str(s).strip().lower())
        if s2 and s2 not in out:
            out.append(s2)
    for s in _fallback(item):
        if len(out) >= 9:
            break
        if s not in out:
            out.append(s)
    return out[:9]


def role_system(specialty: str, generic: bool = False) -> str:
    who = "attending general internist" if generic else f"attending {specialty} physician"
    return (f"You are an {who} on a hospital diagnostic panel. "
            "Answer the multiple-choice clinical question using your specialty expertise. "
            "Be concise and decisive.")
