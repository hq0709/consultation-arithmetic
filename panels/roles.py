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


# 影像题（item["domain"] == "radiology"）用放射科亚专科组成读片小组，路由器同样只看题干。
RADIOLOGY_ROSTER = [
    "neuroradiology", "head and neck radiology", "thoracic radiology", "cardiovascular radiology",
    "abdominal radiology", "genitourinary radiology", "musculoskeletal radiology", "breast imaging",
    "pediatric radiology", "nuclear medicine", "interventional radiology", "emergency radiology",
]
_RAD_NORM = {s.lower(): s for s in RADIOLOGY_ROSTER}

ROUTER_SYS_RAD = (
    "You triage imaging questions to a radiology reading panel. Given a question about a medical "
    "image, rank the 9 most relevant radiology subspecialties, most relevant first, choosing ONLY "
    "from this list:\n" + ", ".join(RADIOLOGY_ROSTER) +
    '\nReply with JSON only: {"specialties": ["...", ... 9 items ...]}'
)


def is_radiology(item) -> bool:
    return item.get("domain") == "radiology"


def generalist(item) -> str:
    """The single reader's specialty: a general internist for text items, a general radiologist
    for imaging items (role_system turns it into the generic prompt)."""
    return "radiology" if is_radiology(item) else "internal medicine"


def _fallback(item) -> list[str]:
    if is_radiology(item):
        return RADIOLOGY_ROSTER[:9]
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
    rad = is_radiology(item)
    try:
        r = chat(model, [{"role": "user", "content": f"Question:\n{q}"}],
                 system=ROUTER_SYS_RAD if rad else ROUTER_SYS,
                 temperature=0.0, seed=7, max_tokens=200, json_mode=True, tag="router")
        got = json.loads(r["text"]).get("specialties", [])
    except Exception:
        got = []
    out = []
    norm = _RAD_NORM if rad else _NORM
    for s in got:
        s2 = norm.get(str(s).strip().lower())
        if s2 and s2 not in out:
            out.append(s2)
    for s in _fallback(item):
        if len(out) >= 9:
            break
        if s not in out:
            out.append(s)
    return out[:9]


def role_system(specialty: str, generic: bool = False) -> str:
    if specialty == "radiology" or specialty in _RAD_NORM.values():
        who = ("attending general radiologist" if generic
               else f"attending radiologist specializing in {specialty}")
        return (f"You are an {who} on a hospital imaging reading panel. "
                "Answer the multiple-choice question about the attached image using your expertise. "
                "Be concise and decisive.")
    who = "attending general internist" if generic else f"attending {specialty} physician"
    return (f"You are an {who} on a hospital diagnostic panel. "
            "Answer the multiple-choice clinical question using your specialty expertise. "
            "Be concise and decisive.")
