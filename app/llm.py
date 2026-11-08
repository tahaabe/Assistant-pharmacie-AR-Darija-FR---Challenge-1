"""LLM fallback (Groq, OpenAI-compatible chat API).

Used ONLY when the rule-based NLU is not confident. The LLM never answers the patient
directly and never decides which medication exists: it returns
  * an intent (from the fixed list, or "out_of_scope"),
  * a short canonical French rewrite of the request,
and the rewrite is fed back through the same rule-based extractor, so every medication
is still grounded in the CNOPS catalog (an unknown drug stays unknown).

Env: GROQ_API_KEY (required to enable), GROQ_LLM_MODEL (default llama-3.3-70b-versatile),
     LLM_FALLBACK=0 to disable, LLM_CONFIDENCE_THRESHOLD (default 0.45).
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx

MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
THRESHOLD = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.45"))
URL = "https://api.groq.com/openai/v1/chat/completions"
INTENTS = ["order_medication", "check_availability", "ask_price", "ask_reimbursement", "ask_alternative",
           "find_pharmacy", "pharmacy_hours", "pharmacy_contact", "greeting", "out_of_scope"]

SYSTEM = f"""You are the NLU component of a Moroccan pharmacy assistant. Patients write in Moroccan Darija
(Arabic script or Latin "Arabizi" with digits 3/7/9), Modern Standard Arabic, French, or a mix.
Return ONLY a JSON object:
{{"intent": one of {INTENTS},
  "language": "darija" | "darija_latin" | "ar" | "fr",
  "canonical_fr": short French rewrite of the request keeping drug names, dosages, quantities, forms,
                  city and pharmacy names exactly as the patient spelled them (do not correct or invent drug names),
  "medications": [list of drug names as written by the patient],
  "confidence": number between 0 and 1}}
Rules: "out_of_scope" = medical advice (dosage to take, side effects, diagnosis, pregnancy...) or anything
unrelated to ordering/checking medications or finding pharmacies. Never add a drug the patient did not mention.

Examples:
"3tini chi haja l ras dyali kayderni" -> {{"intent":"out_of_scope","language":"darija_latin","canonical_fr":"j'ai mal à la tête, que dois-je prendre ?","medications":[],"confidence":0.9}}
"sme7li, n9dar nl9a 3ndkom dak dwa smito spasfon?" -> {{"intent":"check_availability","language":"darija_latin","canonical_fr":"avez-vous du spasfon ?","medications":["spasfon"],"confidence":0.9}}
"عافاك وجد ليا داك الدوا ديال السكر غلوكوفاج 850 جوج" -> {{"intent":"order_medication","language":"darija","canonical_fr":"je veux commander 2 boîtes de glucophage 850","medications":["glucophage"],"confidence":0.9}}
"chkoun li m7loul had lila f fes" -> {{"intent":"pharmacy_hours","language":"darija_latin","canonical_fr":"pharmacie de garde à fès ce soir","medications":[],"confidence":0.85}}
"""


def enabled() -> bool:
    return bool(os.getenv("GROQ_API_KEY")) and os.getenv("LLM_FALLBACK", "1") != "0"


def interpret(text: str, timeout: float = 8.0) -> Optional[dict]:
    """Return the validated LLM interpretation, or None on any failure (caller keeps the rule result)."""
    if not enabled():
        return None
    try:
        r = httpx.post(URL, headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, timeout=timeout, json={
            "model": MODEL, "temperature": 0, "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        })
        r.raise_for_status()
        out = json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception:            # network, quota, malformed JSON -> silently fall back to rules
        return None
    return validate(out)


def validate(out: dict) -> Optional[dict]:
    if not isinstance(out, dict) or out.get("intent") not in INTENTS:
        return None
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    meds = out.get("medications") or []
    return {"intent": out["intent"], "language": out.get("language"),
            "canonical_fr": str(out.get("canonical_fr") or "")[:300],
            "medications": [str(m)[:60] for m in meds if isinstance(m, (str, int))][:5],
            "confidence": round(conf, 2), "model": MODEL}
