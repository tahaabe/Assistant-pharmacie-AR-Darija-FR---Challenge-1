"""NLU: language ID, intent classification, entity extraction, catalog grounding.

Rule + lexicon + fuzzy-matching design: deterministic, explainable, runs on CPU with
no training data, and every extracted medication is grounded in the CNOPS reference
list rather than generated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import lexicon as L
from .data import Catalog, Medication, _CITY_LOOKUP, dosage_to_mg, public_with_label
from .normalize import AR_PREFIX, ar_variants, has_arabic, normalize, strip_ar_prefix


def _kw_pattern(kw: str) -> re.Pattern:
    k = re.escape(normalize(kw))
    if has_arabic(kw):
        return re.compile(rf"(?:^|\s){AR_PREFIX}{k}(?=\s|$)")
    return re.compile(rf"(?:^|\s){k}(?:s|x)?(?=\s|$)")


_INTENT_PATTERNS = {it: [(_kw_pattern(k), w, k) for k, w in kws.items()] for it, kws in L.INTENT_KEYWORDS.items()}
_PHARMA_PATTERNS = [_kw_pattern(k) for k in L.PHARMACY_WORDS]
_NEAR_PATTERNS = [_kw_pattern(k) for k in L.NEAR_ME]
_CITY_PATTERNS = sorted(((_kw_pattern(k), c, k) for k, c in _CITY_LOOKUP.items()), key=lambda t: -len(t[2]))
_KEYWORD_TOKENS = {t for kws in L.INTENT_KEYWORDS.values() for k in kws for t in normalize(k).split()}
_NUM = {normalize(k): v for k, v in L.NUMBER_WORDS.items()}
_UNITS = {normalize(k): v for k, v in L.UNIT_WORDS.items()}
_FORMS = {normalize(k): v for k, v in L.FORM_WORDS.items()}
_DOSE_UNITS = {normalize(k): v for k, v in L.DOSAGE_UNITS.items()}
_LINKERS = {"d", "de", "des", "les", "dyal", "dial", "dl", "ديال", "من", "د"}
_DUAL = {normalize(k): v for k, v in L.DUAL_UNITS.items()}
_SKIP = ({normalize(w) for w in L.STOPWORDS} | {normalize(w) for w in L.COMMON_WORDS} | _KEYWORD_TOKENS | set(_NUM) | set(_UNITS) | set(_FORMS)
         | set(_DOSE_UNITS) | {normalize(w) for w in L.PHARMACY_WORDS}
         | {t for k in _CITY_LOOKUP for t in k.split()})


# ---------------------------------------------------------------- language
def detect_language(text_norm: str, raw: str) -> dict:
    toks = text_norm.split()
    ar = [t for t in toks if has_arabic(t)]
    lat = [t for t in toks if not has_arabic(t) and re.search(r"[a-z]", t)]
    n_ar_tok = {strip_ar_prefix(t) for t in ar} | set(ar)
    darija = len(n_ar_tok & {normalize(w) for w in L.DARIJA_AR_MARKERS})
    msa = len(n_ar_tok & {normalize(w) for w in L.MSA_MARKERS})
    arabizi = sum(1 for t in lat if t in L.ARABIZI_MARKERS or re.search(r"[a-z][2379]|[2379][a-z]", t))
    french = sum(1 for t in lat if t in L.FRENCH_MARKERS)
    ar_label = "ar" if msa > darija else "darija"          # Moroccan context: Arabic script defaults to Darija
    lat_label = "darija_latin" if arabizi > 0 and arabizi >= french else "fr"
    present = ([ar_label] if ar else []) + ([lat_label] if lat else [])
    if not present:
        present = ["fr"]
    primary = ar_label if ar and len(ar) >= len(lat) else (lat_label if lat else present[0])
    # French words inside an Arabizi sentence (or vice versa) = code-mixing too
    mixed_latin = bool(lat) and arabizi > 0 and french > 0
    return {"primary": primary, "detected": present, "code_mixed": bool(ar and lat) or mixed_latin,
            "script": "arabic" if ar and not lat else "latin" if lat and not ar else "mixed",
            "markers": {"darija_ar": darija, "msa": msa, "arabizi": arabizi, "fr": french}}


# ---------------------------------------------------------------- intent
def classify_intent(text_norm: str, has_med: bool, has_city: bool, has_pharmacy_word: bool,
                    near_me: bool) -> dict:
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    padded = f" {text_norm} "
    for it, pats in _INTENT_PATTERNS.items():
        hits = [(w, kw) for pat, w, kw in pats if pat.search(padded)]
        # don't double count "vous avez" inside "est ce que vous avez"
        hits = [(w, kw) for w, kw in hits if not any(kw != k2 and normalize(kw) in normalize(k2) for _, k2 in hits)]
        for w, kw in hits:
            scores[it] = scores.get(it, 0) + w
            evidence.setdefault(it, []).append(kw)
    # context rules
    if near_me:
        scores["find_pharmacy"] = scores.get("find_pharmacy", 0) + 2.5
    if has_pharmacy_word and has_city and not has_med:
        scores["find_pharmacy"] = scores.get("find_pharmacy", 0) + 1.5
    if not has_med and has_pharmacy_word:
        # "is there a pharmacy open…" / "bghit pharmacie f rbat": the object is the pharmacy, not a drug
        moved = scores.pop("check_availability", 0) + (scores.pop("order_medication", 0) if has_city else 0)
        if moved:
            scores["find_pharmacy"] = scores.get("find_pharmacy", 0) + moved
            evidence.setdefault("find_pharmacy", []).append("<pharmacy as object>")
    if has_med and not any(scores.get(i, 0) > 0 for i in L.INTENT_PRIORITY[:-1]):
        scores["check_availability"] = scores.get("check_availability", 0) + 1.5   # "Doliprane ?"
        evidence.setdefault("check_availability", []).append("<medication only>")
    if has_med and "find_pharmacy" in scores and "order_medication" not in scores and "check_availability" not in scores:
        # "fin nl9a doliprane" = where can I find X -> availability
        scores["check_availability"] = scores.get("check_availability", 0) + 1.0
    if "pharmacy_hours" in scores and "find_pharmacy" in scores:
        # "where is the on-duty pharmacy" is a garde/hours question that also implies location
        scores["pharmacy_hours"] += scores.pop("find_pharmacy")
    if scores.get("out_of_scope", 0) >= 3:
        # a medical-advice question wins over price/availability readings of the same sentence
        scores = {"out_of_scope": scores["out_of_scope"] + max((v for k, v in scores.items() if k != "out_of_scope"), default=0)}
    # greeting only wins if nothing else fired
    if any(v > 0 for k, v in scores.items() if k != "greeting"):
        scores.pop("greeting", None)

    ranked = sorted(((k, v) for k, v in scores.items() if v > 0),
                    key=lambda kv: (-kv[1], L.INTENT_PRIORITY.index(kv[0])))
    if not ranked:
        return {"name": "unknown", "confidence": 0.0, "secondary": [], "evidence": {}}
    total = sum(v for _, v in ranked)
    top, top_s = ranked[0]
    conf = round(min(1.0, top_s / 3.0) * (top_s / total) ** 0.5, 2)
    return {"name": top, "confidence": conf,
            "secondary": [{"name": k, "score": round(v, 2)} for k, v in ranked[1:3]],
            "evidence": {k: evidence.get(k, []) for k, _ in ranked[:3]}}


# ---------------------------------------------------------------- entities
@dataclass
class MedMention:
    start: int
    end: int
    text: str
    kind: str
    key: str
    score: float


def _find_med_mentions(toks: list[str], cat: Catalog) -> list[MedMention]:
    found: list[MedMention] = []
    used: set[int] = set()
    # words right after "pharmacie"/"صيدلية" are usually the pharmacy's name: only near-exact drug hits there
    pharma_words = {normalize(w) for w in L.PHARMACY_WORDS}
    after_pharma = {j for i, t in enumerate(toks) if t in pharma_words or strip_ar_prefix(t) in pharma_words
                    for j in range(i + 1, min(i + 4, len(toks)))}
    for n in (3, 2, 1):
        for i in range(len(toks) - n + 1):
            idx = set(range(i, i + n))
            if idx & used:
                continue
            raw = toks[i:i + n]
            if any(strip_ar_prefix(t) in _SKIP or t in _SKIP or len(t) < 3 for t in (raw[0], raw[-1])) \
                    or any(re.search(r"\d", t) for t in raw):
                continue
            hit = None
            for vi, first in enumerate(ar_variants(raw[0])):   # the first token may carry a clitic (ف/ب/ال…)
                span = " ".join([first] + raw[1:])
                if len(span.replace(" ", "")) < 4:
                    continue
                h = cat.match_span(span)
                if h and vi > 0:
                    h = (h[0], h[1], h[2] + 0.5)     # prefer the reading without the clitic (البيتادين -> بيتادين)
                if h and (hit is None or h[2] > hit[2]):
                    hit = h
            if hit and idx & after_pharma and hit[2] < 95:
                continue
            if hit and (n == 1 or hit[2] >= 90):
                end = i + n
                if hit[0] == "brand":
                    # "doliprane vitamine c", "spasfon lyoc": absorb following words that continue a product name
                    names = [normalize(p.name) for p in cat.products("brand", hit[1])]
                    while end < len(toks) and end not in used and not re.search(r"\d", toks[end]) \
                            and toks[end] not in _DOSE_UNITS and toks[end] not in _UNITS:
                        nxt = " ".join([hit[1]] + toks[i + n:end + 1])
                        if not any(nm == nxt or nm.startswith(nxt + " ") for nm in names):
                            break
                        end += 1
                found.append(MedMention(i, end, " ".join(toks[i:end]), *hit))
                used |= set(range(i, end))
    return sorted(found, key=lambda m: m.start)


def _numbers(toks: list[str]) -> list[dict]:
    """Every numeric item with its position, plus what it is attached to."""
    out = []
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        nxt2 = toks[i + 2] if i + 2 < len(toks) else ""
        val = None
        if re.fullmatch(r"\d+(?:[.,]\d+)?", t):
            val = float(t.replace(",", "."))
        elif t in _NUM:
            val = float(_NUM[t])
        elif t in L.FR_ARTICLE_ONE and (nxt in _UNITS or nxt2 in _UNITS):
            val = 1.0
        elif strip_ar_prefix(t) in _DUAL or t in _DUAL:
            out.append({"i": i, "value": 2.0, "role": "quantity", "unit": _UNITS.get(t, "box")})
            continue
        if val is None and _UNITS.get(t) in ("box", "bottle", "blister", "tube"):
            # bare unit word ("علبة سبسفون", "une boite"): quantity 1 unless a number precedes it closely
            prev = toks[max(0, i - 3):i]
            if not any(re.fullmatch(r"\d+", x) or x in _NUM or x in L.FR_ARTICLE_ONE for x in prev):
                out.append({"i": i, "value": 1.0, "role": "quantity", "unit": _UNITS[t]})
            continue
        if val is None:
            continue
        look = [x for x in toks[i + 1:i + 4] if x not in _LINKERS]
        if nxt in _DOSE_UNITS:
            out.append({"i": i, "value": val, "role": "dosage", "unit": _DOSE_UNITS[nxt]})
        elif look and look[0] in _UNITS:
            unit = _UNITS[look[0]]
            out.append({"i": i, "value": val, "role": "quantity", "unit": unit})
        else:
            out.append({"i": i, "value": val, "role": "bare", "unit": None})
    return out


def _forms(toks: list[str]) -> list[tuple[int, str]]:
    return [(i, _FORMS[strip_ar_prefix(t)]) for i, t in enumerate(toks) if strip_ar_prefix(t) in _FORMS]


def variant_key(p) -> tuple:
    g = (lambda k: p[k]) if isinstance(p, dict) else (lambda k: getattr(p, k))
    return (g("name"), g("dosage_mg"), g("form"))


def narrow(cands: list, dosage_mg: float | None, form: str | None) -> list:
    """Filter product variants by dosage and/or form; a filter that removes everything is ignored."""
    g = (lambda p, k: p[k]) if cands and isinstance(cands[0], dict) else getattr
    if dosage_mg is not None:
        exact = [p for p in cands if g(p, "dosage_mg") is not None and abs(g(p, "dosage_mg") - dosage_mg) < 1e-6]
        cands = exact or cands
    if form:
        f = normalize(form)[:5]
        by_form = [p for p in cands if g(p, "form").startswith(f) or f in g(p, "form")]
        cands = by_form or cands
    return cands


def _resolve(products: list[Medication], dosage_mg: float | None, form: str | None) -> tuple[list[Medication], bool]:
    cands = narrow(products, dosage_mg, form)
    # several presentations of the same product (same name, dose, form) are not a real ambiguity
    return cands, len({variant_key(p) for p in cands}) > 1


def extract(text: str, cat: Catalog) -> dict:
    tn = normalize(text)
    toks = tn.split()
    padded = f" {tn} "

    city = None
    for pat, c, _ in _CITY_PATTERNS:
        if pat.search(padded):
            city = c
            break
    near_me = any(p.search(padded) for p in _NEAR_PATTERNS)
    pharmacy_word = any(p.search(padded) for p in _PHARMA_PATTERNS)
    garde = bool(re.search(r"(?:^|\s)(?:de )?garde(?:\s|$)|حراسة|كارد", tn))

    mentions = _find_med_mentions(toks, cat)
    nums = _numbers(toks)
    forms = _forms(toks)
    med_tokens = {j for m in mentions for j in range(m.start, m.end)}

    def owner(i: int, role: str) -> int:
        """Index of the mention a number/form at token i belongs to: the nearest one;
        on ties quantities go to the following drug ("2 boites de X"), dosages to the preceding ("X 500")."""
        best, best_key = 0, None
        for k, m in enumerate(mentions):
            dist = m.start - i if i < m.start else i - m.end + 1
            after = i >= m.end
            tie = 0 if (after and role != "quantity") or (not after and role == "quantity") else 1
            key = (dist, tie)
            if best_key is None or key < best_key:
                best, best_key = k, key
        return best

    meds_out = []
    used_nums: set[int] = set()
    for k, m in enumerate(mentions):
        products = cat.products(m.kind, m.key)
        if m.kind == "brand" and len(m.text.split()) > 1:
            phrase = " ".join([m.key] + [strip_ar_prefix(t) for t in m.text.split()[1:]])
            products = [p for p in products if normalize(p.name).startswith(phrase)] or products
        known_doses = {p.dosage for p in products} | {p.dosage_mg for p in products}
        dosage_mg, qty, qty_unit = None, None, None
        for nb in nums:
            if nb["i"] in used_nums or owner(nb["i"], nb["role"]) != k:
                continue
            after = nb["i"] >= m.end
            if nb["role"] == "dosage" and dosage_mg is None:
                dosage_mg = dosage_to_mg(nb["value"], nb["unit"])
                used_nums.add(nb["i"])
            elif nb["role"] == "quantity" and qty is None:
                qty, qty_unit = int(nb["value"]), nb["unit"]
                used_nums.add(nb["i"])
            elif nb["role"] == "bare":
                if after and nb["i"] == m.end and nb["value"] in known_doses and dosage_mg is None:
                    # "doliprane 1000" -> dosage of a known variant
                    p = next(p for p in products if nb["value"] in (p.dosage, p.dosage_mg))
                    dosage_mg = dosage_to_mg(nb["value"], p.dosage_unit) if nb["value"] == p.dosage else nb["value"]
                    used_nums.add(nb["i"])
                elif qty is None and nb["value"] <= 20 and float(nb["value"]).is_integer():
                    qty = int(nb["value"])
                    used_nums.add(nb["i"])
        form = next((f for i, f in forms if owner(i, "form") == k), None)
        cands, ambiguous = _resolve(products, dosage_mg, form)
        meds_out.append({
            "mention": m.text, "match_type": m.kind, "matched": m.key, "match_score": m.score,
            "requested_dosage_mg": dosage_mg, "requested_form": form,
            "quantity": qty, "quantity_unit": qty_unit or ("box" if qty else None),
            "resolved": public_with_label(cands[0]) if cands and not ambiguous else None,
            "candidates": [public_with_label(c) for c in cands] if ambiguous else [],
            "needs_variant_choice": ambiguous,
        })

    pharmacy_name = None
    mt = re.search(r"(?:pharmacie|صيدلية|الصيدلية|farmasyan)\s+((?:\w+\s?){1,3})", tn)
    if mt:
        cand = mt.group(1).strip()
        first = cand.split()[0]
        if first not in _SKIP and not any(first == m.text for m in mentions) and not re.search(r"\d", cand):
            pharmacy_name = cand

    return {
        "text_normalized": tn, "medications": meds_out, "city": city, "near_me": near_me,
        "pharmacy_word": pharmacy_word, "garde": garde, "pharmacy_name": pharmacy_name,
        "numbers": [n for n in nums if n["i"] not in used_nums and n["i"] not in med_tokens],
        "forms": [f for _, f in forms],
        "unknown_tokens": [t for i, t in enumerate(toks) if i not in med_tokens and strip_ar_prefix(t) not in _SKIP
                           and t not in _SKIP and not re.search(r"\d", t) and len(t) >= 4
                           and t not in L.FRENCH_MARKERS and t not in L.ARABIZI_MARKERS
                           and strip_ar_prefix(t) not in {normalize(w) for w in L.DARIJA_AR_MARKERS | L.MSA_MARKERS}],
    }
