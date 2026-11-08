"""Dialogue manager: merges NLU with session state, grounds results in the reference
data, decides missing slots, builds the structured pharmacy request and a reply in the
user's language."""
from __future__ import annotations

import time
import uuid

from rapidfuzz import fuzz

from . import llm
from .data import Catalog, PharmacyDirectory
from .data import dosage_to_mg, label_numbers, pretty_dose, public_with_label
from .nlu import classify_intent, detect_language, extract, narrow, variant_key

MED_INTENTS = {"order_medication", "check_availability", "ask_price", "ask_reimbursement", "ask_alternative"}
PHARMA_INTENTS = {"find_pharmacy", "pharmacy_hours", "pharmacy_contact"}
NEEDS_VARIANT = {"order_medication", "ask_price", "ask_alternative"}
SESSION_TTL_S = 1800
MAX_KM = 25.0   # radius for "near me" searches

# ---------------------------------------------------------------- reply templates
T = {
    "fr": {
        "greet": "Bonjour ! Je peux vérifier un médicament (prix, remboursement, générique), préparer une demande pour votre pharmacie ou trouver une pharmacie (de garde). Que puis-je faire pour vous ?",
        "unknown": "Je n'ai pas bien compris. Vous pouvez par exemple écrire : « Avez-vous du Doliprane 1000 ? » ou « pharmacie de garde à Rabat ».",
        "ask_med": "De quel médicament s'agit-il ?",
        "not_found": "Je n'ai pas trouvé « {q} » dans la liste de référence des médicaments. Pouvez-vous vérifier l'orthographe ou envoyer une photo de la boîte ?",
        "ask_variant": "Pour {brand}, quelle version souhaitez-vous (numéro ou dosage) ? {options}",
        "ask_city": "Dans quelle ville (ou quel quartier) êtes-vous ?",
        "price": "{name} : prix public (PPV) {ppv} DH.",
        "price_unknown": "{name} : prix non disponible dans les données chargées.",
        "reimb_yes": "{name} figure parmi les médicaments remboursables{rate}.",
        "reimb_no": "{name} n'apparaît pas comme remboursable dans la liste chargée.",
        "reimb_unknown": "{name} : information de remboursement non disponible dans les données chargées.",
        "generics": "Équivalents de {name} (même DCI et dosage) : {items}.",
        "no_generics": "Aucun équivalent de {name} avec la même DCI et le même dosage n'a été trouvé.",
        "avail": "{items} figure dans le référentiel. Le stock réel dépend de la pharmacie : j'ai préparé une demande de disponibilité.",
        "order": "Demande préparée : {items}.",
        "send_to": "Pharmacies proposées ({city}) : {pharmacies}. Laquelle choisissez-vous ?",
        "pharmacies": "Pharmacies{garde} à {city} : {pharmacies}.",
        "pharmacies_near": "Pharmacies{garde} les plus proches : {pharmacies}.",
        "no_pharmacy": "Je n'ai pas trouvé de pharmacie{garde} pour {city} dans l'annuaire chargé.",
        "contact": "{name} : {phone} — {address}.",
        "garde_word": " de garde",
        "nearby": "à proximité",
        "variant_retry": "Je n'ai pas reconnu cette option.",
        "out_of_scope": "Je ne peux pas donner d'avis médical (posologie, effets secondaires, diagnostic). Demandez conseil à votre pharmacien ou à votre médecin. Je peux en revanche vérifier un médicament ou trouver une pharmacie.",
    },
    "darija": {
        "greet": "السلام! نقدر نعاونك تعرف ثمن الدوا، واش كيتعوض، البديل ديالو، نوجد طلب للصيدلية ولا نلقى ليك صيدلية (ديال الحراسة). شنو بغيتي؟",
        "unknown": "ما فهمتش مزيان. مثلا كتب: «واش كاين دوليبران 1000؟» ولا «صيدلية الحراسة فالرباط».",
        "ask_med": "شنو هو الدوا اللي بغيتي؟",
        "not_found": "ما لقيتش «{q}» فلائحة الأدوية. عافاك تأكد من السمية ولا صيفط تصويرة ديال العلبة.",
        "ask_variant": "{brand}: شمن وحدة بغيتي (الرقم ولا الدوز)؟ {options}",
        "ask_city": "فاش من مدينة (ولا حومة) نتا؟",
        "price": "{name}: الثمن ديالو {ppv} درهم.",
        "price_unknown": "{name}: الثمن ماكاينش فالمعطيات اللي عندي.",
        "reimb_yes": "{name} كيتعوض{rate}.",
        "reimb_no": "{name} ما كاينش فلائحة الأدوية اللي كتعوض.",
        "reimb_unknown": "{name}: معلومات التعويض ماكايناش فالمعطيات اللي عندي.",
        "generics": "البدائل ديال {name} (نفس المادة ونفس الدوز): {items}.",
        "no_generics": "ما لقيتش بديل ل{name} بنفس المادة ونفس الدوز.",
        "avail": "{items} كاين فالمرجع. الستوك الحقيقي كيتعرف غير من الصيدلية: وجدت ليك طلب باش نتأكدو.",
        "order": "وجدت الطلب: {items}.",
        "send_to": "الصيدليات ({city}): {pharmacies}. شمن وحدة بغيتي؟",
        "pharmacies": "الصيدليات{garde} ف {city}: {pharmacies}.",
        "pharmacies_near": "أقرب صيدليات{garde}: {pharmacies}.",
        "no_pharmacy": "ما لقيتش صيدلية{garde} ف {city}.",
        "contact": "{name}: {phone} — {address}.",
        "garde_word": " ديال الحراسة",
        "variant_retry": "ما عرفتش هاد الاختيار.",
        "out_of_scope": "ما نقدرش نعطيك نصيحة طبية (الدوز، الأعراض الجانبية، التشخيص). سول الفرمسيان ولا الطبيب ديالك. ولكن نقدر نشوف ليك دوا ولا نلقى ليك صيدلية.",
        "nearby": "قراب ليك",
    },
    "ar": {
        "greet": "مرحبا! يمكنني مساعدتك في معرفة ثمن الدواء وتعويضه وبدائله، أو إعداد طلب للصيدلية، أو إيجاد صيدلية (حراسة). كيف أساعدك؟",
        "unknown": "لم أفهم طلبك جيدا. مثلا: «هل يوجد دوليبران 1000؟» أو «صيدلية الحراسة في الرباط».",
        "ask_med": "ما هو اسم الدواء؟",
        "not_found": "لم أجد «{q}» في لائحة الأدوية المرجعية. المرجو التحقق من الاسم أو إرسال صورة العلبة.",
        "ask_variant": "{brand}: ما هي النسخة المطلوبة (الرقم أو الجرعة)؟ {options}",
        "ask_city": "في أي مدينة (أو حي) أنت؟",
        "price": "{name}: السعر العمومي {ppv} درهم.",
        "price_unknown": "{name}: السعر غير متوفر في المعطيات المحملة.",
        "reimb_yes": "{name} من الأدوية القابلة للتعويض{rate}.",
        "reimb_no": "{name} لا يظهر في لائحة الأدوية القابلة للتعويض.",
        "reimb_unknown": "{name}: معلومات التعويض غير متوفرة في المعطيات المحملة.",
        "generics": "بدائل {name} (نفس المادة الفعالة والجرعة): {items}.",
        "no_generics": "لم أجد بديلا ل{name} بنفس المادة والجرعة.",
        "avail": "{items} موجود في المرجع. المخزون الفعلي تحدده الصيدلية: تم إعداد طلب للتحقق من التوفر.",
        "order": "تم إعداد الطلب: {items}.",
        "send_to": "الصيدليات المقترحة ({city}): {pharmacies}. أيها تختار؟",
        "pharmacies": "الصيدليات{garde} في {city}: {pharmacies}.",
        "pharmacies_near": "أقرب الصيدليات{garde}: {pharmacies}.",
        "no_pharmacy": "لم أجد صيدلية{garde} في {city}.",
        "contact": "{name}: {phone} — {address}.",
        "garde_word": " الحراسة",
        "variant_retry": "لم أتعرف على هذا الاختيار.",
        "out_of_scope": "لا يمكنني تقديم نصيحة طبية (الجرعة، الآثار الجانبية، التشخيص). يرجى استشارة الصيدلي أو الطبيب. يمكنني مساعدتك في التحقق من دواء أو إيجاد صيدلية.",
        "nearby": "بالقرب منك",
    },
    "darija_latin": {
        "greet": "Salam! N9der n3awnek f taman dyal dwa, wach kayt3wed, l bdil dyalo, nwjjed talab l pharmacie wla nl9a lik pharmacie (de garde). Chno bghiti?",
        "unknown": "Ma fhmtch mzyan. Ktb matalan: « wach kayn doliprane 1000? » wla « pharmacie de garde f rbat ».",
        "ask_med": "Chno howa dwa li bghiti?",
        "not_found": "Ma l9itch « {q} » f la liste dyal dwayat. 3afak t2kd mn smiya wla sift tswira dyal l3lba.",
        "ask_variant": "{brand}: chmen wa7da bghiti (raqm wla dosage)? {options}",
        "ask_city": "F ina mdina (wla 7ouma) nta?",
        "price": "{name}: taman dyalo {ppv} DH.",
        "price_unknown": "{name}: taman makaynch f data li 3ndi.",
        "reimb_yes": "{name} kayt3wed{rate}.",
        "reimb_no": "{name} makaynch f la liste dyal dwayat li kayt3wdo.",
        "reimb_unknown": "{name}: ma3ndich ma3loumat 3la remboursement.",
        "generics": "Bdayl dyal {name} (nfs DCI w nfs dosage): {items}.",
        "no_generics": "Ma l9itch bdil l {name} b nfs DCI w dosage.",
        "avail": "{items} kayn f référentiel. Stock l7a9i3i kay3rfo ghir pharmacie: wjjedt lik talab bach nt2kdo.",
        "order": "Wjjedt talab: {items}.",
        "send_to": "Pharmacies ({city}): {pharmacies}. Chmen wa7da bghiti?",
        "pharmacies": "Pharmacies{garde} f {city}: {pharmacies}.",
        "pharmacies_near": "A9rab pharmacies{garde}: {pharmacies}.",
        "no_pharmacy": "Ma l9itch pharmacie{garde} f {city}.",
        "contact": "{name}: {phone} — {address}.",
        "garde_word": " de garde",
        "nearby": "qrab lik",
        "variant_retry": "Ma 3reftch had l khtiyar.",
        "out_of_scope": "Ma n9derch n3tik nasi7a tibiya (dose, effets secondaires, diagnostic). Sowl l pharmacien wla tbib dyalk. Walakin n9der nchof lik dwa wla nl9a lik pharmacie.",
    },
}


class SessionStore:
    def __init__(self):
        self._s: dict[str, dict] = {}

    def get(self, sid: str) -> dict:
        now = time.time()
        for k in [k for k, v in self._s.items() if now - v["ts"] > SESSION_TTL_S]:
            del self._s[k]
        st = self._s.setdefault(sid, {"ts": now, "pending_intent": None, "pending_slot": None,
                                      "meds": [], "city": None, "lang": None, "turns": 0})
        st["ts"] = now
        return st


MAX_OPTIONS = 8


def _dose_txt(c: dict) -> str:
    return pretty_dose(c.get("dosage_label") or "") or (
        f"{c['dosage']:g} {c['dosage_unit']}" if c.get("dosage") is not None else "")


def variant_groups(cands: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group the candidates on the dimension that actually differs (dosage, then form, then full label).
    Returns [(option text, candidates)] — the options are numbered so the patient can answer '2'."""
    for key in (_dose_txt, lambda c: c["form"], lambda c: c["label"]):
        groups: dict[str, list[dict]] = {}
        for c in cands:
            k = key(c)
            if k:
                groups.setdefault(k, []).append(c)
        if len(groups) > 1:
            out = []
            for k, cs in groups.items():
                forms = {c["form"] for c in cs}
                # show the form next to a dosage when it is the same for the whole group
                out.append((f"{k} {forms.pop()}" if key is _dose_txt and len(forms) == 1 else k, cs))
            return out
    return [(cands[0]["label"], cands)] if cands else []


def variant_options(cands: list[dict]) -> str:
    g = variant_groups(cands)
    txt = " ; ".join(f"{i}) {k}" for i, (k, _) in enumerate(g[:MAX_OPTIONS], 1))
    return txt + (" …" if len(g) > MAX_OPTIONS else "")


def _med_label(m: dict) -> str:
    r = m["resolved"]
    base = r.get("label", r["name"]).split(" (")[0] if r else m["matched"].upper()
    return f"{m['quantity']} × {base}" if m.get("quantity") else base


def _pharm_label(p: dict) -> str:
    extra = f", {p['distance_km']} km" if "distance_km" in p else ""
    return f"{p['name']} ({p['phone']}{extra})"


class Assistant:
    def __init__(self, catalog: Catalog | None = None, pharmacies: PharmacyDirectory | None = None):
        self.cat = catalog or Catalog.load()
        self.ph = pharmacies or PharmacyDirectory.load()
        self.sessions = SessionStore()

    @staticmethod
    def _needs_fallback(intent: dict, ent: dict) -> bool:
        """Call the LLM only when the rules are unsure: nothing recognised, low confidence,
        or a medication question whose drug could not be grounded."""
        if not llm.enabled():
            return False
        if intent["name"] == "unknown" or intent["confidence"] < llm.THRESHOLD:
            return not (intent["name"] == "greeting" and len(ent["text_normalized"].split()) <= 3)
        if intent.get("evidence") == {"check_availability": ["<medication only>"]} and len(ent["text_normalized"].split()) > 3:
            return True      # a drug name but no intent keyword in a real sentence: the intent is a guess
        return intent["name"] in MED_INTENTS and not ent["medications"] and bool(ent["unknown_tokens"])

    def handle(self, text: str, session_id: str | None = None, lat: float | None = None,
               lon: float | None = None, channel: str = "text", transcript_meta: dict | None = None) -> dict:
        sid = session_id or uuid.uuid4().hex[:12]
        st = self.sessions.get(sid)
        st["turns"] += 1

        ent = extract(text, self.cat)
        lang = detect_language(ent["text_normalized"], text)
        intent = classify_intent(ent["text_normalized"], bool(ent["medications"]), bool(ent["city"]),
                                 ent["pharmacy_word"], ent["near_me"])
        nlu_source, llm_out = "rules", None
        short_followup = bool(st["pending_intent"]) and len(ent["text_normalized"].split()) <= 4
        if not short_followup and self._needs_fallback(intent, ent):
            llm_out = llm.interpret(text)
            if llm_out:
                nlu_source = "rules+llm"
                if llm_out["canonical_fr"] and llm_out["intent"] not in ("out_of_scope", "greeting"):
                    ent2 = extract(llm_out["canonical_fr"], self.cat)       # re-ground through the catalog
                    old = {m["matched"]: m for m in ent["medications"]}
                    if ent2["medications"] and set(old) <= {m["matched"] for m in ent2["medications"]}:
                        # the rewrite found the same drugs (or more) with cleaner dosage/form: use it,
                        # keeping quantities the rules already read from the original wording
                        for m in ent2["medications"]:
                            if m["quantity"] is None and m["matched"] in old:
                                m["quantity"], m["quantity_unit"] = old[m["matched"]]["quantity"], old[m["matched"]]["quantity_unit"]
                        ent["medications"] = ent2["medications"]
                    for k in ("numbers", "forms"):
                        if ent2[k] and not ent[k]:
                            ent[k] = ent2[k]
                    for k in ("city", "pharmacy_name"):
                        ent[k] = ent[k] or ent2[k]
                    for k in ("garde", "near_me", "pharmacy_word"):
                        ent[k] = ent[k] or ent2[k]
                    if ent["medications"]:
                        ent["unknown_tokens"] = []
                intent = {"name": llm_out["intent"], "confidence": llm_out["confidence"], "secondary": [],
                          "evidence": {"llm": llm_out["model"]}, "rules_intent": intent["name"]}
                if lang["markers"] and sum(lang["markers"].values()) == 0 and llm_out.get("language") in T:
                    lang = {**lang, "primary": llm_out["language"]}
        L = lang["primary"]
        if L not in T:
            L = "fr"
        # keep the session's language when the turn is too short to tell (e.g. "1000", "Rabat")
        if st["lang"] and len(ent["text_normalized"].split()) <= 2:
            L = st["lang"]
        st["lang"] = L
        tpl = T[L]

        # -------- merge with session state (slot filling across turns)
        name = intent["name"]
        followup = False
        amb = [m for m in st["meds"] if m["needs_variant_choice"]]
        variant_retry = False
        if amb and not ent["medications"] and (ent["numbers"] or ent["forms"] or len(ent["text_normalized"].split()) <= 4):
            # "1000" / "sirop" / "1 g comprimé" / "200" / "2" answers "which variant?" for the first ambiguous drug
            m = amb[0]
            dose = None
            if ent["numbers"]:
                nb = ent["numbers"][0]
                dose = dosage_to_mg(nb["value"], nb["unit"]) if nb["role"] == "dosage" else nb["value"]
                if not any(dose in (c["dosage"], c["dosage_mg"]) for c in m["candidates"]):
                    dose = None
                elif not any(c["dosage_mg"] == dose for c in m["candidates"]):
                    dose = next(c["dosage_mg"] for c in m["candidates"] if c["dosage"] == dose)
            form = ent["forms"][0] if ent["forms"] else None
            left = narrow(m["candidates"], dose, form)
            if len(left) == len(m["candidates"]) and ent["numbers"]:
                vals = {n["value"] for n in ent["numbers"]} | {
                    dosage_to_mg(n["value"], n["unit"]) for n in ent["numbers"] if n["role"] == "dosage"}
                vals.discard(None)
                # any number of a combination counts: "200" picks "1 g / 200 mg"
                by_num = [c for c in left if vals and vals & label_numbers(c)]
                if by_num and len(by_num) < len(left):
                    left = by_num
                else:
                    # "2" = second option of the list we showed
                    groups = variant_groups(m["candidates"])
                    k = int(ent["numbers"][0]["value"]) if float(ent["numbers"][0]["value"]).is_integer() else 0
                    if len(ent["numbers"]) == 1 and 1 <= k <= min(len(groups), MAX_OPTIONS) and ent["numbers"][0]["role"] == "bare":
                        left = groups[k - 1][1]
            if len({variant_key(c) for c in left}) > 1 and len(ent["text_normalized"].split()) >= 2:
                # "comprimé sécable": pick the variant whose full form matches the answer best
                sc = {id(c): fuzz.token_set_ratio(ent["text_normalized"], c["form"]) for c in left}
                best = max(sc.values())
                if best >= 85:
                    left = [c for c in left if sc[id(c)] == best]
            if len(left) == len(m["candidates"]) and intent["name"] in ("unknown", "greeting") \
                    and not (ent["city"] or lat is not None):
                variant_retry = True            # answer not understood: ask the same question again
            if len(left) < len(m["candidates"]):
                followup = True
                ent["medications"] = st["meds"]
                if dose is not None:
                    m["requested_dosage_mg"] = dose
                if len({variant_key(c) for c in left}) == 1:
                    m["resolved"], m["candidates"], m["needs_variant_choice"] = left[0], [], False
                else:
                    m["candidates"] = left
        if variant_retry and st["pending_intent"]:
            name = st["pending_intent"]
            intent = {**intent, "name": name, "confidence": 0.5, "from_context": True}
            ent["medications"] = st["meds"]
        if name in ("unknown", "greeting") and st["pending_intent"] and (followup or ent["city"] or ent["medications"] or lat is not None):
            name = st["pending_intent"]
            intent = {**intent, "name": name, "confidence": 0.6, "from_context": True}
        if name in MED_INTENTS and not ent["medications"] and st["meds"]:
            ent["medications"] = st["meds"]          # "ch7al taman dyalo?" -> previous drug
        city = ent["city"] or st["city"]
        if ent["city"]:
            st["city"] = ent["city"]

        missing: list[str] = []
        replies: list[str] = []
        results: dict = {"medications": [], "pharmacies": []}
        request = None
        meds = ent["medications"]

        # -------- medication intents
        if name in MED_INTENTS:
            if not meds:
                missing.append("medication")
                unk = ent["unknown_tokens"]
                if unk:
                    sugg = self.cat.search(unk[-1], 3)
                    results["suggestions"] = [x for x in sugg if x["score"] >= 75]
                    replies.append(tpl["not_found"].format(q=unk[-1]))
                else:
                    replies.append(tpl["ask_med"])
            for m in meds:
                if m["needs_variant_choice"] and name in NEEDS_VARIANT:
                    missing.append("medication_variant")
                    opts = variant_options(m["candidates"])
                    if variant_retry and tpl["variant_retry"] not in replies:
                        replies.append(tpl["variant_retry"])
                    replies.append(tpl["ask_variant"].format(brand=m["matched"].upper(), options=opts))
                    continue
                prods = [m["resolved"]] if m["resolved"] else m["candidates"]
                for p in prods:
                    info = dict(p)
                    if name == "ask_alternative":
                        med_obj = next(x for x in self.cat.meds if x.code == p["code"])
                        info["equivalents"] = [public_with_label(g) for g in self.cat.generics_for(med_obj)[:5]]
                    results["medications"].append(info)
                if name == "ask_price":
                    for p in prods:
                        replies.append(tpl["price"].format(name=p.get("label", p["name"]).split(" (")[0], ppv=f"{p['ppv']:.2f}")
                                       if p["ppv"] else tpl["price_unknown"].format(name=p["name"]))
                elif name == "ask_reimbursement":
                    seen = set()
                    prods = [p for p in prods if not (variant_key(p) in seen or seen.add(variant_key(p)))]
                    for p in prods[:4]:
                        rate = f" ({int(p['reimb_rate'])}%)" if p.get("reimb_rate") else ""
                        key = {True: "reimb_yes", False: "reimb_no", None: "reimb_unknown"}[p["reimbursable"]]
                        replies.append(tpl[key].format(name=p.get("label", p["name"]).split(" (")[0], rate=rate))
                elif name == "ask_alternative" and m["resolved"]:
                    eq = results["medications"][-1]["equivalents"]
                    replies.append(tpl["generics"].format(name=m["resolved"]["name"], items=", ".join(
                        e["name"] + (f" ({e['ppv']:.2f} DH)" if e["ppv"] else "") for e in eq))
                        if eq else tpl["no_generics"].format(name=m["resolved"]["name"]))

            if name in ("order_medication", "check_availability") and meds and "medication_variant" not in missing:
                items = [{"code": (m["resolved"] or {}).get("code"), "name": (m["resolved"] or {}).get("name") or m["matched"].upper(),
                          "label": (m["resolved"] or {}).get("label"),
                          "dci": (m["resolved"] or {}).get("dci"), "quantity": m["quantity"] or 1,
                          "unit": m["quantity_unit"] or "box", "variant_confirmed": bool(m["resolved"])} for m in meds]
                label = ", ".join(_med_label(m) for m in meds)
                replies.append((tpl["order"] if name == "order_medication" else tpl["avail"]).format(items=label))
                pharms = self.ph.search(city=None if lat is not None else city, lat=lat, lon=lon,
                                        garde=ent["garde"] or None, limit=3, max_km=MAX_KM) if (city or lat is not None) else []
                results["pharmacies"] = pharms
                if pharms:
                    replies.append(tpl["send_to"].format(city=tpl["nearby"] if lat is not None else city, pharmacies="; ".join(map(_pharm_label, pharms))).rstrip())
                    missing.append("target_pharmacy")
                else:
                    missing.append("city_or_location")
                    replies.append(tpl["ask_city"])
                request = {
                    "request_type": "reservation" if name == "order_medication" else "availability_check",
                    "items": items, "city": city, "patient_location": {"lat": lat, "lon": lon} if lat is not None else None,
                    "candidate_pharmacy_ids": [p["id"] for p in pharms], "status": "draft",
                    "channel": channel, "language": L,
                }

        # -------- pharmacy intents
        elif name in PHARMA_INTENTS:
            garde = ent["garde"] or name == "pharmacy_hours"
            gw = tpl["garde_word"] if garde else ""
            if name == "pharmacy_contact" and ent["pharmacy_name"]:
                found = self.ph.by_name(ent["pharmacy_name"], city)
                results["pharmacies"] = found
                for p in found[:2]:
                    replies.append(tpl["contact"].format(name=p["name"], phone=p["phone"], address=p["address"]))
                if not found:
                    replies.append(tpl["no_pharmacy"].format(garde="", city=ent["pharmacy_name"]))
            elif city or lat is not None:
                pharms = self.ph.search(city=None if lat is not None else city, garde=garde, lat=lat, lon=lon, limit=5,
                                        max_km=MAX_KM)
                results["pharmacies"] = pharms
                if pharms:
                    key = "pharmacies_near" if lat is not None else "pharmacies"
                    replies.append(tpl[key].format(garde=gw, city=city, pharmacies="; ".join(map(_pharm_label, pharms))))
                else:
                    replies.append(tpl["no_pharmacy"].format(garde=gw, city=city))
            else:
                missing.append("city_or_location")
                replies.append(tpl["ask_city"])
        elif name == "out_of_scope":
            replies.append(tpl["out_of_scope"])
        elif name == "greeting":
            replies.append(tpl["greet"])
        else:
            replies.append(tpl["unknown"])

        # -------- persist state
        if meds:
            st["meds"] = meds
        st["pending_intent"] = name if missing and name not in ("unknown", "greeting") else None
        st["pending_slot"] = ("variant" if "medication_variant" in missing else
                              "city" if "city_or_location" in missing else
                              "medication" if "medication" in missing else None)

        return {
            "session_id": sid, "turn": st["turns"],
            "input": {"text": text, "channel": channel, **({"asr": transcript_meta} if transcript_meta else {})},
            "language": lang | {"reply_language": L},
            "intent": intent,
            "entities": {"medications": [{k: v for k, v in m.items() if k not in ("candidates", "resolved")}
                                         | {"resolved_code": (m["resolved"] or {}).get("code"),
                                            "candidate_codes": [c["code"] for c in m["candidates"]]}
                                         for m in meds],
                         "city": city, "near_me": ent["near_me"], "garde": ent["garde"],
                         "pharmacy_name": ent["pharmacy_name"]},
            "results": results,
            "nlu_source": nlu_source,
            **({"llm": llm_out} if llm_out else {}),
            "missing_slots": missing,
            "needs_clarification": bool(missing),
            "pharmacy_request": request,
            "reply": " ".join(replies),
            "data_sources": {"medications": self.cat.source_file, "pharmacies": self.ph.source_file},
        }
