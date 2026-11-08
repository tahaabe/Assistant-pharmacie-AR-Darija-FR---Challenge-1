import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.normalize import skeleton

client = TestClient(app)


@pytest.mark.parametrize("latin,arabic", [("doliprane", "دوليبران"), ("augmentin", "اوغمنتين"),
                                          ("clamoxyl", "كلاموكسيل"), ("glucophage", "كلوكوفاج"),
                                          ("lasilix", "لازيليكس")])
def test_skeleton_crosses_scripts(latin, arabic):
    assert skeleton(latin) == skeleton(arabic)


def ask(text, sid=None, **kw):
    r = client.post("/v1/analyze", json={"text": text, "session_id": sid, **kw})
    assert r.status_code == 200
    return r.json()


def test_darija_order_two_drugs_structured():
    o = ask("بغيت جوج علب ديال دوليبران 500 و علبة سبسفون ف سلا")
    assert o["intent"]["name"] == "order_medication"
    assert o["language"]["primary"] == "darija"
    items = o["pharmacy_request"]["items"]
    assert [(i["code"], i["quantity"]) for i in items] == [("S001", 2), ("S009", 1)]
    assert o["entities"]["city"] == "Salé"


def test_arabizi_price_with_dosage_in_grams():
    o = ask("ch7al taman augmentin 1g")
    assert o["intent"]["name"] == "ask_price"
    assert o["entities"]["medications"][0]["resolved_code"] == "S005"


def test_multi_turn_variant_then_city():
    o = ask("واش كاين الدوليبران؟", "t1")
    assert o["entities"]["medications"][0]["needs_variant_choice"]
    o = ask("1000", "t1")
    assert o["intent"]["name"] == "check_availability"
    assert o["entities"]["medications"][0]["resolved_code"] == "S002"
    o = ask("f rbat", "t1")
    assert o["pharmacy_request"]["city"] == "Rabat" and o["results"]["pharmacies"]


def test_garde_pharmacy_rabat():
    o = ask("fin kayna pharmacie de garde f rabat")
    assert o["intent"]["name"] == "pharmacy_hours"
    assert all(p["garde"] for p in o["results"]["pharmacies"])


def test_near_me_uses_coordinates():
    o = ask("pharmacie qriba liya", lat=33.59, lon=-7.64)
    assert o["results"]["pharmacies"][0]["distance_km"] < 5


def test_unknown_drug_is_not_hallucinated():
    o = ask("bghit zitromax")
    assert o["entities"]["medications"] == [] and "medication" in o["missing_slots"]


def test_submit_request_roundtrip(tmp_path, monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "REQUESTS_LOG", tmp_path / "r.jsonl")
    o = ask("Je voudrais commander 2 boîtes de Spasfon à Rabat")
    r = client.post("/v1/requests", json={"request": o["pharmacy_request"], "pharmacy_id": "P001"})
    assert r.status_code == 201 and r.json()["status"] == "sent_to_pharmacy"


def test_catalog_endpoints():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/medications/search", params={"q": "dolipran"}).json()[0]["brand"] == "doliprane"
    assert client.get("/v1/pharmacies", params={"city": "casa"}).json()


def test_voice_endpoint_with_stubbed_asr(monkeypatch):
    import app.main as m
    monkeypatch.setattr(m.asr, "transcribe", lambda audio, fn, lang=None: {
        "text": "بغيت جوج علب ديال دوليبران 1000", "language": "ar", "backend": "stub", "model": "stub"})
    r = client.post("/v1/analyze/voice", files={"audio": ("v.webm", b"fake-bytes", "audio/webm")})
    o = r.json()
    assert r.status_code == 200 and o["input"]["channel"] == "voice"
    assert o["entities"]["medications"][0]["resolved_code"] == "S002"
    assert o["entities"]["medications"][0]["quantity"] == 2


def test_variant_answer_by_second_dose_number_and_by_index():
    from app.data import pretty_dose
    assert pretty_dose("1 / 125 G / MG") == "1 g / 125 mg"
    o = ask("pourriez-vous me donner le prix du doliprane", "v1")
    assert "medication_variant" in o["missing_slots"] and "1)" in o["reply"]
    o = ask("2", "v1")          # 2nd option in the numbered list
    assert o["entities"]["medications"][0]["resolved_code"] is not None


def test_unrecognised_variant_answer_repeats_question():
    ask("je voudrais du doliprane", "v2")
    o = ask("euh", "v2")
    assert o["intent"]["name"] == "order_medication" and "medication_variant" in o["missing_slots"]
