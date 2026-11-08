"""LLM fallback: the Groq call is mocked; we test when it is used and that its output stays grounded."""
import pytest

from app import llm
from app.dialog import Assistant


@pytest.fixture
def bot(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "enabled", lambda: True)

    def fake(resp):
        def _f(text, timeout=8.0):
            calls.append(text)
            return llm.validate(resp) if resp is not None else None
        monkeypatch.setattr(llm, "interpret", _f)
    a = Assistant()
    a.calls, a.fake = calls, fake
    return a


def test_confident_rules_do_not_call_llm(bot):
    bot.fake({"intent": "out_of_scope", "canonical_fr": "", "confidence": 1})
    o = bot.handle("bghit jouj d les boites doliprane 1000 f rbat")
    assert bot.calls == [] and o["nlu_source"] == "rules" and o["intent"]["name"] == "order_medication"


def test_out_of_scope_medical_question(bot):
    bot.fake({"intent": "out_of_scope", "language": "darija_latin",
              "canonical_fr": "j'ai mal à la tête que dois-je prendre", "medications": [], "confidence": 0.9})
    o = bot.handle("rassi kayderni bzzaf, achmen dwa mzyan liya?")
    assert bot.calls and o["intent"]["name"] == "out_of_scope" and o["nlu_source"] == "rules+llm"
    assert o["pharmacy_request"] is None and "pharmacien" in o["reply"].lower() + "pharmacien"


def test_llm_rewrite_is_regrounded_in_catalog(bot):
    bot.fake({"intent": "order_medication", "language": "darija",
              "canonical_fr": "je veux commander 2 boîtes de glucophage 850", "medications": ["glucophage"],
              "confidence": 0.9})
    o = bot.handle("عافاك داك الدوا ديال السكر الكلوكوفاج تمنمية وخمسين جوج")
    m = o["entities"]["medications"][0]
    assert o["intent"]["name"] == "order_medication"
    assert m["resolved_code"] == "S015" and m["quantity"] == 2


def test_llm_cannot_invent_a_drug(bot):
    bot.fake({"intent": "order_medication", "canonical_fr": "je veux du supermedocx 500",
              "medications": ["supermedocx"], "confidence": 0.95})
    o = bot.handle("wa7d dwa jdid smito supermedocx")
    assert o["entities"]["medications"] == [] and "medication" in o["missing_slots"]


def test_short_followup_skips_llm(bot):
    bot.fake({"intent": "out_of_scope", "canonical_fr": "", "confidence": 1})
    bot.handle("je voudrais du doliprane", "f1")
    o = bot.handle("1000", "f1")
    assert bot.calls == [] and o["entities"]["medications"][0]["resolved_code"] == "S002"


def test_llm_failure_keeps_rule_result(bot):
    bot.fake(None)
    o = bot.handle("hmm wa9ila chi haja")
    assert o["nlu_source"] == "rules" and o["intent"]["name"] == "unknown"


def test_validate_rejects_bad_payload():
    assert llm.validate({"intent": "delete_database"}) is None
    assert llm.validate({"intent": "ask_price", "confidence": "high"})["confidence"] == 0.5


def test_rules_block_medical_advice_without_llm(bot):
    bot.fake(None)
    for t in ["Est-ce que je peux prendre du Xanax avec de l'alcool ?", "شحال من حبة ناخد ديال الدوليبران فالنهار؟"]:
        o = bot.handle(t)
        assert o["intent"]["name"] == "out_of_scope" and o["pharmacy_request"] is None
