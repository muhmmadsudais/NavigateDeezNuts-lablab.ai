"""Tests for app/ai.py: JSON extraction, fallbacks, model picking and garbage LLM output."""

import sqlite3
from datetime import date, timedelta

import pytest

from app import ai, db, logic

TODAY = date(2026, 3, 2)


@pytest.fixture()
def data():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    db.seed(conn, TODAY)
    snap = db.load_snapshot(conn)
    conn.close()
    return snap


@pytest.fixture()
def headline_facts(data):
    s = logic.get_shipment(data, "SH-1042")
    s["current_eta"] = (TODAY + timedelta(days=17)).isoformat()
    s["status"] = "delayed"
    return logic.full_analysis(data, "SH-1042", TODAY)["facts"]


@pytest.fixture(autouse=True)
def clear_cache():
    ai._cache.clear()
    ai._cooldown_until = 0.0
    yield


def test_extract_json_variants():
    assert ai.extract_json('{"a": 1}') == {"a": 1}
    assert ai.extract_json('```json\n{"a": "x"}\n```') == {"a": "x"}
    assert ai.extract_json('Sure! Here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert ai.extract_json('text {"s": "brace } inside"} more') == {"s": "brace } inside"}
    assert ai.extract_json("{bad json} then {\"ok\": true}") == {"ok": True}
    for garbage in [None, "", "no json here", "[1, 2, 3]", "{unclosed"]:
        assert ai.extract_json(garbage) is None


def test_pick_model_prefers_instruct_chat_models():
    names = ["text-embedding-3", "whisper-large", "deepseek-r1", "mistral-7b-instruct", "llama-3.1-70b-instruct"]
    assert ai.pick_model(names) == "mistral-7b-instruct"
    assert ai.pick_model(["qwen2.5-coder", "Qwen2.5-32B-Instruct"]) == "Qwen2.5-32B-Instruct"
    assert ai.pick_model(["bge-embed"]) is None
    assert ai.pick_model([]) is None


def text_model(model_id, efforts=("minimal", "low")):
    return {"id": model_id, "output_modalities": [{"type": "text"}],
            "reasoning": {"mandatory": False, "supported_efforts": list(efforts)}}


def test_pick_model_uses_vultr_metadata():
    # Shaped like the real https://api.vultrinference.com/v1/models catalog.
    catalog = [
        {"id": "chatterbox", "output_modalities": [{"type": "speech"}]},
        {"id": "qwen3-embedding-4b", "output_modalities": [{"type": "embeddings"}]},
        {"id": "clef", "output_modalities": [{"type": "decisions"}]},
        text_model("deepseek-v4.1-flash-1017"),
        text_model("nemotron-3.5-content-safety"),
        text_model("qwen3.8-27b"),
        text_model("qwen3.8-flash-next"),
    ]
    assert ai.pick_model(catalog) == "qwen3.8-flash-next"
    assert ai.pick_model(catalog[:3]) is None
    assert ai._reasoning_effort(catalog[-1]) == "minimal"
    assert ai._reasoning_effort(text_model("x", efforts=("high", "low"))) == "low"
    assert ai._reasoning_effort({"id": "plain"}) is None


def test_no_api_key_means_fallback(headline_facts):
    assert ai.status()["configured"] is False
    text, source = ai.recommend(headline_facts)
    assert source == "fallback"
    assert "ChipSource Asia" in text and "1,786" in text and "SH-1042" in text


def test_fallback_recommendation_no_stockout(data):
    facts = logic.full_analysis(data, "SH-1044", TODAY)["facts"]
    text = ai.fallback_recommendation(facts)
    assert "no customer orders are at risk" in text


def test_fallback_recommendation_no_alternatives(headline_facts):
    facts = dict(headline_facts, best_alternative_supplier="none available")
    assert "No alternative supplier" in ai.fallback_recommendation(facts)
    assert ai.fallback_recommendation(headline_facts, simulation=True).startswith("In this hypothetical scenario")


def test_fallback_emails(headline_facts):
    emails, source = ai.draft_emails(headline_facts)
    assert source == "fallback"
    assert emails["supplier_email"].startswith("Subject:") and "SH-1042" in emails["supplier_email"]
    assert "ORD-2001" in emails["customer_email"]
    assert "$" not in emails["customer_email"]  # never leak internal prices to customers


def test_fallback_parse_sample_email(data):
    fields, source = ai.parse_email(ai.sample_email(data, TODAY), data, TODAY)
    assert source == "fallback"
    assert fields["supplier"] == "TechParts Ltd."
    assert fields["product"] == "Microchips"
    assert fields["quantity"] == 5000
    assert fields["shipment_id"] == "SH-1042"
    assert fields["new_delivery_date"] == (TODAY + timedelta(days=17)).isoformat()
    assert "power outage" in fields["reason"]
    assert logic.match_shipment(data, fields)["id"] == "SH-1042"


@pytest.mark.parametrize("text,expected", [
    ("New ETA is 2026-04-01.", "2026-04-01"),
    ("We can deliver on 3rd April 2026 instead.", "2026-04-03"),
    ("Delivery moved to Apr 5, 2026", "2026-04-05"),
    ("Revised date: 07/04/2026", "2026-04-07"),
])
def test_fallback_date_formats(data, text, expected):
    assert ai.fallback_parse_email(text, data)["new_delivery_date"] == expected


def test_fallback_parse_relative_delay_and_cancel(data):
    fields = ai.fallback_parse_email("PowerCell here: the 4,000 batteries are delayed by 6 days due to a strike.", data)
    assert fields["supplier"] == "PowerCell Inc." and fields["product"] == "Batteries"
    assert fields["quantity"] == 4000 and fields["delay_days"] == 6 and fields["reason"] == "a strike"
    assert logic.match_shipment(data, fields)["id"] == "SH-1043"
    assert ai.fallback_parse_email("Order SH-1046 has been cancelled.", data)["cancelled"] is True


def test_fallback_parse_garbage_never_crashes(data):
    for text in ["", "!!!", "x" * 5000, "{\"supplier\": 3}", "SH-"]:
        fields = ai.fallback_parse_email(text, data)
        assert set(fields) >= {"supplier", "product", "quantity", "new_delivery_date", "reason"}


def test_llm_garbage_output_falls_back(monkeypatch, data, headline_facts):
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: "I am not JSON at all {{{")
    fields, source = ai.parse_email(ai.sample_email(data, TODAY), data, TODAY)
    assert source == "fallback" and fields["quantity"] == 5000
    emails, source = ai.draft_emails(headline_facts)
    assert source == "fallback"
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: "ok")  # too short to be a real recommendation
    assert ai.recommend(headline_facts)[1] == "fallback"


def test_llm_bad_field_values_are_ignored(monkeypatch, data):
    reply = '```json\n{"supplier": "TechParts Ltd.", "product": null, "quantity": "lots", ' \
            '"new_delivery_date": "next week", "reason": "storm", "shipment_id": "DROP TABLE"}\n```'
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: reply)
    fields, source = ai.parse_email(ai.sample_email(data, TODAY), data, TODAY)
    assert source == "vultr"
    assert fields["reason"] == "storm"
    assert fields["quantity"] == 5000  # regex value kept, LLM's "lots" rejected
    assert fields["shipment_id"] == "SH-1042"
    assert fields["new_delivery_date"] == (TODAY + timedelta(days=17)).isoformat()


def test_llm_success_paths(monkeypatch, headline_facts):
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: "Order 1,786 units from ChipSource Asia now. " * 2)
    text, source = ai.recommend(headline_facts)
    assert source == "vultr" and "ChipSource" in text
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: '{"supplier_email": "Subject: A\\n' + "x" * 50 +
                        '", "customer_email": "Subject: B\\n' + "y" * 50 + '"}')
    emails, source = ai.draft_emails(headline_facts)
    assert source == "vultr" and emails["customer_email"].startswith("Subject: B")


def test_chat_failure_is_caught(monkeypatch):
    class Broken:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise TimeoutError("network down")

    monkeypatch.setattr(ai, "_get_client", lambda: Broken())
    monkeypatch.setattr(ai, "_model", "some-model")
    assert ai._chat([{"role": "user", "content": "hi"}]) is None
    assert ai._cooldown_until > 0  # skips the LLM for a while so pages stay fast


def test_chat_retries_without_reasoning_effort_on_400(monkeypatch):
    calls = []

    class BadRequest(Exception):
        status_code = 400

    class Message:
        content = "<think>hmm</think>Hello from the model"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    if kwargs.get("extra_body"):
                        raise BadRequest()
                    return Response()

    monkeypatch.setattr(ai, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(ai, "_model", "some-model")
    monkeypatch.setattr(ai, "_effort", "minimal")
    assert ai._chat([{"role": "user", "content": "hi"}]) == "Hello from the model"
    assert calls[0]["extra_body"] == {"reasoning_effort": "minimal"} and "extra_body" not in calls[1]
    assert ai._effort is None
