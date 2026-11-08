"""Smoke tests for every page and workflow, with the LLM disabled (offline fallback)."""

import html
import re

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:  # the context manager runs the lifespan (DB init + seed)
        c.post("/reset")
        yield c


def sample_email_text(client) -> str:
    page = client.get("/ingest?sample=1").text
    return html.unescape(re.search(r"<textarea[^>]*>(.*?)</textarea>", page, re.S).group(1))


def delay_sh1042(client):
    """Run the real ingest flow with the sample email (SH-1042 delayed 10 days)."""
    parsed = client.post("/ingest", data={"email_text": sample_email_text(client)})
    new_date = re.search(r'name="new_delivery_date" value="([^"]*)"', parsed.text).group(1)
    return client.post("/ingest/confirm", data={"shipment_id": "SH-1042", "new_delivery_date": new_date,
                                                "status": "delayed", "reason": "power outage"})


def query(sql, params=()):
    with db.get_conn() as conn:
        return db.rows(conn, sql, params)


@pytest.mark.parametrize("url", ["/", "/shipments/SH-1042", "/shipments/SH-1044", "/shipments/SH-1045",
                                 "/simulator", "/actions", "/ingest", "/ingest?sample=1", "/activity",
                                 "/partials/activity", "/health", "/api/shipments", "/static/style.css"])
def test_every_page_returns_200(client, url):
    response = client.get(url)
    assert response.status_code == 200
    if response.headers["content-type"].startswith("text/html"):
        assert "NavigateDeezNuts" in response.text or url.startswith("/partials")
        assert "Traceback" not in response.text


def test_health_and_api(client):
    assert client.get("/health").json() == {"status": "ok"}
    data = client.get("/api/shipments").json()
    assert data["count"] == 10
    assert {"id", "status", "delay_days", "at_risk"} <= set(data["shipments"][0])


def test_unknown_ids_give_friendly_404(client):
    for url in ["/shipments/SH-0000", "/does-not-exist"]:
        r = client.get(url)
        assert r.status_code == 404 and "Back to dashboard" in r.text
    assert client.post("/shipments/SH-0000/plan").status_code == 404
    assert client.post("/actions/9999/approve").status_code == 404
    assert client.post("/actions/abc/approve").status_code == 400
    assert client.post("/actions/1/explode").status_code == 404
    assert client.get("/api/nope").status_code == 404


def test_dashboard_initially_healthy(client):
    page = client.get("/").text
    assert "Stockout risk" not in page  # only a minor delay at start
    assert "SH-1042" in page and "Latest agent activity" in page
    assert "AI offline fallback" in page or "offline fallback" in page


def test_reset_restores_seed_data(client):
    delay_sh1042(client)
    assert query("SELECT status FROM shipments WHERE id = 'SH-1042'")[0]["status"] == "delayed"
    r = client.post("/reset", follow_redirects=False)
    assert r.status_code == 303
    assert query("SELECT status FROM shipments WHERE id = 'SH-1042'")[0]["status"] == "on_time"
    assert query("SELECT COUNT(*) AS n FROM actions")[0]["n"] == 0


def test_ingest_with_fallback_parser_updates_shipment(client):
    parsed = client.post("/ingest", data={"email_text": sample_email_text(client)})
    assert parsed.status_code == 200
    assert "AI: offline fallback" in parsed.text
    assert "TechParts Ltd." in parsed.text and "5,000" in parsed.text
    assert '<option value="SH-1042" selected' in parsed.text

    r = delay_sh1042(client)
    assert r.status_code == 200 and "/shipments/SH-1042" in str(r.url)
    s = query("SELECT * FROM shipments WHERE id = 'SH-1042'")[0]
    assert s["status"] == "delayed" and s["current_eta"] > s["expected_date"]
    steps = {row["step"] for row in query("SELECT step FROM activity_log")}
    assert {"Ingested", "Detected", "Analyzed", "Sought alternatives", "Recommended"} <= steps
    # impact page now shows the headline scenario
    assert "Stockout risk" in r.text and "ChipSource Asia" in r.text and "ORD-2001" in r.text


def test_ingest_validation(client):
    assert client.post("/ingest", data={"email_text": "  "}).status_code == 400
    r = client.post("/ingest", data={"email_text": "hello, nothing useful here"})
    assert r.status_code == 200 and "No match" in r.text
    r = client.post("/ingest/confirm", data={"shipment_id": "SH-1042", "new_delivery_date": "not-a-date"})
    assert "valid new delivery date" in r.text
    r = client.post("/ingest/confirm", data={"shipment_id": "SH-9999", "new_delivery_date": "2030-01-01"})
    assert "choose which shipment" in r.text


def test_action_plan_approve_and_reject(client):
    delay_sh1042(client)
    r = client.post("/shipments/SH-1042/plan")
    assert r.status_code == 200 and "Pending approval" in r.text
    actions = {a["type"]: a for a in query("SELECT * FROM actions")}
    assert set(actions) == {"reorder_alternative", "reserve_inventory", "request_commitment", "notify_customers"}
    assert "Subject:" in r.text  # drafted emails are visible

    # Approve reorder -> a new shipment appears and the stockout risk is gone.
    client.post(f"/actions/{actions['reorder_alternative']['id']}/approve")
    new = query("SELECT * FROM shipments WHERE notes LIKE 'Emergency reorder%'")
    assert len(new) == 1 and new[0]["quantity"] == 1786 and new[0]["supplier_id"] == 2
    assert not [s for s in client.get("/api/shipments").json()["shipments"] if s["at_risk"]]

    # Approve notify -> orders marked as customer notified.
    client.post(f"/actions/{actions['notify_customers']['id']}/approve")
    statuses = {o["id"]: o["status"] for o in query("SELECT * FROM customer_orders")}
    assert statuses["ORD-2001"] == "customer notified" and statuses["ORD-2002"] == "customer notified"

    # Reject -> status rejected, no state change.
    client.post(f"/actions/{actions['request_commitment']['id']}/reject")
    decided = {a["type"]: a["status"] for a in query("SELECT * FROM actions")}
    assert decided["request_commitment"] == "rejected"
    assert decided["reorder_alternative"] == "approved"

    # Deciding twice is handled gracefully.
    r = client.post(f"/actions/{actions['request_commitment']['id']}/approve")
    assert r.status_code == 200 and "already rejected" in r.text


def test_plan_for_healthy_shipment_is_not_created(client):
    r = client.post("/shipments/SH-1043/plan")
    assert r.status_code == 200 and "no action plan is needed" in r.text
    assert query("SELECT COUNT(*) AS n FROM actions")[0]["n"] == 0


@pytest.mark.parametrize("scenario", ["shipment_delay", "supplier_unavailable", "cost_increase"])
def test_simulator_runs_and_does_not_change_db(client, scenario):
    before = query("SELECT * FROM shipments ORDER BY id")
    r = client.post("/simulator", data={"scenario": scenario, "shipment_id": "SH-1042", "supplier_id": "1",
                                        "product_id": "1", "days": "10", "percent": "15"})
    assert r.status_code == 200
    assert "SIMULATION - HYPOTHETICAL" in r.text
    assert query("SELECT * FROM shipments ORDER BY id") == before


def test_simulator_headline_numbers(client):
    r = client.post("/simulator", data={"scenario": "shipment_delay", "shipment_id": "SH-1042", "days": "10"})
    assert "ChipSource Asia" in r.text and "ORD-2001" in r.text and "ORD-2002" in r.text


@pytest.mark.parametrize("form", [{}, {"scenario": "bogus"}, {"scenario": "shipment_delay", "days": "abc"},
                                  {"scenario": "cost_increase", "product_id": "1", "percent": "x"}])
def test_simulator_bad_input_is_friendly(client, form):
    r = client.post("/simulator", data=form)
    assert r.status_code == 200 and "flash-err" in r.text


def test_scan_logs_detection(client):
    r = client.post("/scan")
    assert r.status_code == 200 and "Agent scan complete" in r.text
    assert query("SELECT COUNT(*) AS n FROM activity_log WHERE step = 'Detected'")[0]["n"] == 1


def test_csv_upload_valid_and_invalid(client):
    good = "product_id,on_hand\n1,9000\n"
    r = client.post("/upload", data={"kind": "inventory"}, files={"file": ("inv.csv", good, "text/csv")})
    assert r.status_code == 200 and "Imported 1 inventory row" in r.text
    assert query("SELECT on_hand FROM inventory WHERE product_id = 1")[0]["on_hand"] == 9000

    bad = "id,supplier_id,product_id,quantity,expected_date,current_eta,status\nSH-2000,1,1,-5,tomorrow,2026-01-01,lost\n"
    r = client.post("/upload", data={"kind": "shipments"}, files={"file": ("s.csv", bad, "text/csv")})
    assert r.status_code == 400 and "Row 2" in r.text and "nothing is imported" in r.text
    assert not query("SELECT * FROM shipments WHERE id = 'SH-2000'")

    r = client.post("/upload", data={"kind": "shipments"}, files={"file": ("s.csv", "a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400 and "Missing column" in r.text


def test_empty_tables_do_not_crash(client):
    with db.get_conn() as conn:
        for table in ["actions", "customer_orders", "shipments", "suppliers", "inventory", "products", "activity_log"]:
            conn.execute(f"DELETE FROM {table}")
    for url in ["/", "/simulator", "/actions", "/activity", "/api/shipments", "/ingest"]:
        assert client.get(url).status_code == 200
    assert client.post("/simulator", data={"scenario": "shipment_delay", "shipment_id": "SH-1042",
                                           "days": "3"}).status_code == 200
    assert client.post("/ingest", data={"email_text": "SH-1042 delayed"}).status_code == 200


def test_unexpected_error_shows_friendly_page(monkeypatch):
    from app import main

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(main, "dashboard_context", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/")
    assert r.status_code == 500
    assert "Unexpected error" in r.text and "kaboom" not in r.text and "Traceback" not in r.text
