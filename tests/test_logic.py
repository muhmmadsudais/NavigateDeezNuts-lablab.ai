"""Unit tests for app/logic.py, including the headline SH-1042 scenario."""

import copy
import sqlite3
from datetime import date, timedelta

import pytest

from app import db, logic

TODAY = date(2026, 3, 2)  # fixed date so results never depend on when tests run


def make_data():
    """Seed an in-memory DB for TODAY and return the snapshot."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    db.seed(conn, TODAY)
    data = db.load_snapshot(conn)
    conn.close()
    return data


def delay(data, shipment_id, days):
    s = logic.get_shipment(data, shipment_id)
    s["current_eta"] = (logic.to_date(s["expected_date"]) + timedelta(days=days)).isoformat()
    s["status"] = "delayed"
    return data


def day(n):
    return TODAY + timedelta(days=n)


# ---------------------------------------------------------------- helpers

def test_to_date_handles_bad_input():
    assert logic.to_date("2026-03-02") == date(2026, 3, 2)
    assert logic.to_date("not a date") is None
    assert logic.to_date(None) is None
    assert logic.to_date("") is None


# ---------------------------------------------------------------- detect_disruptions

def test_detect_disruptions_initial_state_has_one_minor_delay():
    found = logic.detect_disruptions(make_data())
    assert [d["id"] for d in found] == ["SH-1044"]
    assert found[0]["delay_days"] == 2


def test_detect_disruptions_includes_delayed_and_cancelled():
    data = delay(make_data(), "SH-1042", 10)
    logic.get_shipment(data, "SH-1046")["status"] = "cancelled"
    ids = {d["id"]: d["delay_days"] for d in logic.detect_disruptions(data)}
    assert ids["SH-1042"] == 10
    assert "SH-1046" in ids
    assert "SH-1045" not in ids  # delivered


def test_detect_disruptions_empty_tables():
    assert logic.detect_disruptions({}) == []
    assert logic.detect_disruptions({"shipments": []}) == []


# ---------------------------------------------------------------- project_stock

def test_project_stock_healthy_microchips_never_run_out():
    result = logic.project_stock(make_data(), 1, TODAY)
    assert result["stockout_date"] is None
    assert len(result["points"]) == logic.HORIZON_DAYS + 1
    assert result["points"][0]["stock"] == 1500
    # SH-1042 (5,000) arrives on day 7: 1500 - 1000 + 5000 = 5500
    assert round(result["points"][7]["stock"]) == 5500


def test_project_stock_excludes_cancelled_shipments():
    data = make_data()
    logic.get_shipment(data, "SH-1042")["status"] = "cancelled"
    assert logic.project_stock(data, 1, TODAY)["stockout_date"] == day(11)


def test_project_stock_unknown_product():
    result = logic.project_stock(make_data(), 999, TODAY)
    assert result["points"] == [] and result["stockout_date"] is None


def test_project_stock_ignores_bad_dates():
    data = make_data()
    logic.get_shipment(data, "SH-1042")["current_eta"] = "garbage"
    assert logic.project_stock(data, 1, TODAY)["stockout_date"] == day(11)


# ---------------------------------------------------------------- impact_analysis (headline scenario)

def test_headline_sh1042_delay_causes_stockout_before_new_eta():
    impact = logic.impact_analysis(delay(make_data(), "SH-1042", 10), "SH-1042", TODAY)
    assert impact["delay_days"] == 10
    assert impact["arrival_date"] == day(17)
    assert impact["stockout_without_delay"] is None
    assert impact["stockout_with_delay"] == day(11)
    assert impact["stockout_with_delay"] < impact["arrival_date"]
    assert impact["stockout_before_arrival"] and impact["at_risk"] and impact["caused_by_delay"]
    assert impact["recovery_date"] == day(17)
    assert impact["production_shortfall"] == 786  # ceil(16 * 1000/7 - 1500)


def test_headline_two_orders_at_risk_one_high_priority():
    impact = logic.impact_analysis(delay(make_data(), "SH-1042", 10), "SH-1042", TODAY)
    at_risk = impact["orders_at_risk"]
    assert [o["id"] for o in at_risk] == ["ORD-2001", "ORD-2002"]
    assert [o["priority"] for o in at_risk] == ["high", "medium"]
    assert [o["days_late"] for o in at_risk] == [5, 2]
    assert impact["orders_at_risk_qty"] == 1000
    assert impact["shortage_qty"] == 1786


def test_impact_no_delay_means_no_risk():
    impact = logic.impact_analysis(make_data(), "SH-1042", TODAY)
    assert impact["delay_days"] == 0
    assert impact["stockout_with_delay"] is None
    assert impact["orders_at_risk"] == [] and impact["shortage_qty"] == 0
    assert not impact["at_risk"]


def test_impact_minor_screen_delay_is_not_at_risk():
    impact = logic.impact_analysis(make_data(), "SH-1044", TODAY)
    assert impact["is_disrupted"] and impact["delay_days"] == 2
    assert not impact["at_risk"]


def test_impact_unknown_shipment_returns_none():
    assert logic.impact_analysis(make_data(), "SH-9999", TODAY) is None
    assert logic.full_analysis(make_data(), "nope", TODAY) is None


def test_impact_does_not_mutate_input():
    data = delay(make_data(), "SH-1042", 10)
    before = copy.deepcopy(data)
    logic.impact_analysis(data, "SH-1042", TODAY)
    assert data == before


# ---------------------------------------------------------------- rank_alternatives

def test_headline_alternative_arrives_in_time_and_is_cheapest():
    alts = logic.rank_alternatives(make_data(), 1, 1786, day(11), TODAY)
    assert [a["supplier"] for a in alts] == ["ChipSource Asia", "RapidChip EU"]
    best = alts[0]
    assert best["arrival_date"] == day(4) and best["arrives_in_time"]
    assert best["available_qty"] == 2000 and best["coverage_qty"] == 1786 and best["covers_full"]
    assert best["total_cost"] == pytest.approx(1786 * 3.10)
    assert best["cost_premium"] == pytest.approx(1786 * (3.10 - 2.50))
    assert best["rank"] == 1


def test_rank_alternatives_prefers_full_coverage():
    # 2,500 units: ChipSource can only do 2,000, RapidChip can do all of it.
    alts = logic.rank_alternatives(make_data(), 1, 2500, day(11), TODAY)
    assert alts[0]["supplier"] == "RapidChip EU" and alts[0]["covers_full"]


def test_rank_alternatives_late_supplier_ranked_last():
    alts = logic.rank_alternatives(make_data(), 1, 1000, day(4), TODAY)  # stockout on day 4
    assert alts[0]["supplier"] == "RapidChip EU"  # 3-day lead time, the only one in time
    assert not alts[1]["arrives_in_time"]


def test_rank_alternatives_none_available_and_zero_shortage():
    assert logic.rank_alternatives(make_data(), 4, 500, day(5), TODAY) == []  # casings: primary only
    zero = logic.rank_alternatives(make_data(), 1, 0, None, TODAY)
    assert all(a["coverage_qty"] == 0 and a["total_cost"] == 0 for a in zero)


# ---------------------------------------------------------------- full_analysis / facts

def test_full_analysis_facts_are_strings_and_numbers_only():
    analysis = logic.full_analysis(delay(make_data(), "SH-1042", 10), "SH-1042", TODAY)
    facts = analysis["facts"]
    assert facts["best_alternative_supplier"] == "ChipSource Asia"
    assert facts["orders_at_risk"] == 2
    assert facts["stockout_before_shipment_arrives"] == "yes"
    assert all(isinstance(v, (str, int)) for v in facts.values())


# ---------------------------------------------------------------- build_action_plan

def test_build_action_plan_headline():
    data = delay(make_data(), "SH-1042", 10)
    plan = logic.build_action_plan(data, "SH-1042", TODAY, {"supplier_email": "S", "customer_email": "C"})
    types = [a["type"] for a in plan]
    assert types == ["reorder_alternative", "reserve_inventory", "request_commitment", "notify_customers"]
    reorder = plan[0]
    assert reorder["payload"]["supplier"] == "ChipSource Asia"
    assert reorder["payload"]["quantity"] == 1786
    assert reorder["estimated_cost"] == pytest.approx(5536.6)
    assert plan[1]["payload"]["order_ids"] == ["ORD-2001"]
    assert plan[2]["payload"]["email"] == "S"
    assert plan[3]["payload"]["order_ids"] == ["ORD-2001", "ORD-2002"]


def test_build_action_plan_healthy_or_unknown_is_empty():
    assert logic.build_action_plan(make_data(), "SH-1042", TODAY) == []
    assert logic.build_action_plan(make_data(), "SH-0000", TODAY) == []


def test_build_action_plan_no_alternatives_still_requests_commitment():
    data = make_data()
    for s in data["suppliers"]:
        if s["product_id"] == 1 and not s["is_primary"]:
            s["available_qty"] = 0
    plan = logic.build_action_plan(delay(data, "SH-1042", 10), "SH-1042", TODAY)
    types = [a["type"] for a in plan]
    assert "reorder_alternative" not in types
    assert "request_commitment" in types and "notify_customers" in types


def test_build_action_plan_minor_delay_has_no_reorder():
    plan = logic.build_action_plan(make_data(), "SH-1044", TODAY)
    assert [a["type"] for a in plan] == ["request_commitment"]


# ---------------------------------------------------------------- simulate

def test_simulate_shipment_delay_matches_headline_and_does_not_touch_data():
    data = make_data()
    before = copy.deepcopy(data)
    result = logic.simulate(data, {"type": "shipment_delay", "shipment_id": "SH-1042", "days": 10}, TODAY)
    assert data == before
    assert result["label"] == "SIMULATION - hypothetical"
    assert result["error"] is None
    analysis = result["analyses"][0]
    assert analysis["stockout_with_delay"] == day(11)
    assert len(analysis["orders_at_risk"]) == 2
    assert analysis["best_alternative"]["supplier"] == "ChipSource Asia"


def test_simulate_supplier_unavailable():
    result = logic.simulate(make_data(), {"type": "supplier_unavailable", "supplier_id": 1, "days": 10}, TODAY)
    assert result["error"] is None
    assert [a["shipment"]["id"] for a in result["analyses"]] == ["SH-1042", "SH-1047"]
    assert result["analyses"][0]["at_risk"]


def test_simulate_supplier_without_shipments():
    result = logic.simulate(make_data(), {"type": "supplier_unavailable", "supplier_id": 2, "days": 5}, TODAY)
    assert result["analyses"] == [] and "no open shipments" in result["error"]


def test_simulate_cost_increase():
    result = logic.simulate(make_data(), {"type": "cost_increase", "product_id": 1, "percent": 20}, TODAY)
    ci = result["cost_impact"]
    assert ci["old_unit_price"] == 2.50 and ci["new_unit_price"] == 3.00
    assert ci["extra_cost_per_week"] == pytest.approx(500.0)
    assert ci["extra_cost_open_shipments"] == pytest.approx(9000 * 0.5)
    # alternatives are re-priced too
    assert result["analyses"][0]["alternatives"][0]["unit_price"] == pytest.approx(3.72)


@pytest.mark.parametrize("event", [
    {},
    {"type": "bogus"},
    {"type": "shipment_delay", "shipment_id": "SH-1042", "days": "abc"},
    {"type": "shipment_delay", "shipment_id": "SH-1042", "days": 0},
    {"type": "shipment_delay", "shipment_id": "SH-404", "days": 3},
    {"type": "shipment_delay", "shipment_id": "SH-1045", "days": 3},
    {"type": "supplier_unavailable", "supplier_id": 999, "days": 3},
    {"type": "cost_increase", "product_id": 999, "percent": 10},
    {"type": "cost_increase", "product_id": 1, "percent": None},
])
def test_simulate_bad_input_returns_error_not_exception(event):
    result = logic.simulate(make_data(), event, TODAY)
    assert result["error"]
    assert result["label"] == "SIMULATION - hypothetical"


# ---------------------------------------------------------------- shipment_overview

def test_shipment_overview_flags_at_risk_first():
    rows = logic.shipment_overview(delay(make_data(), "SH-1042", 10), TODAY)
    assert rows[0]["id"] == "SH-1042" and rows[0]["at_risk"]
    assert rows[1]["id"] == "SH-1044" and rows[1]["disrupted"] and not rows[1]["at_risk"]
    assert sum(r["at_risk"] for r in rows) == 1
