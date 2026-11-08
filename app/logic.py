"""All supply chain calculations live here, as plain Python functions.

Every function works on a *snapshot* dict (see db.load_snapshot):
    {"products": [...], "inventory": [...], "suppliers": [...],
     "shipments": [...], "customer_orders": [...]}
Nothing in this module touches the database, which keeps it easy to unit
test and lets the simulator run on a copy of the data.

The AI never does arithmetic: every number shown to users is computed here.
"""

import copy
import math
from datetime import date, timedelta

HORIZON_DAYS = 60
SIMULATION_LABEL = "SIMULATION - hypothetical"
OPEN_ORDER_EXCLUDE = {"delivered", "cancelled"}
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------- small helpers

def to_date(value) -> date | None:
    """Parse an ISO date string (or date) safely. Returns None for bad input."""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (ValueError, TypeError):
        return None


def find(items: list[dict], key: str, value) -> dict | None:
    for item in items:
        if str(item.get(key)) == str(value):
            return item
    return None


def get_product(data: dict, product_id) -> dict | None:
    return find(data.get("products", []), "id", product_id)


def get_supplier(data: dict, supplier_id) -> dict | None:
    return find(data.get("suppliers", []), "id", supplier_id)


def get_shipment(data: dict, shipment_id) -> dict | None:
    return find(data.get("shipments", []), "id", shipment_id)


def on_hand(data: dict, product_id) -> int:
    inv = find(data.get("inventory", []), "product_id", product_id)
    return int(inv["on_hand"]) if inv else 0


def primary_supplier(data: dict, product_id) -> dict | None:
    for s in data.get("suppliers", []):
        if str(s["product_id"]) == str(product_id) and s.get("is_primary"):
            return s
    return None


def delay_days(shipment: dict) -> int:
    expected, eta = to_date(shipment.get("expected_date")), to_date(shipment.get("current_eta"))
    if expected is None or eta is None:
        return 0
    return max(0, (eta - expected).days)


def is_disrupted(shipment: dict) -> bool:
    if shipment.get("status") == "delivered":
        return False
    return shipment.get("status") in ("delayed", "cancelled") or delay_days(shipment) > 0


# ---------------------------------------------------------------- 1. detection

def detect_disruptions(data: dict) -> list[dict]:
    """Shipments that are late (ETA after promise) or flagged delayed/cancelled."""
    result = []
    for s in data.get("shipments", []):
        if is_disrupted(s):
            item = dict(s)
            item["delay_days"] = delay_days(s)
            result.append(item)
    return result


# ---------------------------------------------------------------- 2. stock projection

def project_stock(data: dict, product_id, today: date | None = None, horizon: int = HORIZON_DAYS) -> dict:
    """Day-by-day stock projection for one product.

    stock(day) = on_hand - daily_usage * day + all shipments that arrive by that day
    (cancelled and already-delivered shipments are excluded; delivered stock is
    already counted in on_hand). Returns the points and the first day stock < 0.
    """
    today = today or date.today()
    product = get_product(data, product_id)
    if product is None:
        return {"points": [], "stockout_date": None, "daily_usage": 0.0, "on_hand": 0}

    daily_usage = float(product["weekly_usage"]) / 7.0
    start = on_hand(data, product_id)

    arrivals: list[tuple[date, int]] = []
    for s in data.get("shipments", []):
        if str(s["product_id"]) != str(product_id) or s.get("status") in ("cancelled", "delivered"):
            continue
        eta = to_date(s.get("current_eta"))
        if eta is None:
            continue
        arrivals.append((max(eta, today), int(s["quantity"])))  # overdue shipments count as arriving today

    points = []
    stockout_date = None
    for day in range(horizon + 1):
        current = today + timedelta(days=day)
        arrived = sum(qty for eta, qty in arrivals if eta <= current)
        stock = start - daily_usage * day + arrived
        points.append({"date": current, "stock": stock})
        if stock < 0 and stockout_date is None:
            stockout_date = current
    return {"points": points, "stockout_date": stockout_date, "daily_usage": daily_usage, "on_hand": start}


# ---------------------------------------------------------------- 3. impact analysis

def impact_analysis(data: dict, shipment_id, today: date | None = None) -> dict | None:
    """What does this shipment's delay do to stock and customer orders?

    Returns None if the shipment does not exist.
    """
    today = today or date.today()
    shipment = get_shipment(data, shipment_id)
    if shipment is None:
        return None
    product_id = shipment["product_id"]
    product = get_product(data, product_id) or {"id": product_id, "name": "Unknown product", "weekly_usage": 0, "unit_cost": 0}

    # Projection as things stand now (with the delay)...
    with_delay = project_stock(data, product_id, today)
    # ...and as it would have been if the supplier had kept its promise.
    on_promise = copy.deepcopy(data)
    original = get_shipment(on_promise, shipment_id)
    if original.get("status") != "delivered":
        original["current_eta"] = original["expected_date"]
        original["status"] = "on_time"
    without_delay = project_stock(on_promise, product_id, today)

    stockout = with_delay["stockout_date"]
    points = with_delay["points"]
    arrival = None if shipment.get("status") in ("cancelled", "delivered") else to_date(shipment.get("current_eta"))

    # Recovery = first day after the stockout where stock is back to >= 0.
    recovery = None
    shortfall = 0
    if stockout is not None:
        window = [p for p in points if p["date"] >= stockout]
        for p in window:
            if p["stock"] >= 0:
                recovery = p["date"]
                break
        deficit = [p["stock"] for p in window if recovery is None or p["date"] < recovery]
        shortfall = math.ceil(-min(deficit)) if deficit else 0

    # Orders due inside the gap [stockout, recovery) compete for whatever stock
    # is left the day before the stockout; allocate high priority first.
    orders_at_risk = []
    if stockout is not None:
        horizon_end = points[-1]["date"] if points else stockout
        gap_end = recovery or horizon_end + timedelta(days=1)
        candidates = []
        for o in data.get("customer_orders", []):
            due = to_date(o.get("due_date"))
            if str(o["product_id"]) != str(product_id) or o.get("status") in OPEN_ORDER_EXCLUDE or due is None:
                continue
            if stockout <= due < gap_end:
                candidates.append((PRIORITY_RANK.get(o.get("priority"), 3), due, o))
        candidates.sort(key=lambda c: (c[0], c[1], str(c[2]["id"])))

        day_before = stockout - timedelta(days=1)
        remaining = max(0, math.floor(next((p["stock"] for p in points if p["date"] == day_before), 0)))
        for _, due, o in candidates:
            qty = int(o["quantity_needed"])
            if qty <= remaining:
                remaining -= qty  # this order can still be served from existing stock
                continue
            item = dict(o)
            item["days_late"] = max(1, (gap_end - due).days)
            orders_at_risk.append(item)

    at_risk_qty = sum(int(o["quantity_needed"]) for o in orders_at_risk)
    stockout_before_arrival = stockout is not None and (arrival is None or stockout < arrival)
    caused_by_delay = stockout is not None and (
        without_delay["stockout_date"] is None or stockout < without_delay["stockout_date"]
    )

    return {
        "shipment": dict(shipment),
        "product": product,
        "supplier": get_supplier(data, shipment["supplier_id"]),
        "delay_days": delay_days(shipment),
        "is_disrupted": is_disrupted(shipment),
        "on_hand": with_delay["on_hand"],
        "daily_usage": with_delay["daily_usage"],
        "arrival_date": arrival,
        "stockout_with_delay": stockout,
        "stockout_without_delay": without_delay["stockout_date"],
        "stockout_before_arrival": stockout_before_arrival,
        "caused_by_delay": caused_by_delay,
        "recovery_date": recovery,
        "production_shortfall": shortfall,
        "orders_at_risk": orders_at_risk,
        "orders_at_risk_qty": at_risk_qty,
        "shortage_qty": shortfall + at_risk_qty if stockout is not None else 0,
        "projection": points,
        "at_risk": is_disrupted(shipment) and stockout_before_arrival,
    }


# ---------------------------------------------------------------- 4. alternatives

def rank_alternatives(data: dict, product_id, shortage_qty: int, needed_by: date | None,
                      today: date | None = None) -> list[dict]:
    """Score every non-primary supplier for this product.

    Ranking: arrives in time first, then covers the full shortage, then lowest
    total cost, then highest reliability.
    """
    today = today or date.today()
    shortage_qty = max(0, int(shortage_qty or 0))
    primary = primary_supplier(data, product_id)
    product = get_product(data, product_id)
    base_price = float(primary["unit_price"]) if primary else float(product["unit_cost"]) if product else 0.0

    result = []
    for s in data.get("suppliers", []):
        if str(s["product_id"]) != str(product_id) or s.get("is_primary"):
            continue
        available = max(0, int(s["available_qty"]))
        coverage = min(available, shortage_qty)
        arrival = today + timedelta(days=int(s["lead_time_days"]))
        price = float(s["unit_price"])
        result.append({
            "supplier_id": s["id"],
            "supplier": s["name"],
            "unit_price": price,
            "lead_time_days": int(s["lead_time_days"]),
            "available_qty": available,
            "reliability": float(s["reliability"]),
            "coverage_qty": coverage,
            "covers_full": shortage_qty > 0 and coverage >= shortage_qty,
            "arrival_date": arrival,
            "total_cost": round(coverage * price, 2),
            "cost_premium": round(coverage * (price - base_price), 2),
            "arrives_in_time": needed_by is None or arrival < needed_by,
        })
    result.sort(key=lambda a: (not a["arrives_in_time"], not a["covers_full"], a["total_cost"], -a["reliability"]))
    for rank, alt in enumerate(result, start=1):
        alt["rank"] = rank
    return result


# ---------------------------------------------------------------- facts for the AI + the UI

def fmt_date(d: date | None) -> str:
    return d.strftime("%a %d %b %Y") if d else "none"


def build_facts(impact: dict, alternatives: list[dict]) -> dict:
    """Flat, human-readable facts. This exact dict is shown in the UI and sent to the AI."""
    s, p = impact["shipment"], impact["product"]
    best = alternatives[0] if alternatives and alternatives[0]["coverage_qty"] > 0 else None
    high = [o for o in impact["orders_at_risk"] if o.get("priority") == "high"]
    facts = {
        "shipment_id": s["id"],
        "supplier": impact["supplier"]["name"] if impact["supplier"] else "Unknown supplier",
        "product": p["name"],
        "shipment_quantity": f"{int(s['quantity']):,}",
        "shipment_status": s["status"],
        "original_delivery_date": fmt_date(to_date(s["expected_date"])),
        "current_eta": fmt_date(impact["arrival_date"]) if impact["arrival_date"] else "cancelled / not expected",
        "delay_days": impact["delay_days"],
        "on_hand_units": f"{impact['on_hand']:,}",
        "weekly_usage_units": f"{int(p['weekly_usage']):,}",
        "stockout_date_with_delay": fmt_date(impact["stockout_with_delay"]),
        "stockout_date_without_delay": fmt_date(impact["stockout_without_delay"]),
        "stockout_before_shipment_arrives": "yes" if impact["stockout_before_arrival"] else "no",
        "production_shortfall_units": f"{impact['production_shortfall']:,}",
        "orders_at_risk": len(impact["orders_at_risk"]),
        "high_priority_orders_at_risk": len(high),
        "order_units_at_risk": f"{impact['orders_at_risk_qty']:,}",
        "affected_orders": ", ".join(f"{o['id']} ({o['customer']}, {o['priority']}, {o['days_late']} days late)"
                                     for o in impact["orders_at_risk"]) or "none",
        "total_units_needed": f"{impact['shortage_qty']:,}",
        "best_alternative_supplier": best["supplier"] if best else "none available",
    }
    if best:
        facts.update({
            "best_alternative_quantity": f"{best['coverage_qty']:,}",
            "best_alternative_arrival": fmt_date(best["arrival_date"]),
            "best_alternative_arrives_before_stockout": "yes" if best["arrives_in_time"] else "no",
            "best_alternative_unit_price": f"${best['unit_price']:,.2f}",
            "best_alternative_total_cost": f"${best['total_cost']:,.2f}",
            "best_alternative_cost_premium": f"${best['cost_premium']:,.2f}",
        })
    return facts


def full_analysis(data: dict, shipment_id, today: date | None = None) -> dict | None:
    """Impact + ranked alternatives + facts, in one call. None if the shipment is unknown."""
    today = today or date.today()
    impact = impact_analysis(data, shipment_id, today)
    if impact is None:
        return None
    alternatives = rank_alternatives(data, impact["product"]["id"], impact["shortage_qty"],
                                     impact["stockout_with_delay"], today)
    impact["alternatives"] = alternatives
    impact["best_alternative"] = alternatives[0] if alternatives and alternatives[0]["coverage_qty"] > 0 else None
    impact["facts"] = build_facts(impact, alternatives)
    return impact


# ---------------------------------------------------------------- 5. action plan

def build_action_plan(data: dict, shipment_id, today: date | None = None, emails: dict | None = None) -> list[dict]:
    """Return the list of pending actions to propose for a disrupted shipment.

    `emails` may contain "supplier_email" and "customer_email" drafted by ai.py.
    Returns [] for unknown or healthy shipments. The caller saves them to the DB.
    """
    analysis = full_analysis(data, shipment_id, today)
    if analysis is None or not analysis["is_disrupted"]:
        return []
    emails = emails or {}
    s, product = analysis["shipment"], analysis["product"]
    supplier_name = analysis["supplier"]["name"] if analysis["supplier"] else "the supplier"
    actions = []

    best = analysis["best_alternative"]
    if best and analysis["shortage_qty"] > 0:
        actions.append({
            "type": "reorder_alternative",
            "description": (f"Order {best['coverage_qty']:,} {product['name']} from {best['supplier']} "
                            f"(arrives {fmt_date(best['arrival_date'])}"
                            f"{', before the stockout' if best['arrives_in_time'] else ', AFTER the stockout'})."),
            "estimated_cost": best["total_cost"],
            "payload": {
                "supplier_id": best["supplier_id"], "supplier": best["supplier"],
                "product_id": product["id"], "quantity": best["coverage_qty"],
                "unit_price": best["unit_price"], "arrival_date": best["arrival_date"].isoformat(),
                "cost_premium": best["cost_premium"],
            },
        })

    high = [o for o in analysis["orders_at_risk"] if o.get("priority") == "high"]
    if high:
        actions.append({
            "type": "reserve_inventory",
            "description": (f"Reserve {sum(int(o['quantity_needed']) for o in high):,} {product['name']} "
                            f"for high-priority orders: {', '.join(o['id'] for o in high)}."),
            "estimated_cost": 0.0,
            "payload": {"order_ids": [o["id"] for o in high]},
        })

    actions.append({
        "type": "request_commitment",
        "description": f"Ask {supplier_name} for a firm, updated delivery commitment for {s['id']}.",
        "estimated_cost": 0.0,
        "payload": {"supplier": supplier_name, "email": emails.get("supplier_email", "")},
    })

    if analysis["orders_at_risk"]:
        actions.append({
            "type": "notify_customers",
            "description": (f"Notify {len(analysis['orders_at_risk'])} affected customer(s): "
                            f"{', '.join(o['customer'] for o in analysis['orders_at_risk'])}."),
            "estimated_cost": 0.0,
            "payload": {"order_ids": [o["id"] for o in analysis["orders_at_risk"]],
                        "email": emails.get("customer_email", "")},
        })
    return actions


# ---------------------------------------------------------------- 6. simulation

SCENARIOS = {
    "supplier_unavailable": "Supplier unavailable for N days",
    "shipment_delay": "Shipment delayed by N days",
    "cost_increase": "Raw material cost +X%",
}


def simulate(data: dict, event: dict, today: date | None = None) -> dict:
    """Run a what-if scenario on a deep COPY of the data. The real DB is never touched.

    event examples:
      {"type": "supplier_unavailable", "supplier_id": 1, "days": 14}
      {"type": "shipment_delay", "shipment_id": "SH-1042", "days": 10}
      {"type": "cost_increase", "product_id": 1, "percent": 15}
    Returns {"label", "title", "error", "analyses": [full_analysis...], "cost_impact"}.
    Bad input gives a result with "error" set instead of raising.
    """
    today = today or date.today()
    sim = copy.deepcopy(data)
    result = {"label": SIMULATION_LABEL, "title": "", "error": None, "analyses": [], "cost_impact": None}
    kind = event.get("type")

    def as_int(name, low, high):
        try:
            value = int(float(event.get(name)))
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number")
        if not low <= value <= high:
            raise ValueError(f"'{name}' must be between {low} and {high}")
        return value

    try:
        if kind == "shipment_delay":
            days = as_int("days", 1, 365)
            s = get_shipment(sim, event.get("shipment_id"))
            if s is None:
                raise ValueError("Unknown shipment")
            if s["status"] in ("delivered", "cancelled"):
                raise ValueError(f"Shipment {s['id']} is already {s['status']}")
            base = to_date(s["current_eta"]) or today
            s["current_eta"] = (base + timedelta(days=days)).isoformat()
            s["status"] = "delayed"
            result["title"] = f"What if {s['id']} is delayed by another {days} days?"
            result["analyses"].append(full_analysis(sim, s["id"], today))

        elif kind == "supplier_unavailable":
            days = as_int("days", 1, 365)
            supplier = get_supplier(sim, event.get("supplier_id"))
            if supplier is None:
                raise ValueError("Unknown supplier")
            supplier["lead_time_days"] = int(supplier["lead_time_days"]) + days
            affected = [s for s in sim["shipments"] if str(s["supplier_id"]) == str(supplier["id"])
                        and s["status"] not in ("delivered", "cancelled")]
            for s in affected:
                base = to_date(s["current_eta"]) or today
                s["current_eta"] = (base + timedelta(days=days)).isoformat()
                s["status"] = "delayed"
            result["title"] = f"What if {supplier['name']} is unavailable for {days} days?"
            for s in sorted(affected, key=lambda s: s["current_eta"]):
                result["analyses"].append(full_analysis(sim, s["id"], today))
            if not affected:
                result["error"] = f"{supplier['name']} has no open shipments, so stock is not affected."

        elif kind == "cost_increase":
            percent = as_int("percent", -90, 1000)
            product = get_product(sim, event.get("product_id"))
            if product is None:
                raise ValueError("Unknown product")
            factor = 1 + percent / 100.0
            product["unit_cost"] = round(float(product["unit_cost"]) * factor, 4)
            for s in sim["suppliers"]:
                if str(s["product_id"]) == str(product["id"]):
                    s["unit_price"] = round(float(s["unit_price"]) * factor, 4)
            open_units = sum(int(s["quantity"]) for s in sim["shipments"]
                             if str(s["product_id"]) == str(product["id"]) and s["status"] not in ("delivered", "cancelled"))
            original_primary = primary_supplier(data, product["id"])
            if original_primary:
                base_price = float(original_primary["unit_price"])
            else:
                base_price = float(get_product(data, product["id"])["unit_cost"])
            weekly = int(product["weekly_usage"])
            result["title"] = f"What if {product['name']} cost rises by {percent}%?"
            result["cost_impact"] = {
                "product": product["name"],
                "percent": percent,
                "old_unit_price": round(base_price, 2),
                "new_unit_price": round(base_price * factor, 2),
                "open_shipment_units": open_units,
                "extra_cost_open_shipments": round(open_units * base_price * (factor - 1), 2),
                "extra_cost_per_week": round(weekly * base_price * (factor - 1), 2),
                "extra_cost_per_year": round(weekly * 52 * base_price * (factor - 1), 2),
            }
            # Show the stock picture for the product's next open shipment, with re-priced alternatives.
            open_ships = sorted([s for s in sim["shipments"] if str(s["product_id"]) == str(product["id"])
                                 and s["status"] not in ("delivered", "cancelled")], key=lambda s: s["current_eta"])
            if open_ships:
                result["analyses"].append(full_analysis(sim, open_ships[0]["id"], today))
        else:
            raise ValueError("Please choose a scenario")
    except ValueError as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------- dashboard helper

def shipment_overview(data: dict, today: date | None = None) -> list[dict]:
    """Every shipment with delay_days and an at_risk flag, for the dashboard and the JSON API."""
    today = today or date.today()
    out = []
    for s in data.get("shipments", []):
        item = dict(s)
        item["delay_days"] = delay_days(s)
        item["disrupted"] = is_disrupted(s)
        item["at_risk"] = False
        if item["disrupted"]:
            impact = impact_analysis(data, s["id"], today)
            item["at_risk"] = bool(impact and impact["at_risk"])
        supplier = get_supplier(data, s["supplier_id"])
        product = get_product(data, s["product_id"])
        item["supplier_name"] = supplier["name"] if supplier else "Unknown"
        item["product_name"] = product["name"] if product else "Unknown"
        out.append(item)
    out.sort(key=lambda i: (not i["at_risk"], not i["disrupted"], i["status"] == "delivered", i["current_eta"]))
    return out
