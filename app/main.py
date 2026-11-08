"""NavigateDeezNuts - web routes.

Run locally with:  uvicorn app.main:app --reload
All calculations live in logic.py, all AI calls in ai.py, all SQL in db.py
(plus a few simple UPDATE/INSERT statements below for approving actions).
"""

import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import ai, db, logic

APP_NAME = "NavigateDeezNuts"
APP_TAGLINE = "AI supply chain disruption agent"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("navigatedeeznuts")

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()  # create tables + seed demo data on first start
    status = ai.status()
    logger.info("%s started. AI mode: %s", APP_NAME,
                "Vultr Serverless Inference (API key set)" if status["configured"]
                else "offline fallback (no VULTR_API_KEY)")
    yield


app = FastAPI(title=APP_NAME, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------- template filters

def money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def num(value) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "-"


def nice_date(value) -> str:
    d = logic.to_date(value)
    return d.strftime("%a %d %b %Y") if d else "-"


def short_date(value) -> str:
    d = logic.to_date(value)
    return d.strftime("%a %d %b") if d else "-"


def label(value) -> str:
    return str(value or "").replace("_", " ").capitalize()


templates.env.filters.update(money=money, num=num, nice_date=nice_date, short_date=short_date, label=label)


def render(request: Request, name: str, context: dict | None = None, status_code: int = 200):
    """TemplateResponse with the values every page needs (Starlette 0.41 signature)."""
    ctx = {
        "app_name": APP_NAME,
        "tagline": APP_TAGLINE,
        "ai_status": ai.status(),
        "today": date.today(),
        "notice": request.query_params.get("notice", ""),
        "error": request.query_params.get("error", ""),
        "simulation_label": logic.SIMULATION_LABEL,
    }
    ctx.update(context or {})
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def redirect(url: str, **params) -> RedirectResponse:
    """POST/redirect/GET helper; params become ?notice=... etc."""
    query = urlencode({k: v for k, v in params.items() if v})
    return RedirectResponse(f"{url}?{query}" if query else url, status_code=303)


# ---------------------------------------------------------------- error pages

@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    title = "Page not found" if exc.status_code == 404 else "Something went wrong"
    return render(request, "error.html", {"title": title, "message": exc.detail, "code": exc.status_code},
                  status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Invalid request"}, status_code=400)
    return render(request, "error.html", {"title": "Invalid request",
                                          "message": "Some of the submitted values were missing or invalid.",
                                          "code": 400}, status_code=400)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return render(request, "error.html", {"title": "Unexpected error",
                                          "message": "The agent hit an unexpected problem. It has been logged; "
                                                     "please go back and try again.", "code": 500},
                  status_code=500)


# ---------------------------------------------------------------- shared helpers

def snapshot() -> dict:
    with db.get_conn() as conn:
        return db.load_snapshot(conn)


def chart_for(analysis: dict) -> dict:
    """Turn a stock projection into SVG bar geometry (server-side, no chart library)."""
    points = analysis["projection"]
    width, height, top, bottom, left = 720, 220, 12, 30, 52
    plot_h = height - top - bottom
    if not points:
        return {"bars": [], "width": width, "height": height}
    values = [p["stock"] for p in points]
    vmax, vmin = max(max(values), 1.0), min(min(values), 0.0)
    span = vmax - vmin
    zero_y = top + plot_h * vmax / span
    bar_w = (width - left - 4) / len(points)
    weekly = float(analysis["product"]["weekly_usage"] or 0)
    stockout, arrival = analysis["stockout_with_delay"], analysis["arrival_date"]
    bars, ticks, markers = [], [], []
    for i, p in enumerate(points):
        h = abs(p["stock"]) / span * plot_h
        y = zero_y - h if p["stock"] >= 0 else zero_y
        color = "red" if p["stock"] < 0 else "amber" if p["stock"] < weekly else "green"
        x = left + i * bar_w
        bars.append({"x": round(x, 1), "y": round(y, 1), "w": round(max(bar_w - 1.5, 1), 1),
                     "h": round(max(h, 0.5), 1), "color": color,
                     "title": f"{p['date'].strftime('%d %b')}: {int(round(p['stock'])):,} units"})
        if i % 7 == 0:
            ticks.append({"x": round(x + bar_w / 2, 1), "text": p["date"].strftime("%d %b")})
        if arrival and p["date"] == arrival:
            markers.append({"x": round(x + bar_w / 2, 1), "text": "Shipment ETA", "cls": "marker-eta"})
        if stockout and p["date"] == stockout:
            markers.append({"x": round(x + bar_w / 2, 1), "text": "Stockout", "cls": "marker-stockout"})
    return {"bars": bars, "ticks": ticks, "markers": markers, "width": width, "height": height,
            "zero_y": round(zero_y, 1), "left": left, "top": top, "plot_bottom": top + plot_h,
            "vmax": int(round(vmax)), "vmin": int(round(vmin))}


def step_trace(analysis: dict, actions: list[dict]) -> list[dict]:
    """Detected -> Analyzed -> Sought alternatives -> Recommended -> Awaiting approval -> Decided."""
    disrupted = analysis["is_disrupted"]
    pending = any(a["status"] == "pending" for a in actions)
    decided = bool(actions) and not pending
    steps = [
        ("Detected", disrupted, f"{analysis['delay_days']} day delay" if disrupted else "no disruption"),
        ("Analyzed", disrupted, f"stockout {nice_date(analysis['stockout_with_delay'])}"
         if analysis["stockout_with_delay"] else "no stockout"),
        ("Sought alternatives", disrupted, f"{len(analysis['alternatives'])} supplier(s) compared"),
        ("Recommended", disrupted, "see recommendation below"),
        ("Awaiting approval", pending or decided, "actions pending" if pending else
         "actions decided" if decided else "generate a plan"),
    ]
    trace = []
    for i, (name, done, detail) in enumerate(steps):
        state = "done" if done else "todo"
        if name == "Awaiting approval" and pending:
            state = "current"
        trace.append({"name": name, "state": state, "detail": detail if done else ""})
    return trace


def analysis_view(analysis: dict, simulation: bool = False) -> dict:
    text, source = ai.recommend(analysis["facts"], simulation=simulation)
    return {"a": analysis, "chart": chart_for(analysis), "recommendation": text, "ai_source": source}


def run_agent_pipeline(conn, shipment_id: str) -> None:
    """Log the agent's reasoning steps for one shipment to the activity feed."""
    data = db.load_snapshot(conn)
    a = logic.full_analysis(data, shipment_id)
    if a is None or not a["is_disrupted"]:
        return
    supplier = a["supplier"]["name"] if a["supplier"] else "supplier"
    s = a["shipment"]
    db.log(conn, "Detected", f"{s['id']} from {supplier} is {s['status']} "
                             f"({a['delay_days']} days late, ETA {nice_date(a['arrival_date'])}).")
    if a["stockout_before_arrival"]:
        db.log(conn, "Analyzed", f"{a['product']['name']} stock runs out {nice_date(a['stockout_with_delay'])}, "
                                 f"before the shipment arrives. {len(a['orders_at_risk'])} customer order(s) at risk; "
                                 f"{num(a['shortage_qty'])} units needed.")
    else:
        db.log(conn, "Analyzed", f"Current {a['product']['name']} stock covers the delay. No orders at risk.")
    best = a["best_alternative"]
    db.log(conn, "Sought alternatives", f"Compared {len(a['alternatives'])} alternative supplier(s)"
           + (f"; best is {best['supplier']} ({num(best['coverage_qty'])} units, arrives "
              f"{nice_date(best['arrival_date'])}, {money(best['total_cost'])})." if best and a['shortage_qty'] else "."))
    db.log(conn, "Recommended", f"Recommendation ready for {s['id']}. Open the impact page to review and "
                                "generate an action plan.")


# ---------------------------------------------------------------- pages: dashboard

def dashboard_context(upload_errors: list[str] | None = None) -> dict:
    with db.get_conn() as conn:
        data = db.load_snapshot(conn)
        activity = db.rows(conn, "SELECT * FROM activity_log ORDER BY id DESC LIMIT 8")
        pending = conn.execute("SELECT COUNT(*) FROM actions WHERE status = 'pending'").fetchone()[0]
    overview = logic.shipment_overview(data)
    active = [s for s in overview if s["status"] not in ("delivered", "cancelled")]
    stock = []
    for p in data["products"]:
        on_hand = logic.on_hand(data, p["id"])
        weekly = int(p["weekly_usage"]) or 1
        stock.append({"name": p["name"], "on_hand": on_hand, "weekly_usage": p["weekly_usage"],
                      "weeks": on_hand / weekly,
                      "stockout": logic.project_stock(data, p["id"])["stockout_date"]})
    return {
        "kpis": {
            "active": len(active),
            "at_risk": sum(1 for s in overview if s["at_risk"]),
            "delayed": sum(1 for s in overview if s["disrupted"]),
            "suppliers": len(data["suppliers"]),
            "pending": pending,
        },
        "shipments": overview,
        "stock": stock,
        "activity": activity,
        "upload_errors": upload_errors or [],
    }


@app.get("/")
def dashboard(request: Request):
    return render(request, "dashboard.html", dashboard_context())


@app.post("/scan")
def scan():
    """Run the detection step across all shipments and log what the agent finds."""
    with db.get_conn() as conn:
        found = logic.detect_disruptions(db.load_snapshot(conn))
        db.log(conn, "Monitoring", f"Agent scan complete: {len(found)} disrupted shipment(s) found.")
        for s in found:
            run_agent_pipeline(conn, s["id"])
    return redirect("/", notice=f"Agent scan complete: {len(found)} disrupted shipment(s).")


@app.post("/reset")
def reset():
    db.reset_db()
    return redirect("/", notice="Demo data has been reset.")


# ---------------------------------------------------------------- CSV upload (optional feature)

SHIPMENT_COLUMNS = ["id", "supplier_id", "product_id", "quantity", "expected_date", "current_eta", "status"]
INVENTORY_COLUMNS = ["product_id", "on_hand"]
VALID_STATUSES = {"on_time", "delayed", "cancelled", "delivered"}


def validate_csv(kind: str, text: str, data: dict) -> tuple[list[tuple], list[str]]:
    """Return (rows to write, error messages). Nothing is written if there is any error."""
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    required = SHIPMENT_COLUMNS if kind == "shipments" else INVENTORY_COLUMNS
    missing = [c for c in required if c not in header]
    if missing:
        return [], [f"Missing column(s): {', '.join(missing)}. Expected header: {','.join(required)}"]
    supplier_ids = {str(s["id"]) for s in data["suppliers"]}
    product_ids = {str(p["id"]) for p in data["products"]}
    out, errors = [], []
    for line_no, raw in enumerate(reader, start=2):
        r = {str(k).strip(): (v or "").strip() for k, v in raw.items() if k is not None}
        if not any(r.values()):
            continue
        problems = []
        if r.get("product_id") not in product_ids:
            problems.append(f"unknown product_id '{r.get('product_id')}'")
        if kind == "shipments":
            if not r["id"]:
                problems.append("id is empty")
            if r.get("supplier_id") not in supplier_ids:
                problems.append(f"unknown supplier_id '{r.get('supplier_id')}'")
            if not r["quantity"].isdigit() or int(r["quantity"]) <= 0:
                problems.append("quantity must be a positive whole number")
            for col in ("expected_date", "current_eta"):
                if logic.to_date(r[col]) is None or len(r[col]) != 10:
                    problems.append(f"{col} must be YYYY-MM-DD")
            if r["status"] not in VALID_STATUSES:
                problems.append(f"status must be one of {', '.join(sorted(VALID_STATUSES))}")
            if not problems:
                out.append((r["id"].upper(), int(r["supplier_id"]), int(r["product_id"]), int(r["quantity"]),
                            r["expected_date"], r["current_eta"], r["status"], r.get("notes", "")[:300]))
        else:
            if not r.get("on_hand", "").isdigit():
                problems.append("on_hand must be a whole number >= 0")
            if not problems:
                out.append((int(r["product_id"]), int(r["on_hand"])))
        if problems:
            errors.append(f"Row {line_no}: " + "; ".join(problems))
    if not out and not errors:
        errors.append("The file has no data rows.")
    return out, errors[:20]


@app.post("/upload")
async def upload(request: Request, kind: str = Form(""), file: UploadFile | None = File(None)):
    errors: list[str] = []
    if kind not in ("shipments", "inventory"):
        errors.append("Choose what the file contains (shipments or inventory).")
    content = await file.read(1_000_001) if file is not None else b""
    if not content:
        errors.append("Please choose a non-empty CSV file.")
    elif len(content) > 1_000_000:
        errors.append("File is too large (max 1 MB).")
    if not errors:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            errors.append("File must be UTF-8 encoded CSV text.")
        else:
            rows_to_write, errors = validate_csv(kind, text, snapshot())
    if errors:
        return render(request, "dashboard.html", dashboard_context(errors), status_code=400)
    with db.get_conn() as conn:
        if kind == "shipments":
            conn.executemany("INSERT OR REPLACE INTO shipments VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows_to_write)
        else:
            conn.executemany("INSERT OR REPLACE INTO inventory VALUES (?, ?)", rows_to_write)
        db.log(conn, "Ingested", f"Imported {len(rows_to_write)} {kind} row(s) from CSV.")
    return redirect("/", notice=f"Imported {len(rows_to_write)} {kind} row(s).")


# ---------------------------------------------------------------- pages: shipment impact

@app.get("/shipments/{shipment_id}")
def shipment_page(request: Request, shipment_id: str):
    with db.get_conn() as conn:
        data = db.load_snapshot(conn)
        actions = db.rows(conn, "SELECT * FROM actions WHERE shipment_id = ? ORDER BY id", (shipment_id,))
    analysis = logic.full_analysis(data, shipment_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"There is no shipment with id '{shipment_id}'.")
    return render(request, "shipment.html", {
        "view": analysis_view(analysis),
        "trace": step_trace(analysis, actions),
        "actions": actions,
        "has_pending": any(a["status"] == "pending" for a in actions),
    })


@app.post("/shipments/{shipment_id}/plan")
def generate_plan(shipment_id: str):
    data = snapshot()
    analysis = logic.full_analysis(data, shipment_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"There is no shipment with id '{shipment_id}'.")
    if not analysis["is_disrupted"]:
        return redirect(f"/shipments/{shipment_id}", notice="This shipment is on track; no action plan is needed.")
    emails, source = ai.draft_emails(analysis["facts"])
    plan = logic.build_action_plan(data, shipment_id, emails=emails)
    for action in plan:
        action["payload"]["ai_source"] = source
    with db.get_conn() as conn:
        count = db.save_actions(conn, shipment_id, plan)
        db.log(conn, "Awaiting approval", f"Action plan for {shipment_id}: {count} action(s) waiting for approval "
                                          f"(est. {money(sum(a['estimated_cost'] for a in plan))}).")
    return redirect("/actions", notice=f"Action plan created for {shipment_id}: {count} action(s) awaiting approval.")


# ---------------------------------------------------------------- pages: simulator

def simulator_options(data: dict) -> dict:
    return {
        "scenarios": logic.SCENARIOS,
        "suppliers": data["suppliers"],
        "products": data["products"],
        "open_shipments": [s for s in data["shipments"] if s["status"] not in ("delivered", "cancelled")],
    }


@app.get("/simulator")
def simulator_page(request: Request):
    data = snapshot()
    form = {"scenario": "shipment_delay", "shipment_id": "SH-1042", "supplier_id": "1", "product_id": "1",
            "days": "10", "percent": "15"}
    return render(request, "simulator.html", {**simulator_options(data), "form": form, "result": None})


@app.post("/simulator")
def simulator_run(request: Request, scenario: str = Form(""), shipment_id: str = Form(""),
                  supplier_id: str = Form(""), product_id: str = Form(""), days: str = Form(""),
                  percent: str = Form("")):
    data = snapshot()
    form = {"scenario": scenario, "shipment_id": shipment_id, "supplier_id": supplier_id,
            "product_id": product_id, "days": days, "percent": percent}
    event = {"type": scenario, "shipment_id": shipment_id, "supplier_id": supplier_id,
             "product_id": product_id, "days": days, "percent": percent}
    result = logic.simulate(data, event)
    views = [analysis_view(a, simulation=True) for a in result["analyses"]]
    if not result["error"]:
        with db.get_conn() as conn:
            db.log(conn, "Simulation", f"{result['title']} (hypothetical, real data unchanged)")
    return render(request, "simulator.html", {**simulator_options(data), "form": form,
                                              "result": result, "views": views})


# ---------------------------------------------------------------- pages: action center

def apply_action(conn, action: dict) -> str:
    """Make an approved action change the world (DB state). Returns a log message."""
    payload = json.loads(action["payload"] or "{}")
    kind = action["type"]
    if kind == "reorder_alternative":
        supplier = db.row(conn, "SELECT * FROM suppliers WHERE id = ?", (payload["supplier_id"],))
        if supplier is None:
            raise ValueError("The alternative supplier no longer exists.")
        qty = int(payload["quantity"])
        arrival = (date.today() + timedelta(days=int(supplier["lead_time_days"]))).isoformat()
        new_id = db.next_shipment_id(conn)
        conn.execute("INSERT INTO shipments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (new_id, supplier["id"], supplier["product_id"], qty, arrival, arrival, "on_time",
                      f"Emergency reorder for {action['shipment_id']}"))
        conn.execute("UPDATE suppliers SET available_qty = MAX(0, available_qty - ?) WHERE id = ?",
                     (qty, supplier["id"]))
        return f"Purchase order placed: new shipment {new_id} ({qty:,} units from {supplier['name']}, ETA {nice_date(arrival)})."
    if kind == "reserve_inventory":
        for order_id in payload.get("order_ids", []):
            conn.execute("UPDATE customer_orders SET status = CASE WHEN status IN ('customer notified', "
                         "'reserved + notified') THEN 'reserved + notified' ELSE 'inventory reserved' END "
                         "WHERE id = ?", (order_id,))
        return f"Inventory reserved for {', '.join(payload.get('order_ids', []))}."
    if kind == "notify_customers":
        for order_id in payload.get("order_ids", []):
            conn.execute("UPDATE customer_orders SET status = CASE WHEN status IN ('inventory reserved', "
                         "'reserved + notified') THEN 'reserved + notified' ELSE 'customer notified' END "
                         "WHERE id = ?", (order_id,))
        return f"Delay notification sent (simulated) for {', '.join(payload.get('order_ids', []))}."
    if kind == "request_commitment":
        conn.execute("UPDATE shipments SET notes = TRIM(notes || ' · Commitment requested ' || ?, ' ·') WHERE id = ?",
                     (date.today().isoformat(), action["shipment_id"]))
        return f"Commitment request sent (simulated) to {payload.get('supplier', 'the supplier')}."
    return "Action approved."


@app.get("/actions")
def actions_page(request: Request):
    with db.get_conn() as conn:
        # pending: oldest first (plan order); decided: newest first
        actions = db.rows(conn, "SELECT * FROM actions ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, "
                                "CASE status WHEN 'pending' THEN id ELSE -id END")
        orders = db.rows(conn, "SELECT o.*, p.name AS product_name FROM customer_orders o "
                               "JOIN products p ON p.id = o.product_id ORDER BY o.due_date")
    for a in actions:
        try:
            a["payload_data"] = json.loads(a["payload"] or "{}")
        except json.JSONDecodeError:
            a["payload_data"] = {}
    groups = {s: [a for a in actions if a["status"] == s] for s in ("pending", "approved", "rejected")}
    total_pending = sum(a["estimated_cost"] for a in groups["pending"])
    return render(request, "actions.html", {"groups": groups, "orders": orders, "total_pending": total_pending})


@app.post("/actions/{action_id}/{decision}")
def decide_action(action_id: int, decision: str):
    if decision not in ("approve", "reject"):
        raise HTTPException(status_code=404, detail="Unknown decision.")
    with db.get_conn() as conn:
        action = db.row(conn, "SELECT * FROM actions WHERE id = ?", (action_id,))
        if action is None:
            raise HTTPException(status_code=404, detail=f"There is no action #{action_id}.")
        if action["status"] != "pending":
            return redirect("/actions", error=f"Action #{action_id} was already {action['status']}.")
        if decision == "approve":
            try:
                message = apply_action(conn, action)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Could not apply action %s: %s", action_id, exc)
                return redirect("/actions", error=f"Action #{action_id} could not be applied.")
            new_status, step = "approved", "Approved"
        else:
            message, new_status, step = f"Rejected: {action['description']}", "rejected", "Rejected"
        conn.execute("UPDATE actions SET status = ?, decided_at = ? WHERE id = ?", (new_status, db.now_str(), action_id))
        db.log(conn, step, message)
    return redirect("/actions", notice=message)


# ---------------------------------------------------------------- pages: ingest supplier email

@app.get("/ingest")
def ingest_page(request: Request, sample: int = 0):
    text = ai.sample_email(snapshot()) if sample else ""
    return render(request, "ingest.html", {"email_text": text, "parsed": None})


@app.post("/ingest")
def ingest_parse(request: Request, email_text: str = Form("")):
    data = snapshot()
    if not email_text.strip():
        return render(request, "ingest.html", {"email_text": "", "parsed": None,
                                               "form_error": "Please paste a supplier email first."}, status_code=400)
    fields, source = ai.parse_email(email_text[:10000], data)
    shipment = logic.match_shipment(data, fields)
    new_date = fields.get("new_delivery_date") or ""
    if not new_date and shipment and fields.get("delay_days"):
        base = logic.to_date(shipment["current_eta"]) or date.today()
        new_date = (base + timedelta(days=int(fields["delay_days"]))).isoformat()
    suggested = "cancelled" if fields.get("cancelled") else "delayed"
    if shipment and new_date and logic.to_date(new_date) and logic.to_date(new_date) <= logic.to_date(shipment["expected_date"]):
        suggested = "on_time"
    return render(request, "ingest.html", {
        "email_text": email_text, "parsed": fields, "ai_source": source, "shipment": shipment,
        "new_date": new_date, "suggested_status": suggested,
        "open_shipments": [s for s in data["shipments"] if s["status"] not in ("delivered",)],
    })


@app.post("/ingest/confirm")
def ingest_confirm(shipment_id: str = Form(""), new_delivery_date: str = Form(""),
                   status: str = Form("delayed"), reason: str = Form("")):
    new_date = logic.to_date(new_delivery_date)
    if status not in ("delayed", "cancelled", "on_time"):
        status = "delayed"
    with db.get_conn() as conn:
        shipment = db.row(conn, "SELECT * FROM shipments WHERE id = ?", (shipment_id,))
        if shipment is None:
            return redirect("/ingest", error="Please choose which shipment this email is about.")
        if new_date is None and status != "cancelled":
            return redirect("/ingest", error="Please enter a valid new delivery date (YYYY-MM-DD).")
        eta = new_date.isoformat() if new_date else shipment["current_eta"]
        if status != "cancelled":
            status = "delayed" if eta > shipment["expected_date"] else "on_time"
        note = f"Supplier email: {reason.strip()[:200]}" if reason.strip() else "Updated from supplier email"
        conn.execute("UPDATE shipments SET current_eta = ?, status = ?, notes = ? WHERE id = ?",
                     (eta, status, note, shipment_id))
        db.log(conn, "Ingested", f"Supplier email processed: {shipment_id} is now {status.replace('_', ' ')}, "
                                 f"ETA {nice_date(eta)}.")
        run_agent_pipeline(conn, shipment_id)
    return redirect(f"/shipments/{shipment_id}", notice=f"Shipment {shipment_id} updated from supplier email.")


# ---------------------------------------------------------------- pages: activity

@app.get("/activity")
def activity_page(request: Request):
    with db.get_conn() as conn:
        entries = db.rows(conn, "SELECT * FROM activity_log ORDER BY id DESC LIMIT 500")
    return render(request, "activity.html", {"activity": entries})


@app.get("/partials/activity")
def activity_partial(request: Request, limit: int = 8):
    limit = max(1, min(limit, 500))
    with db.get_conn() as conn:
        entries = db.rows(conn, "SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (limit,))
    return render(request, "_activity_feed.html", {"activity": entries})


# ---------------------------------------------------------------- API

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/shipments")
def api_shipments():
    out = []
    for s in logic.shipment_overview(snapshot()):
        out.append({k: s[k] for k in ("id", "supplier_name", "product_name", "quantity", "expected_date",
                                      "current_eta", "status", "delay_days", "disrupted", "at_risk", "notes")})
    return {"shipments": out, "count": len(out)}
