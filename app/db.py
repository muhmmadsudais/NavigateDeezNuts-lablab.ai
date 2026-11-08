"""Database layer: schema, seed data and small helpers.

We use the standard-library `sqlite3` module (no ORM). Every query uses
`?` placeholders (parameterized SQL) so user input can never inject SQL.

All seed dates are relative to *today* so the demo never goes stale.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    unit_cost    REAL NOT NULL,
    weekly_usage INTEGER NOT NULL          -- units consumed per week by production
);

CREATE TABLE IF NOT EXISTS inventory (
    product_id INTEGER PRIMARY KEY REFERENCES products(id),
    on_hand    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS suppliers (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    product_id     INTEGER NOT NULL REFERENCES products(id),
    unit_price     REAL NOT NULL,
    lead_time_days INTEGER NOT NULL,
    available_qty  INTEGER NOT NULL,
    reliability    REAL NOT NULL,          -- 0..1
    is_primary     INTEGER NOT NULL        -- 1 = primary supplier, 0 = alternative
);

CREATE TABLE IF NOT EXISTS shipments (
    id            TEXT PRIMARY KEY,        -- e.g. SH-1042
    supplier_id   INTEGER NOT NULL REFERENCES suppliers(id),
    product_id    INTEGER NOT NULL REFERENCES products(id),
    quantity      INTEGER NOT NULL,
    expected_date TEXT NOT NULL,           -- ISO date the supplier originally promised
    current_eta   TEXT NOT NULL,           -- ISO date we currently expect
    status        TEXT NOT NULL,           -- on_time / delayed / cancelled / delivered
    notes         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS customer_orders (
    id              TEXT PRIMARY KEY,      -- e.g. ORD-2001
    customer        TEXT NOT NULL,
    product_id      INTEGER NOT NULL REFERENCES products(id),
    quantity_needed INTEGER NOT NULL,
    due_date        TEXT NOT NULL,
    priority        TEXT NOT NULL,         -- high / medium / low
    status          TEXT NOT NULL          -- open / inventory reserved / customer notified / delivered
);

CREATE TABLE IF NOT EXISTS actions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id    TEXT NOT NULL,
    type           TEXT NOT NULL,          -- reorder_alternative / reserve_inventory / request_commitment / notify_customers
    description    TEXT NOT NULL,
    estimated_cost REAL NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'pending',   -- pending / approved / rejected
    created_at     TEXT NOT NULL,
    decided_at     TEXT,
    payload        TEXT NOT NULL DEFAULT '{}'         -- JSON (drafted emails, quantities, ...)
);

CREATE TABLE IF NOT EXISTS activity_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    step      TEXT NOT NULL,
    message   TEXT NOT NULL
);
"""

TABLES = ["products", "inventory", "suppliers", "shipments", "customer_orders", "actions", "activity_log"]


def db_path() -> Path:
    """Location of the SQLite file. Override with the NDN_DB_PATH env var (used by tests)."""
    custom = os.getenv("NDN_DB_PATH")
    if custom:
        return Path(custom)
    return PROJECT_ROOT / "data" / "navigatedeeznuts.db"


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["name"]
    return conn


@contextmanager
def get_conn():
    """Open a connection, commit on success, always close."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    """Create tables if missing and seed demo data if the DB is empty. Safe to call many times."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if count == 0:
            seed(conn)


def reset_db(today: date | None = None) -> None:
    """Wipe every table and re-seed the demo data."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        for table in TABLES:
            conn.execute(f"DELETE FROM {table}")  # table names come from our constant list, not user input
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('actions', 'activity_log')")
        seed(conn, today)


def seed(conn: sqlite3.Connection, today: date | None = None) -> None:
    """Insert the demo data set.

    Headline scenario (microchips):
      * 1,500 on hand, 1,000 used per week (~143/day) -> runs out in ~10.5 days.
      * SH-1042 (TechParts Ltd., 5,000 units) is due in 7 days -> healthy.
      * If SH-1042 slips 10 days (to day 17) stock runs out on day 11, and
        two microchip orders due on days 12 and 15 (one high priority) are at risk.
      * ChipSource Asia can deliver 2,000 units in 4 days at a higher price.
    """
    today = today or date.today()

    def d(days: int) -> str:
        return (today + timedelta(days=days)).isoformat()

    products = [
        # id, name, unit_cost, weekly_usage
        (1, "Microchips", 2.50, 1000),
        (2, "Batteries", 4.20, 800),
        (3, "Screens", 18.00, 500),
        (4, "Casings", 1.20, 900),
        (5, "Sensors", 6.80, 600),
    ]
    inventory = [(1, 1500), (2, 3000), (3, 1200), (4, 4000), (5, 2500)]
    suppliers = [
        # id, name, product_id, unit_price, lead_time_days, available_qty, reliability, is_primary
        (1, "TechParts Ltd.", 1, 2.50, 21, 20000, 0.95, 1),
        (2, "ChipSource Asia", 1, 3.10, 4, 2000, 0.92, 0),
        (3, "RapidChip EU", 1, 3.60, 3, 3000, 0.88, 0),
        (4, "PowerCell Inc.", 2, 4.20, 14, 15000, 0.94, 1),
        (5, "VoltEdge Supply", 2, 5.10, 5, 1500, 0.90, 0),
        (6, "DisplayWorks", 3, 18.00, 18, 6000, 0.93, 1),
        (7, "ClearView Panels", 3, 21.50, 6, 800, 0.89, 0),
        (8, "MoldPro Plastics", 4, 1.20, 10, 30000, 0.97, 1),
        (9, "SenseTech", 5, 6.80, 12, 10000, 0.96, 1),
    ]
    shipments = [
        # id, supplier_id, product_id, quantity, expected_date, current_eta, status, notes
        ("SH-1042", 1, 1, 5000, d(7), d(7), "on_time", "Weekly microchip replenishment"),
        ("SH-1043", 4, 2, 4000, d(10), d(10), "on_time", ""),
        ("SH-1044", 6, 3, 2000, d(12), d(14), "delayed", "Port congestion, minor delay"),
        ("SH-1045", 8, 4, 3000, d(-3), d(-3), "delivered", "Received in full"),
        ("SH-1046", 9, 5, 3000, d(9), d(9), "on_time", ""),
        ("SH-1047", 1, 1, 4000, d(35), d(35), "on_time", ""),
        ("SH-1048", 4, 2, 3000, d(30), d(30), "on_time", ""),
        ("SH-1049", 6, 3, 2000, d(40), d(40), "on_time", ""),
        ("SH-1050", 8, 4, 5000, d(20), d(20), "on_time", ""),
        ("SH-1051", 9, 5, 2000, d(45), d(45), "on_time", ""),
    ]
    orders = [
        # id, customer, product_id, quantity_needed, due_date, priority, status
        ("ORD-2001", "Acme Robotics", 1, 600, d(12), "high", "open"),
        ("ORD-2002", "Nordic Devices", 1, 400, d(15), "medium", "open"),
        ("ORD-2003", "BrightHome Appliances", 1, 300, d(25), "low", "open"),
        ("ORD-2004", "Voltway Mobility", 2, 500, d(20), "medium", "open"),
        ("ORD-2005", "Lumina Medical", 3, 300, d(18), "high", "open"),
        ("ORD-2006", "UrbanGear", 4, 1000, d(14), "low", "open"),
        ("ORD-2007", "AgriSense Co.", 5, 400, d(21), "medium", "open"),
        ("ORD-2008", "Pulse Wearables", 2, 200, d(8), "high", "open"),
    ]

    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)
    conn.executemany("INSERT INTO inventory VALUES (?, ?)", inventory)
    conn.executemany("INSERT INTO suppliers VALUES (?, ?, ?, ?, ?, ?, ?, ?)", suppliers)
    conn.executemany("INSERT INTO shipments VALUES (?, ?, ?, ?, ?, ?, ?, ?)", shipments)
    conn.executemany("INSERT INTO customer_orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders)
    log(conn, "System", "Demo data loaded: 5 components, 9 suppliers, 10 shipments, 8 customer orders.")
    log(conn, "Monitoring", "Agent is monitoring all inbound shipments and supplier messages.")


# ---------------------------------------------------------------- helpers

def log(conn: sqlite3.Connection, step: str, message: str) -> None:
    """Append one line to the agent activity feed."""
    conn.execute(
        "INSERT INTO activity_log (timestamp, step, message) VALUES (?, ?, ?)",
        (now_str(), step, message),
    )


def rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def row(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
    r = conn.execute(sql, params).fetchone()
    return dict(r) if r else None


def load_snapshot(conn: sqlite3.Connection) -> dict:
    """Read every business table into plain Python lists of dicts.

    The logic module works on this snapshot, which makes it easy to test and
    lets the simulator work on a deep copy without touching the real DB.
    """
    return {
        "products": rows(conn, "SELECT * FROM products ORDER BY id"),
        "inventory": rows(conn, "SELECT * FROM inventory ORDER BY product_id"),
        "suppliers": rows(conn, "SELECT * FROM suppliers ORDER BY id"),
        "shipments": rows(conn, "SELECT * FROM shipments ORDER BY id"),
        "customer_orders": rows(conn, "SELECT * FROM customer_orders ORDER BY id"),
    }


def save_actions(conn: sqlite3.Connection, shipment_id: str, actions: list[dict]) -> int:
    """Replace any *pending* actions for this shipment with a fresh plan."""
    conn.execute("DELETE FROM actions WHERE shipment_id = ? AND status = 'pending'", (shipment_id,))
    for a in actions:
        conn.execute(
            "INSERT INTO actions (shipment_id, type, description, estimated_cost, status, created_at, payload) "
            "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
            (shipment_id, a["type"], a["description"], float(a.get("estimated_cost", 0)), now_str(),
             json.dumps(a.get("payload", {}))),
        )
    return len(actions)


def next_shipment_id(conn: sqlite3.Connection) -> str:
    """Return the next free id like SH-1052."""
    highest = 1000
    for r in conn.execute("SELECT id FROM shipments").fetchall():
        try:
            highest = max(highest, int(str(r["id"]).split("-")[-1]))
        except ValueError:
            continue
    return f"SH-{highest + 1}"
