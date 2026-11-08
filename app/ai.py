"""AI layer: Vultr Serverless Inference (OpenAI-compatible) + offline fallbacks.

Rules this module follows:
  * The LLM never does arithmetic or invents data. It only receives a JSON of
    facts computed by logic.py and is told to use nothing else.
  * Every function returns (result, source) where source is "vultr" if the LLM
    answered, or "fallback" if a deterministic template was used instead.
  * Any failure (no API key, network error, timeout, garbage output) falls back
    silently, so the app is fully demoable with no API key at all.
  * The API key is never logged or displayed.
"""

import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta

logger = logging.getLogger("navigatedeeznuts.ai")

DEFAULT_BASE_URL = "https://api.vultrinference.com/v1"
COOLDOWN_SECONDS = 60  # after a failed call, skip the LLM for a minute so pages stay fast

_lock = threading.Lock()
_client = None
_model: str | None = None
_effort: str | None = None  # reasoning_effort to request, if the model supports it
_cooldown_until = 0.0
_cache: dict[str, str] = {}

SOURCE_LLM = "vultr"
SOURCE_FALLBACK = "fallback"


# ---------------------------------------------------------------- client + model selection

def _api_key() -> str:
    if os.getenv("NDN_DISABLE_LLM") == "1":
        return ""
    return os.getenv("VULTR_API_KEY", "").strip()


def _get_client():
    """Create the OpenAI client once. Returns None when no API key is configured."""
    global _client
    if not _api_key():
        return None
    if _client is None:
        from openai import OpenAI  # imported lazily so the app starts even if something is off

        _client = OpenAI(
            api_key=_api_key(),
            base_url=os.getenv("VULTR_BASE_URL", "").strip() or DEFAULT_BASE_URL,
            timeout=30.0,
            max_retries=2,
        )
    return _client


NON_CHAT_WORDS = ("embed", "whisper", "tts", "asr", "speech", "music", "rerank", "guard", "safety",
                  "diffusion", "flux", "image", "vision", "ocr")


def _model_info(model) -> dict:
    """Model entry from client.models.list() as a plain dict (keeps Vultr's extra metadata)."""
    if isinstance(model, dict):
        return model
    if isinstance(model, str):
        return {"id": model}
    try:
        return model.model_dump()
    except Exception:
        return {"id": str(getattr(model, "id", ""))}


def score_model(model) -> int:
    """Higher is better; -1 means "not a chat model". Used when VULTR_MODEL is not set."""
    info = _model_info(model)
    name = str(info.get("id", "")).lower()
    outputs = [o.get("type") for o in info.get("output_modalities") or [] if isinstance(o, dict)]
    if not name or (outputs and "text" not in outputs) or any(w in name for w in NON_CHAT_WORDS):
        return -1
    score = 1 if outputs else 0  # the endpoint itself confirms it outputs text
    if any(family in name for family in ("llama", "qwen", "mistral")):
        score += 2
    if "instruct" in name or "chat" in name:
        score += 3
    if "flash" in name or "mini" in name:
        score += 1  # faster answers for a live demo
    if "coder" in name:
        score -= 1
    return score


def pick_model(models: list) -> str | None:
    """Pick the best chat model from a list of model ids or metadata dicts."""
    scored = [(score_model(m), _model_info(m)["id"]) for m in models]
    scored = [(score, name) for score, name in scored if score >= 0]
    if not scored:
        return None
    best = max(score for score, _ in scored)
    return next(name for score, name in scored if score == best)  # keep the API's order on ties


def _reasoning_effort(info: dict) -> str | None:
    """Lowest reasoning effort the model advertises, so answers come back fast and within max_tokens."""
    reasoning = info.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    supported = reasoning.get("supported_efforts") or []
    return next((e for e in ("minimal", "low") if e in supported), None)


def _get_model(client) -> str | None:
    """VULTR_MODEL if set, otherwise auto-pick from client.models.list() once and cache it."""
    global _model, _effort
    if _model:
        return _model
    configured = os.getenv("VULTR_MODEL", "").strip()
    try:
        infos = [_model_info(m) for m in client.models.list().data]
    except Exception as exc:
        if not configured:
            raise
        logger.info("Could not list models (%s); using VULTR_MODEL as configured.", type(exc).__name__)
        infos = []
    _model = configured or pick_model(infos)
    if _model:
        _effort = _reasoning_effort(next((i for i in infos if i.get("id") == _model), {}))
        logger.info("Vultr Serverless Inference model selected: %s", _model)
    else:
        logger.warning("No suitable chat model found on Vultr endpoint; using offline fallback.")
    return _model


def status() -> dict:
    """Small status dict for the UI (never includes the key)."""
    return {"configured": bool(_api_key()), "model": _model or os.getenv("VULTR_MODEL", "").strip() or None}


def _chat(messages: list[dict], max_tokens: int = 700) -> str | None:
    """One chat completion. Returns the text, or None on ANY problem."""
    global _cooldown_until, _effort
    client = _get_client()
    if client is None or time.time() < _cooldown_until:
        return None
    try:
        with _lock:
            model = _get_model(client)
        if not model:
            _cooldown_until = time.time() + COOLDOWN_SECONDS
            return None
        kwargs = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens + 800}
        try:
            resp = client.chat.completions.create(
                **kwargs, extra_body={"reasoning_effort": _effort} if _effort else None)
        except Exception as exc:
            if not _effort or getattr(exc, "status_code", None) != 400:
                raise
            logger.info("Endpoint rejected reasoning_effort; retrying without it.")
            _effort = None
            resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content if resp.choices else None
        if not text:
            return None
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()  # reasoning models
        return text or None
    except Exception as exc:  # network, auth, timeout, bad model... never crash the page
        code = getattr(exc, "status_code", "")
        logger.warning("LLM call failed (%s %s); using offline fallback.", type(exc).__name__, code)
        _cooldown_until = time.time() + COOLDOWN_SECONDS
        return None


def extract_json(text: str | None) -> dict | None:
    """Pull the first JSON object out of LLM output (handles code fences and chatter)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start:i + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None


# ---------------------------------------------------------------- job 1: parse a supplier email

MONTH_FORMATS = ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%d %B, %Y"]


def _parse_any_date(text: str) -> date | None:
    """Find the first recognisable date in free text."""
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    month = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
    patterns = [rf"\b{month}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{month},?\s+\d{{4}}\b"]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            candidate = re.sub(r"(\d)(st|nd|rd|th)", r"\1", m.group(0)).replace(".", "").replace("Sept", "Sep")
            for fmt in MONTH_FORMATS:
                try:
                    return datetime.strptime(candidate.strip(), fmt).date()
                except ValueError:
                    continue
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)  # assume DD/MM/YYYY
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


def fallback_parse_email(text: str, data: dict) -> dict:
    """Regex/keyword parser used when the LLM is unavailable. Never raises."""
    text = text or ""
    lower = text.lower()
    fields = {"supplier": "", "product": "", "quantity": None, "new_delivery_date": "",
              "reason": "", "shipment_id": "", "delay_days": None, "cancelled": False}

    m = re.search(r"\bSH-?\s?(\d{3,6})\b", text, flags=re.IGNORECASE)
    if m:
        fields["shipment_id"] = f"SH-{m.group(1)}"

    for s in data.get("suppliers", []):
        name = s["name"]
        first_word = name.split()[0].lower().rstrip(".,")
        if name.lower().rstrip(".") in lower or (len(first_word) > 4 and first_word in lower):
            fields["supplier"] = name
            break

    for p in data.get("products", []):
        singular = p["name"].lower().rstrip("s")
        if singular in lower:
            fields["product"] = p["name"]
            break

    without_ids = re.sub(r"\bSH-?\s?\d+\b|\b\d{4}-\d{2}-\d{2}\b", " ", text, flags=re.IGNORECASE)
    qty = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,7})\s*(?:units?|pcs|pieces)\b", without_ids, flags=re.IGNORECASE)
    if not qty and fields["product"]:
        singular = re.escape(fields["product"].lower().rstrip("s"))
        qty = re.search(rf"\b(\d{{1,3}}(?:,\d{{3}})+|\d{{2,7}})\s+(?:\w+\s+)?{singular}", without_ids, flags=re.IGNORECASE)
    if qty:
        fields["quantity"] = int(qty.group(1).replace(",", ""))

    new_date = _parse_any_date(text)
    if new_date:
        fields["new_delivery_date"] = new_date.isoformat()
    m = re.search(r"\b(?:delayed|pushed back|postponed|late)\s+(?:by\s+)?(\d{1,3})\s+days?\b|\b(\d{1,3})[- ]day delay\b",
                  text, flags=re.IGNORECASE)
    if m:
        fields["delay_days"] = int(m.group(1) or m.group(2))

    m = re.search(r"\b(?:due to|because of|caused by|owing to|as a result of)\s+([^.\n]{3,160})", text, flags=re.IGNORECASE)
    if m:
        fields["reason"] = m.group(1).strip().rstrip(",;")
    fields["cancelled"] = bool(re.search(r"\bcancel(?:l?ed|lation)?\b", lower))
    return fields


def _valid_llm_fields(raw: dict | None) -> dict:
    """Keep only well-formed values from the LLM's JSON."""
    if not isinstance(raw, dict):
        return {}
    clean = {}
    for key in ("supplier", "product", "reason", "shipment_id"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() not in ("null", "none", "unknown"):
            clean[key] = value.strip()[:200]
    qty = raw.get("quantity")
    try:
        qty = int(str(qty).replace(",", "").strip())
        if 0 < qty < 10_000_000:
            clean["quantity"] = qty
    except (TypeError, ValueError):
        pass
    d = raw.get("new_delivery_date")
    if isinstance(d, str):
        try:
            clean["new_delivery_date"] = date.fromisoformat(d.strip()[:10]).isoformat()
        except ValueError:
            pass
    if "shipment_id" in clean and not re.fullmatch(r"SH-\d{3,6}", clean["shipment_id"], flags=re.IGNORECASE):
        del clean["shipment_id"]
    return clean


def parse_email(text: str, data: dict, today: date | None = None) -> tuple[dict, str]:
    """Extract {supplier, product, quantity, new_delivery_date, reason, ...} from an email."""
    today = today or date.today()
    fields = fallback_parse_email(text, data)
    source = SOURCE_FALLBACK
    if not text or not text.strip():
        return fields, source

    supplier_names = ", ".join(s["name"] for s in data.get("suppliers", []))
    product_names = ", ".join(p["name"] for p in data.get("products", []))
    messages = [
        {"role": "system", "content": (
            "You extract structured data from supplier emails. Reply with ONLY one JSON object, no prose, "
            "with keys: supplier (string), product (string), quantity (integer), "
            "new_delivery_date (YYYY-MM-DD string), reason (short string), shipment_id (like SH-1042). "
            "Use null for anything not stated in the email. Never guess. "
            f"Today is {today.isoformat()}. Known suppliers: {supplier_names}. Known products: {product_names}.")},
        {"role": "user", "content": text[:4000]},
    ]
    llm = _valid_llm_fields(extract_json(_chat(messages, max_tokens=300)))
    if llm:
        source = SOURCE_LLM
        fields.update(llm)
    return fields, source


# ---------------------------------------------------------------- job 2: executive recommendation

def fallback_recommendation(facts: dict, simulation: bool = False) -> str:
    """Deterministic recommendation text built only from the computed facts."""
    prefix = "In this hypothetical scenario, shipment" if simulation else "Shipment"
    head = (f"{prefix} {facts.get('shipment_id')} from {facts.get('supplier')} "
            f"({facts.get('shipment_quantity')} {facts.get('product')}) is now expected {facts.get('current_eta')}")
    delay = facts.get("delay_days", 0)
    head += f", {delay} days late. " if delay else ". "

    if facts.get("stockout_before_shipment_arrives") != "yes":
        return head + (f"Current stock of {facts.get('on_hand_units')} units covers production until the shipment "
                       "arrives, so no customer orders are at risk. Recommendation: request an updated commitment "
                       "from the supplier and keep monitoring.")

    text = head + (f"With {facts.get('on_hand_units')} units on hand and {facts.get('weekly_usage_units')} used per week, "
                   f"{facts.get('product')} run out on {facts.get('stockout_date_with_delay')}, before the shipment arrives. ")
    if facts.get("orders_at_risk"):
        text += (f"{facts.get('orders_at_risk')} customer order(s) ({facts.get('order_units_at_risk')} units, "
                 f"{facts.get('high_priority_orders_at_risk')} high priority) are at risk. ")
    if facts.get("best_alternative_supplier", "none available") != "none available":
        timing = ("arriving " + facts.get("best_alternative_arrival", "") +
                  (", before the stockout" if facts.get("best_alternative_arrives_before_stockout") == "yes"
                   else ", after the stockout"))
        text += (f"Recommendation: order {facts.get('best_alternative_quantity')} units from "
                 f"{facts.get('best_alternative_supplier')} ({timing}) for {facts.get('best_alternative_total_cost')}, "
                 f"a premium of {facts.get('best_alternative_cost_premium')} over the primary supplier; ")
    else:
        text += "No alternative supplier is available, so recommendation: "
    text += (f"reserve stock for high-priority orders, request a firm commitment from {facts.get('supplier')}, "
             "and notify affected customers now.")
    return text


def recommend(facts: dict, simulation: bool = False) -> tuple[str, str]:
    """Short executive recommendation. Cached per set of facts so pages reload instantly."""
    key = json.dumps({"facts": facts, "sim": simulation}, sort_keys=True, default=str)
    if key in _cache:
        return _cache[key], SOURCE_LLM
    messages = [
        {"role": "system", "content": (
            "You are a supply chain analyst writing for a busy executive. Use ONLY the facts in the JSON "
            "provided. Do not calculate, estimate or invent any numbers, dates, names or costs; copy them "
            "exactly as given. Write 3 to 5 plain sentences: the problem, the business impact, and a clear "
            "recommended action. No headings, no bullet points, no markdown."
            + (" This is a hypothetical simulation; say so." if simulation else ""))},
        {"role": "user", "content": json.dumps(facts, default=str)},
    ]
    text = _chat(messages, max_tokens=400)
    if text and 40 <= len(text) <= 2500 and not text.lstrip().startswith("{"):
        text = text.replace("**", "").strip()
        if len(_cache) > 200:
            _cache.clear()
        _cache[key] = text
        return text, SOURCE_LLM
    return fallback_recommendation(facts, simulation), SOURCE_FALLBACK


# ---------------------------------------------------------------- job 3: draft emails

def fallback_emails(facts: dict) -> dict:
    supplier_email = (
        f"Subject: Request for updated delivery commitment - {facts.get('shipment_id')}\n\n"
        f"Dear {facts.get('supplier')} team,\n\n"
        f"Shipment {facts.get('shipment_id')} ({facts.get('shipment_quantity')} {facts.get('product')}) was due "
        f"{facts.get('original_delivery_date')} and is now expected {facts.get('current_eta')}.\n"
        + (f"At this date our {facts.get('product')} stock runs out on {facts.get('stockout_date_with_delay')}, "
           "which puts customer deliveries at risk.\n" if facts.get("stockout_before_shipment_arrives") == "yes" else "")
        + "\nPlease reply within 24 hours with:\n"
          "1. A firm, committed delivery date.\n"
          "2. Whether a partial shipment can be expedited.\n"
          "3. The root cause and what is being done to prevent a repeat.\n\n"
          "Best regards,\nSupply Chain Operations"
    )
    customer_email = (
        f"Subject: Update on your order\n\n"
        "Dear valued customer,\n\n"
        f"We want to let you know early that a delay from one of our {facts.get('product', '').lower()} suppliers "
        f"may affect your order. Affected order(s): {facts.get('affected_orders')}.\n\n"
        "We are already sourcing components from an alternative supplier and protecting stock for priority "
        "orders to keep any delay as short as possible. We will confirm a firm delivery date within 2 business days.\n\n"
        "We apologise for the inconvenience and appreciate your patience.\n\n"
        "Kind regards,\nCustomer Success Team"
    )
    return {"supplier_email": supplier_email, "customer_email": customer_email}


def draft_emails(facts: dict) -> tuple[dict, str]:
    """Supplier follow-up + customer delay notification, from computed facts only."""
    messages = [
        {"role": "system", "content": (
            "You draft short, professional business emails. Use ONLY the facts in the JSON provided; never "
            "invent numbers, dates, names or prices. Do not mention internal costs or prices to customers. "
            "Reply with ONLY one JSON object with two string keys: "
            "\"supplier_email\" (to the supplier, requesting a firm updated delivery commitment) and "
            "\"customer_email\" (to affected customers, explaining the possible delay and next steps). "
            "Start each email with a 'Subject:' line.")},
        {"role": "user", "content": json.dumps(facts, default=str)},
    ]
    raw = extract_json(_chat(messages, max_tokens=900))
    if raw:
        s_mail, c_mail = raw.get("supplier_email"), raw.get("customer_email")
        if isinstance(s_mail, str) and isinstance(c_mail, str) and len(s_mail) > 40 and len(c_mail) > 40:
            return {"supplier_email": s_mail.strip(), "customer_email": c_mail.strip()}, SOURCE_LLM
    return fallback_emails(facts), SOURCE_FALLBACK


# ---------------------------------------------------------------- sample email for the demo

def sample_email(data: dict, today: date | None = None) -> str:
    """Realistic delay email for SH-1042 (10 days late), with dates relative to today."""
    today = today or date.today()
    shipment = next((s for s in data.get("shipments", []) if s["id"] == "SH-1042"), None)
    expected = date.fromisoformat(shipment["expected_date"]) if shipment else today + timedelta(days=7)
    new_date = expected + timedelta(days=10)
    return (
        "From: logistics@techparts-ltd.example\n"
        "Subject: URGENT - Delivery delay for shipment SH-1042\n\n"
        "Dear Procurement Team,\n\n"
        "We regret to inform you that your order of 5,000 units of Microchips (shipment SH-1042) "
        "will not ship as planned due to a power outage at our wafer fabrication plant.\n\n"
        f"The new expected delivery date is {new_date.strftime('%B %d, %Y')}.\n\n"
        "We apologise for the inconvenience and will share further updates as soon as possible.\n\n"
        "Kind regards,\nTechParts Ltd. Logistics Team"
    )
