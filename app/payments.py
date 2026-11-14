"""Purchase execution -- the step AFTER the governance gate says yes.

Safety rules (enforced here, independent of the gate):
* Default mode is ``dry_run``: nothing leaves the process.
* ``razorpay_test`` mode only works with ``rzp_test_`` keys read from the
  environment (RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET). Live keys are refused.
  Keys are never logged or returned.
* A hard per-order cap (PURCHASE_HARD_CAP_INR, default 50,000) applies even if
  the gate allowed the purchase.
* Idempotent: the same (session, product) can only create one order.

Razorpay test mode creates an *Order* (a payable commitment). It does not move
money; completing payment needs Razorpay Checkout. We report status "created",
never "paid".
"""
from __future__ import annotations

import os
import uuid

import httpx

ORDERS_URL = "https://api.razorpay.com/v1/orders"
_issued: dict[str, dict] = {}


class PaymentRefused(Exception):
    pass


def mode() -> str:
    return os.environ.get("PAYMENT_MODE", "dry_run").strip().lower()


def hard_cap() -> float:
    return float(os.environ.get("PURCHASE_HARD_CAP_INR", "50000"))


def reset() -> None:
    _issued.clear()


def execute_purchase(session_id: str, product_id: str, product_name: str, amount_inr: float) -> dict:
    key = f"{session_id}:{product_id}"
    if key in _issued:
        return {**_issued[key], "idempotent_replay": True}
    if amount_inr <= 0 or amount_inr > hard_cap():
        raise PaymentRefused(f"amount ₹{amount_inr:,.2f} outside hard cap ₹{hard_cap():,.0f}")

    m = mode()
    if m == "dry_run":
        order = {"mode": "dry_run", "status": "simulated", "order_id": f"dry_{uuid.uuid4().hex[:14]}",
                 "amount_inr": amount_inr, "product_id": product_id}
    elif m == "razorpay_test":
        key_id, secret = os.environ.get("RAZORPAY_KEY_ID", ""), os.environ.get("RAZORPAY_KEY_SECRET", "")
        if not key_id.startswith("rzp_test_") or not secret:
            raise PaymentRefused("razorpay_test mode requires rzp_test_ credentials in the environment")
        r = httpx.post(ORDERS_URL, auth=(key_id, secret), timeout=15, json={
            "amount": int(round(amount_inr * 100)), "currency": "INR",
            "receipt": key[:40], "notes": {"product_id": product_id, "product": product_name[:200]}})
        if r.status_code >= 400:
            raise PaymentRefused(f"Razorpay rejected the order (HTTP {r.status_code})")
        d = r.json()
        order = {"mode": "razorpay_test", "status": d.get("status", "created"), "order_id": d["id"],
                 "amount_inr": amount_inr, "product_id": product_id}
    else:
        raise PaymentRefused(f"unknown PAYMENT_MODE {m!r}")
    _issued[key] = order
    return order


def verify_order(order: dict, expected_amount_inr: float | None = None) -> dict:
    """Retrieve the order and check it matches what we intended to buy.

    razorpay_test: GET /v1/orders/{id} from Razorpay and compare amount,
    currency and status against our record. dry_run: check the local record
    (nothing external exists to retrieve). Never reports "paid".
    """
    want = expected_amount_inr if expected_amount_inr is not None else order.get("amount_inr")
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if order.get("mode") == "razorpay_test":
        key_id, secret = os.environ.get("RAZORPAY_KEY_ID", ""), os.environ.get("RAZORPAY_KEY_SECRET", "")
        if not key_id.startswith("rzp_test_") or not secret:
            return {"verified": False, "source": "razorpay", "checks": [],
                    "error": "rzp_test_ credentials missing in environment"}
        try:
            r = httpx.get(f"{ORDERS_URL}/{order['order_id']}", auth=(key_id, secret), timeout=15)
        except httpx.HTTPError as exc:
            return {"verified": False, "source": "razorpay", "checks": [], "error": f"retrieval failed: {type(exc).__name__}"}
        if r.status_code >= 400:
            return {"verified": False, "source": "razorpay", "checks": [], "error": f"Razorpay HTTP {r.status_code}"}
        d = r.json()
        add("order exists at Razorpay", d.get("id") == order["order_id"], f"id {d.get('id')}")
        add("amount matches", d.get("amount") == int(round(want * 100)), f"Razorpay {d.get('amount')} paise vs expected {int(round(want * 100))}")
        add("currency is INR", d.get("currency") == "INR", str(d.get("currency")))
        add("status reported honestly", d.get("status") in ("created", "attempted", "paid"), f"Razorpay status: {d.get('status')}")
        return {"verified": all(c["ok"] for c in checks), "source": "razorpay", "status": d.get("status"), "checks": checks}

    rec = next((o for o in _issued.values() if o["order_id"] == order.get("order_id")), None)
    add("order exists in local record", rec is not None, order.get("order_id", ""))
    add("amount matches", rec is not None and abs(rec["amount_inr"] - want) < 0.01, f"₹{want:,.2f}")
    add("mode is dry_run (no money moved)", order.get("mode") == "dry_run", order.get("mode", ""))
    return {"verified": all(c["ok"] for c in checks), "source": "local_dry_run", "status": order.get("status"), "checks": checks}


def issued_orders() -> list[dict]:
    return list(_issued.values())
