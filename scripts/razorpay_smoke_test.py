"""Razorpay TEST-mode smoke test: create a small order, retrieve it, verify it.

Setup (never paste keys in chat): put these in agentra_shopping_agent/.env
    RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
    RAZORPAY_KEY_SECRET=xxxxxxxx
Run:  python scripts/razorpay_smoke_test.py
Creates ONE order of Rs 10 in Razorpay TEST mode. No money moves; status stays "created".
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import envload  # noqa: E402

envload.load()
os.environ["PAYMENT_MODE"] = "razorpay_test"
from app import payments  # noqa: E402


def main() -> int:
    if not os.environ.get("RAZORPAY_KEY_ID", "").startswith("rzp_test_"):
        print("RAZORPAY_KEY_ID missing or not a rzp_test_ key -- put TEST keys in .env (live keys are refused).")
        return 2
    sid = f"smoke-{uuid.uuid4().hex[:8]}"
    try:
        order = payments.execute_purchase(sid, "smoke-item", "Agentra smoke test item", 10.0)
    except payments.PaymentRefused as exc:
        print(f"Order refused: {exc}")
        return 1
    print(f"created: {order['order_id']} status={order['status']} amount=Rs {order['amount_inr']:.2f}")
    v = payments.verify_order(order)
    for c in v.get("checks", []):
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
    if v.get("error"):
        print("  error:", v["error"])
    again = payments.execute_purchase(sid, "smoke-item", "Agentra smoke test item", 10.0)
    print("idempotent replay:", bool(again.get("idempotent_replay")))
    try:
        payments.execute_purchase(sid, "too-big", "cap test", payments.hard_cap() + 1)
        print("hard cap: NOT enforced (unexpected)")
        return 1
    except payments.PaymentRefused:
        print("hard cap: enforced")
    print("RESULT:", "PASS" if v.get("verified") else "FAIL")
    return 0 if v.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
