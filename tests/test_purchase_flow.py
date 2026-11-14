import json

import pytest
import respx
from httpx import Response

from app import audit_log, payments
from app.orchestrator import SessionState, handle_turn
from tests.test_orchestrator import _mock_aside


def test_dry_run_is_default_and_idempotent():
    a = payments.execute_purchase("s", "j-102", "Ring", 1499)
    b = payments.execute_purchase("s", "j-102", "Ring", 1499)
    assert a["mode"] == "dry_run" and a["status"] == "simulated"
    assert b["order_id"] == a["order_id"] and b["idempotent_replay"] is True


def test_hard_cap_refuses_even_if_gate_allowed():
    with pytest.raises(payments.PaymentRefused):
        payments.execute_purchase("s", "x", "Big", 10_000_000)


def test_live_keys_refused(monkeypatch):
    monkeypatch.setenv("PAYMENT_MODE", "razorpay_test")
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_abc")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "s")
    with pytest.raises(payments.PaymentRefused):
        payments.execute_purchase("s", "j-102", "Ring", 1499)


@respx.mock
def test_razorpay_test_order_created(monkeypatch):
    monkeypatch.setenv("PAYMENT_MODE", "razorpay_test")
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_abc")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    route = respx.post(payments.ORDERS_URL).mock(return_value=Response(200, json={"id": "order_X1", "status": "created"}))
    o = payments.execute_purchase("s", "j-102", "Ring", 1499.5)
    assert o["order_id"] == "order_X1" and o["status"] == "created"
    assert json.loads(route.calls[0].request.content)["amount"] == 149950  # paise


def test_audit_chain_detects_tampering():
    audit_log.append("a", {"n": 1})
    audit_log.append("b", {"n": 2})
    assert audit_log.verify()["valid"] is True
    p = audit_log.log_path()
    p.write_text(p.read_text().replace('"n":1', '"n":9').replace('"n": 1', '"n": 9'))
    assert audit_log.verify()["valid"] is False


@respx.mock
def test_allowed_purchase_executes_and_is_audited():
    _mock_aside(decision="ALLOW")
    state = SessionState(session_id="s9", agent_id="agent-1", user_id="user-1")
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    # Some items (engraved/permanent) are escalated to human approval by the
    # irreversibility rule; buy the first shortlist item that is not.
    for rp in state.ranked:
        r = handle_turn(state, f"buy {rp.product.id}")
        if rp.product.id in state.orders:
            break
    pid = rp.product.id
    assert "purchased" in r.reply and pid in state.orders
    events = [json.loads(l)["event"] for l in audit_log.log_path().read_text().splitlines()]
    assert events[-2:] == ["gate_decision", "purchase_executed"]
    assert audit_log.verify()["valid"]


@respx.mock
def test_denied_purchase_never_reaches_payment():
    _mock_aside(decision="DENY", risk_score=95)
    state = SessionState(session_id="s10", agent_id="agent-1", user_id="user-1")
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    r = handle_turn(state, f"buy {state.ranked[0].product.id}")
    assert r.gate is not None and r.gate.decision == "DENY"
    assert state.orders == {}
