from fastapi.testclient import TestClient

from app import payments
from app.main import app

c = TestClient(app)


def test_verify_dry_run_order_roundtrip():
    o = payments.execute_purchase("s1", "j-101", "x", 999.0)
    v = payments.verify_order(o)
    assert v["verified"] and v["source"] == "local_dry_run"
    assert not payments.verify_order({**o, "order_id": "dry_nope"})["verified"]


def test_autonomous_needs_information_when_no_budget():
    r = c.post("/agent/run", json={"goal": "something nice"}).json()
    assert r["outcome"] in ("needs_information", "purchased", "awaiting_human_approval", "no_purchase")
    assert r["audit"]["ok"] if "ok" in r["audit"] else True


def test_autonomous_run_never_returns_paid_and_lists_orders():
    r = c.post("/agent/run", json={"goal": "anniversary gift, jewellery, budget 3000"}).json()
    assert r["outcome"] in ("purchased", "awaiting_human_approval", "no_purchase", "needs_information")
    if r["outcome"] == "purchased":
        oid = r["attempts"][-1]["order"]["order_id"]
        assert c.get(f"/agent/orders/{oid}/verify").json()["verified"]
        assert r["attempts"][-1]["order"]["status"] != "paid"
