"""Autonomous purchasing run: goal in, verified order (or an honest stop) out.

The agent drives the same conversation a shopper would (so the same intent
parsing, ranking, ASIDE gate, audit log and payment safety rules apply), but
without a human typing each turn. Boundaries that stay human:
* missing information (e.g. no budget) -> stops and says what it needs;
* a gate result of REQUIRE_*_APPROVAL -> stops and leaves it pending;
* DENY -> never retried against the gate, moves to the next candidate.
Before buying it re-verifies the product record; after buying it retrieves the
order and checks it against what was intended.
"""
from __future__ import annotations

from app import audit_log, catalog, payments
from app.orchestrator import handle_turn
from app.session_store import create_session


def verify_product(product_id: str) -> dict:
    p = catalog.get_by_id(product_id)
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("product exists in catalogue", p is not None, product_id)
    if p is None:
        return {"verified": False, "checks": checks}
    add("in stock", p.in_stock)
    add("price within hard cap", 0 < p.price <= payments.hard_cap(), f"₹{p.price:,.2f}")
    add("provenance recorded", bool(p.source), p.source)
    add("has review evidence", len(p.reviews) > 0, f"{len(p.reviews)} reviews attached")
    return {"verified": all(c["ok"] for c in checks), "source": p.source, "checks": checks}


def run(goal: str, max_candidates: int = 5) -> dict:
    state = create_session(agent_id="autonomous-agent-01")
    steps: list[dict] = []
    audit_log.append("autonomous_run_started", {"session_id": state.session_id, "goal": goal[:300]})

    r = handle_turn(state, goal)
    steps.append({"step": "understand_and_search", "stage": r.stage, "reply": r.reply})
    if r.stage != "PRESENTING":
        return _finish(state, steps, "needs_information",
                       f"I need more detail before I can shop: {r.reply}")

    candidates = [c["id"] for c in r.products][:max_candidates]
    attempts = []
    for pid in candidates:
        pv = verify_product(pid)
        if not pv["verified"]:
            attempts.append({"product_id": pid, "outcome": "skipped_failed_verification", "product_check": pv})
            continue
        t = handle_turn(state, f"buy {pid}")
        gate = t.gates[0] if t.gates else {}
        entry = {"product_id": pid, "gate_decision": gate.get("decision"), "product_check": pv, "reply": t.reply}
        if pid in state.orders:
            order = state.orders[pid]
            ov = payments.verify_order(order)
            audit_log.append("order_verified", {"session_id": state.session_id, "product_id": pid,
                                                "order_id": order["order_id"], "verified": ov["verified"]})
            entry.update(outcome="purchased", order=order, order_check=ov)
            attempts.append(entry)
            return _finish(state, steps, "purchased" if ov["verified"] else "purchased_unverified",
                           t.reply, attempts)
        entry["outcome"] = {"DENY": "denied_by_gate"}.get(gate.get("decision"), "awaiting_human_approval")
        attempts.append(entry)
        if entry["outcome"] == "awaiting_human_approval":
            return _finish(state, steps, "awaiting_human_approval",
                           f"The gate flagged {pid}; it needs a human decision, so I stopped.", attempts)
    return _finish(state, steps, "no_purchase", "No candidate passed verification and the gate.", attempts)


def _finish(state, steps, outcome, message, attempts=None):
    audit_log.append("autonomous_run_finished", {"session_id": state.session_id, "outcome": outcome})
    return {"session_id": state.session_id, "outcome": outcome, "message": message,
            "payment_mode": payments.mode(), "steps": steps, "attempts": attempts or [],
            "audit": audit_log.verify()}
