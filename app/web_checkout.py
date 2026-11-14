"""Buy -> gate -> human approval -> DEMO wallet settlement, for the web shopping assistant.

SIMULATED PAYMENT: no real money moves. Product links/prices are real (read from the seller page); the wallet is a
ledger kept in Agentra's tamper-evident audit chain (data/audit/purchases.jsonl). The balance is DERIVED from the chain
(start - debits + refunds), so editing the log to fake a balance breaks the hash chain.

The gate here is a local deterministic rule set. It does NOT call the external ASIDE service the old /agent pipeline used.
Verdicts: ALLOW (settles at once), REQUIRE_HUMAN_APPROVAL (waits for Approve/Reject), DENY (never settles).
"""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone

from app import audit_log, payments, web_assistant

ALLOW, ESCALATE, DENY = "ALLOW", "REQUIRE_HUMAN_APPROVAL", "DENY"
INJECTION = web_assistant.INJECTION
_ORDERS: dict[str, dict] = {}


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def limits() -> dict:
    return {"start_balance": _f("DEMO_WALLET_START_INR", 5000), "per_purchase_cap": _f("DEMO_PER_PURCHASE_CAP_INR", 3000),
            "daily_cap": _f("DEMO_DAILY_CAP_INR", 4000), "approval_above": _f("DEMO_APPROVAL_ABOVE_INR", 1500),
            "hard_cap": payments.hard_cap()}


# ------------------------------------------------------------------ wallet (derived from the audit chain)
def _money_events() -> list[dict]:
    return [e for e in audit_log._read() if e.get("event") in ("wallet_debit", "wallet_refund", "wallet_topup")]


def wallet() -> dict:
    lim = limits()
    today = datetime.now(timezone.utc).date().isoformat()
    bal, spent_today = lim["start_balance"], 0.0
    for e in _money_events():
        amt = float(e["payload"].get("amount_inr", 0))
        sign = -1 if e["event"] == "wallet_debit" else 1
        bal += sign * amt
        if e["event"] == "wallet_topup":
            continue                       # a top-up changes the balance only, never "spent today"
        if e["ts"][:10] == today:
            spent_today -= sign * amt
    return {"mode": "DEMO (simulated payment, no real money)", "balance_inr": round(bal, 2),
            "spent_today_inr": round(max(spent_today, 0), 2), "limits": lim}


def set_balance(target_inr: float) -> dict:
    """Demo-only: set the balance to target_inr by appending a signed top-up to the audit chain (history, spent today
    and limits are untouched, and the hash chain stays valid)."""
    delta = round(float(target_inr) - wallet()["balance_inr"], 2)
    audit_log.append("wallet_topup", {"amount_inr": delta, "target_balance_inr": float(target_inr), "simulated": True})
    return wallet()


# ------------------------------------------------------------------ gate
def gate(row: dict, w: dict | None = None) -> dict:
    """Pure rule set over an evaluated result row + wallet state. Returns verdict, risk score and plain reasons."""
    w = w or wallet()
    lim = w["limits"]
    deny: list[str] = []
    esc: list[str] = []
    price = row.get("price_inr")
    url = row.get("url") or ""
    if not isinstance(price, (int, float)) or price <= 0:
        deny.append("The seller page shows no usable price, so the amount cannot be verified.")
    if not url.lower().startswith("https://"):
        deny.append("The link is not a secure (https) page.")
    if row.get("link_state") in ("dead", "unreachable"):
        deny.append("The link does not open a working page.")
    if INJECTION.search(" ".join(str(row.get(k, "")) for k in ("name", "why"))):
        deny.append("The product text contains instruction-like wording aimed at an AI agent (possible prompt injection).")
    if isinstance(price, (int, float)) and price > 0:
        if price > lim["hard_cap"]:
            deny.append(f"Above the system hard cap of ₹{lim['hard_cap']:.0f}.")
        if price > lim["per_purchase_cap"]:
            deny.append(f"Above the per-purchase limit of ₹{lim['per_purchase_cap']:.0f}.")
        if price > w["balance_inr"]:
            deny.append(f"Insufficient demo-wallet balance (₹{w['balance_inr']:.0f}).")
        if w["spent_today_inr"] + price > lim["daily_cap"]:
            deny.append(f"Would exceed the daily limit of ₹{lim['daily_cap']:.0f} (₹{w['spent_today_inr']:.0f} already spent today).")
        if price > lim["approval_above"]:
            esc.append(f"Amount is above ₹{lim['approval_above']:.0f}, so a person must approve it.")
    auth = row.get("authenticity") or {}
    if auth.get("suspicious"):
        esc.append("The reviews look unusual (" + "; ".join(auth.get("reasons", []))[:160] + "), so a person should look first.")
    elif isinstance(row.get("risk_score"), (int, float)) and row["risk_score"] >= 25 and row.get("risk_level") != "High risk":
        esc.append(f"Several smaller warning signs add up (risk score {row['risk_score']}).")
    if row.get("risk_level") == "Unverified":
        esc.append("Key facts (price, seller or ratings) could not be confirmed, so a person should check first.")
    if row.get("risk_level") == "High risk":
        esc.append("Agentra's pre-payment check marked this seller page High risk.")
    if not row.get("grounded"):
        esc.append("The page could not be confirmed in the search results.")
    host = web_assistant._registrable(row.get("host") or "")
    if host and host not in web_assistant.KNOWN_SITES:
        esc.append(f"{host} is not on the list of well-known retailers.")
    verdict = DENY if deny else (ESCALATE if esc else ALLOW)
    reasons = deny + esc or ["Known seller, page confirmed, low risk, amount within limits and wallet balance."]
    return {"verdict": verdict, "reasons": reasons, "risk_score": row.get("risk_score"), "risk_level": row.get("risk_level")}


# ------------------------------------------------------------------ flow
def _public(o: dict) -> dict:
    return {k: v for k, v in o.items() if not k.startswith("_")}


def _row_for(sid: str, rank: int, url: str | None = None) -> dict:
    st = web_assistant._SESSIONS.get(sid or "")
    if url:     # exact item the buyer clicked, even if the shortlist has since changed
        row = ((st or {}).get("seen") or {}).get(url)
        if row:
            return row
        raise LookupError("That item is no longer in this chat. Ask for it again.")
    rows = ((st or {}).get("results") or {}).get("results") or []
    for r in rows:
        if r.get("evaluated_rank") == rank:
            return r
    raise LookupError("That item is no longer in this chat. Ask for the shortlist again.")


def _settle(o: dict) -> None:
    w = wallet()
    if o["amount_inr"] > w["balance_inr"]:           # re-check at the moment of payment
        o["status"] = "blocked"
        o["gate"]["reasons"].append("Wallet balance changed before settlement; payment not made.")
        audit_log.append("assistant_order_blocked", {"order_id": o["order_id"], "session_id": o["session_id"], "reason": "balance"})
        return
    o["receipt"] = "DEMO-" + uuid.uuid4().hex[:10].upper()
    o["status"] = "paid_demo"
    audit_log.append("wallet_debit", {"order_id": o["order_id"], "session_id": o["session_id"], "amount_inr": o["amount_inr"],
                                      "receipt": o["receipt"], "name": o["name"], "host": o["host"], "url": o["url"], "simulated": True})
    o["wallet"] = wallet()


def buy(session_id: str, rank: int, url: str | None = None) -> dict:
    row = _row_for(session_id, rank, url)                 # price/risk come from the server's own record, never from the browser
    for o in {**_from_audit(), **_ORDERS}.values():  # idempotent: one live order per session+item
        if o["session_id"] == session_id and o["url"] == row.get("url") and o["status"] in ("pending_approval", "paid_demo"):
            return _public(o)
    g = gate(row)
    o = {"order_id": "ord_" + uuid.uuid4().hex[:8], "session_id": session_id, "name": row.get("name"), "host": row.get("host"),
         "url": row.get("url"), "amount_inr": float(row["price_inr"]) if isinstance(row.get("price_inr"), (int, float)) else 0.0,
         "gate": g, "status": {ALLOW: "approved", ESCALATE: "pending_approval", DENY: "blocked"}[g["verdict"]], "receipt": None}
    _ORDERS[o["order_id"]] = o
    audit_log.append("assistant_gate_decision", {"order_id": o["order_id"], "session_id": session_id, "name": o["name"], "host": o["host"], "url": o["url"],
                                                 "amount_inr": o["amount_inr"], "verdict": g["verdict"], "reasons": g["reasons"],
                                                 "risk_score": g["risk_score"]})
    if g["verdict"] == ALLOW:
        _settle(o)
    return _public(o)


def _from_audit() -> dict[str, dict]:
    """Rebuild every order from the audit chain, so orders and pending approvals survive a server restart."""
    out: dict[str, dict] = {}
    for e in audit_log._read():
        p, ev = e.get("payload", {}), e.get("event")
        oid = p.get("order_id")
        if ev == "assistant_gate_decision":
            v = p.get("verdict")
            out[oid] = {"order_id": oid, "session_id": p.get("session_id"), "name": p.get("name"), "host": p.get("host"),
                        "url": p.get("url"), "amount_inr": p.get("amount_inr", 0.0), "receipt": None,
                        "gate": {"verdict": v, "reasons": p.get("reasons", []), "risk_score": p.get("risk_score"), "risk_level": None},
                        "status": {ALLOW: "approved", ESCALATE: "pending_approval", DENY: "blocked"}.get(v, "blocked")}
        elif oid in out:
            o = out[oid]
            if ev == "assistant_human_decision" and not p.get("approved"):
                o["status"] = "rejected"
            elif ev == "wallet_debit":
                o["status"], o["receipt"] = "paid_demo", p.get("receipt")
            elif ev == "wallet_refund":
                o["status"] = "refunded_demo"
            elif ev == "assistant_order_blocked":
                o["status"] = "blocked"
    return out


def _get(order_id: str) -> dict | None:
    if order_id not in _ORDERS:
        rebuilt = _from_audit().get(order_id)
        if rebuilt:
            _ORDERS[order_id] = rebuilt
    return _ORDERS.get(order_id)


def gate_counts() -> dict:
    c = {"ALLOW": 0, "REQUIRE_HUMAN_APPROVAL": 0, "DENY": 0, "approved_by_human": 0, "rejected_by_human": 0}
    for e in audit_log._read():
        if e.get("event") == "assistant_gate_decision" and e["payload"].get("verdict") in c:
            c[e["payload"]["verdict"]] += 1
        elif e.get("event") == "assistant_human_decision":
            c["approved_by_human" if e["payload"].get("approved") else "rejected_by_human"] += 1
    return c


def decide(order_id: str, approve: bool, who: str = "user") -> dict:
    o = _get(order_id)
    if not o:
        raise LookupError("Unknown order.")
    if o["status"] != "pending_approval":
        return _public(o)                            # already settled/rejected: do nothing twice
    audit_log.append("assistant_human_decision", {"order_id": order_id, "session_id": o["session_id"], "approved": bool(approve), "by": who})
    if approve:
        _settle(o)
    else:
        o["status"] = "rejected"
    return _public(o)


def refund(order_id: str) -> dict:
    o = _get(order_id)
    if not o:
        raise LookupError("Unknown order.")
    if o["status"] != "paid_demo":
        return _public(o)
    o["status"] = "refunded_demo"
    audit_log.append("wallet_refund", {"order_id": order_id, "session_id": o["session_id"], "amount_inr": o["amount_inr"], "simulated": True})
    o["wallet"] = wallet()
    return _public(o)


def orders(session_id: str | None = None) -> list[dict]:
    merged = {**_from_audit(), **_ORDERS}
    return [_public(o) for o in merged.values() if not session_id or o["session_id"] == session_id]
