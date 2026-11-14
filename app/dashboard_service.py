"""Enterprise Dashboard aggregation -- the "Govern" stage of Agentra's
pipeline (proposal §5): behavioral profiles, bias trends, and the audit
trail rolled up across sessions, for the enterprise operator rather than the
shopper.

Bug fix: this used to read only `trajectory.gate_decision` / `.enterprise_view`
-- the MOST RECENT purchase in each session -- so a session with five
evaluated purchases contributed exactly one data point to every stat, and
"recent behavioral profiles" silently dropped everything but each session's
last purchase. It now flattens EVERY entry in `trajectory.purchase_history`
across every session into one list before computing anything, so a session
with five purchases contributes five, and the profile list paginates
properly instead of hard-capping at 25 with no way to see the rest.

Reads directly from the in-memory session store (app/session_store.py) --
same honestly-disclosed limitation as that module: this is a per-process
view, not a durable audit log. A real deployment would persist
DecisionTrajectory rows to a database and query that instead of walking live
session objects.
"""
from __future__ import annotations

from collections import Counter

from app import audit_log
from app.irreversibility import classify_listing
from app.session_store import all_sessions

DEFAULT_PAGE_SIZE = 20


def _assistant_events() -> list[dict]:
    """Purchases gated by the shopping assistant. They live in the tamper-evident audit chain (not in the in-memory pipeline
    sessions), so they survive restarts and are counted here next to the pipeline's own sessions."""
    out = []
    for e in audit_log._read():
        if e.get("event") != "assistant_gate_decision":
            continue
        p = e.get("payload", {})
        out.append({"decision": p.get("verdict", "DENY"), "risk_score": int(p.get("risk_score") or 0), "product_name": p.get("name"),
                    "evaluated_at": e.get("ts", ""), "session_id": p.get("session_id") or "assistant", "user_id": "assistant-user",
                    "enterprise_view": {"irreversibility": classify_listing(
                        float(p.get("amount_inr") or 0),
                        seller_unverified=any("seller" in str(r).lower() for r in p.get("reasons", [])) or int(p.get("risk_score") or 0) >= 40)}})
    return out


def _flatten_purchase_events() -> list[dict]:
    """Every purchase-evaluation event across every session, each tagged
    with its session/user, newest first. This is the single source both the
    summary stats and the paginated profile table are built from, so the
    two can never disagree with each other."""
    events = []
    for state in all_sessions():
        traj = state.trajectory
        if traj is None:
            continue
        for entry in traj.purchase_history:
            events.append({**entry, "session_id": traj.session_id, "user_id": traj.user_id})
    events += _assistant_events()
    events.sort(key=lambda e: e.get("evaluated_at", ""), reverse=True)
    return events


def build_dashboard(page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> dict:
    events = _flatten_purchase_events()
    sessions = all_sessions()

    decision_counts: Counter[str] = Counter()
    risk_scores: list[int] = []
    irreversibility_levels: Counter[str] = Counter()
    homogeneity_flags = 0
    high_position_bias = 0
    dirty_nudge_flags = 0
    counterfactual_divergence_flags = 0
    review_authenticity_flags = 0
    drift_alerts = 0

    for e in events:
        decision_counts[e["decision"]] += 1
        risk_scores.append(e["risk_score"])

        ev = e.get("enterprise_view") or {}
        irr = ev.get("irreversibility")
        if irr:
            irreversibility_levels[irr["level"]] += 1

        bias = ev.get("bias_audit") or {}
        if bias.get("choice_homogeneity", {}).get("homogeneity_flag"):
            homogeneity_flags += 1
        if bias.get("position_bias", {}).get("risk") == "high":
            high_position_bias += 1
        if not bias.get("nudge_exposure", {}).get("all_clean", True):
            dirty_nudge_flags += 1

        cdc = ev.get("counterfactual_check") or {}
        if cdc.get("diverged"):
            counterfactual_divergence_flags += 1

        if bias.get("review_authenticity", {}).get("flagged_count", 0) > 0:
            review_authenticity_flags += 1

        drift = ev.get("behavioral_drift") or {}
        if drift.get("drift_detected"):
            drift_alerts += 1

    avg_risk = round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else None

    total_count = len(events)
    page = max(1, page)
    total_pages = max(1, -(-total_count // page_size)) if page_size > 0 else 1
    start = (page - 1) * page_size
    page_events = events[start:start + page_size]

    profiles = [{
        "session_id": e["session_id"],
        "user_id": e["user_id"],
        "product": e.get("product_name"),
        "decision": e["decision"],
        "risk_score": e["risk_score"],
        "irreversibility_level": (e.get("enterprise_view") or {}).get("irreversibility", {}).get("level"),
        "drift_detected": (e.get("enterprise_view") or {}).get("behavioral_drift", {}).get("drift_detected", False),
        "evaluated_at": e.get("evaluated_at"),
    } for e in page_events]

    return {
        "total_sessions": len(sessions) + len({e["session_id"] for e in events if e.get("user_id") == "assistant-user"}),
        "total_evaluated_purchases": total_count,
        "decision_breakdown": dict(decision_counts),
        "average_risk_score": avg_risk,
        "irreversibility_distribution": dict(irreversibility_levels),
        "bias_flags": {
            "choice_homogeneity_flagged": homogeneity_flags,
            "high_position_bias_flagged": high_position_bias,
            "dirty_nudge_exposure_flagged": dirty_nudge_flags,
            "counterfactual_divergence_flagged": counterfactual_divergence_flags,
            "review_authenticity_flagged": review_authenticity_flags,
        },
        "behavioral_drift_alerts": drift_alerts,
        "recent_profiles": profiles,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_profiles": total_count,
    }
