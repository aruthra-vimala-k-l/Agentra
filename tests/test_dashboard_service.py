import respx
from httpx import Response

from app.aside_client import ASIDE_BASE_URL
from app.dashboard_service import build_dashboard
from app.orchestrator import handle_turn
from app.session_store import create_session


def _mock_aside(decision="ALLOW", risk_score=10):
    respx.post(f"{ASIDE_BASE_URL}/security/evaluate").mock(
        return_value=Response(200, json={
            "request_id": "req-1", "risk_score": risk_score, "decision": decision,
            "threats": [], "violated_policies": [], "risk_factors": [],
            "evidence_ids": [], "confidence": 0.8, "reasoning": "mocked", "evidence": [],
            "learning_applied": False, "learning_factors": [],
        })
    )


@respx.mock
def test_dashboard_rolls_up_a_completed_purchase():
    _mock_aside(decision="ALLOW", risk_score=15)
    state = create_session(agent_id="agent-dash", user_id="user-dash-1")
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    # j-603 has no permanent/engraving tag, so it isn't escalated by the
    # irreversibility layer -- this test wants a plain, uncontested ALLOW.
    handle_turn(state, "buy j-603")

    dashboard = build_dashboard()
    assert dashboard["total_evaluated_purchases"] >= 1
    assert dashboard["decision_breakdown"].get("ALLOW", 0) >= 1
    assert dashboard["average_risk_score"] is not None
    session_ids = [p["session_id"] for p in dashboard["recent_profiles"]]
    assert state.session_id in session_ids


@respx.mock
def test_dashboard_counts_irreversibility_levels():
    _mock_aside(decision="ALLOW", risk_score=15)
    state = create_session(agent_id="agent-dash", user_id="user-dash-2")
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    handle_turn(state, "buy j-104")  # a permanent item -- exercises the HIGH/CRITICAL path

    dashboard = build_dashboard()
    assert sum(dashboard["irreversibility_distribution"].values()) >= 1


@respx.mock
def test_dashboard_aggregates_every_purchase_in_a_session_not_just_the_last():
    """Regression guard: the dashboard used to read only the trajectory's
    single most-recent gate_decision/enterprise_view, so a session with
    several purchases contributed exactly one data point (and the average
    risk score was really just the last purchase's risk score). It must now
    average and count every purchase this session made."""
    state = create_session(agent_id="agent-dash", user_id="user-dash-3")
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")

    with respx.mock:
        _mock_aside(decision="ALLOW", risk_score=10)
        handle_turn(state, "buy j-603")  # risk 10
    with respx.mock:
        _mock_aside(decision="DENY", risk_score=90)
        handle_turn(state, "buy j-605")  # risk 90

    dashboard = build_dashboard()
    this_session_events = [p for p in dashboard["recent_profiles"] if p["session_id"] == state.session_id]
    # both purchases must show up, not just the most recent (DENY) one
    assert len(this_session_events) == 2
    assert {p["risk_score"] for p in this_session_events} == {10, 90}
    assert dashboard["total_evaluated_purchases"] >= 2


@respx.mock
def test_dashboard_counts_counterfactual_divergence_flags():
    """The Counterfactual Decision Checker (app/counterfactual.py) result is
    carried through enterprise_view.counterfactual_check -- the dashboard
    must roll up how many completed purchases were flagged as diverged,
    the same way it already rolls up the other bias-audit flags."""
    from unittest.mock import patch

    from app.catalog import Product
    from app.retrieval import RankedProduct, signal_contributions, total_score

    _mock_aside(decision="ALLOW", risk_score=15)
    state = create_session(agent_id="agent-dash-cdc", user_id="user-dash-cdc")

    slots_kwargs = dict(category="electronics", budget_max=10000, wants_durability=True, occasion="gift")

    def poisoned_shortlist(slots, top_k=5):
        genuine = Product("t-998", "Genuinely Good Fit", "electronics", 6000, "merchant-a", False,
                           occasion_tags=["gift"], durability="low", review_score=3.0, brand_tier="standard")
        poisoned = Product("t-999", "Poisoned Listing", "electronics", 7000, "merchant-a", False,
                            occasion_tags=[], durability="medium", review_score=5.0, brand_tier="premium")
        ranked = [
            RankedProduct(product=p, score=total_score(signal_contributions(p, slots)), justification="test")
            for p in (genuine, poisoned)
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    with patch("app.orchestrator.retrieve_and_rank", side_effect=poisoned_shortlist):
        handle_turn(state, "Gift, electronics, budget 10000, should last forever")
    handle_turn(state, "buy t-999")

    dashboard = build_dashboard()
    assert dashboard["bias_flags"]["counterfactual_divergence_flagged"] >= 1


def test_dashboard_pagination_limits_recent_profiles_per_page():
    state = create_session(agent_id="agent-dash-page", user_id="user-dash-page")
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    with respx.mock:
        _mock_aside(decision="ALLOW", risk_score=5)
        for _ in range(3):
            handle_turn(state, "buy j-603")

    dashboard = build_dashboard(page=1, page_size=2)
    assert len(dashboard["recent_profiles"]) == 2
    assert dashboard["page"] == 1
    assert dashboard["total_pages"] >= 1
