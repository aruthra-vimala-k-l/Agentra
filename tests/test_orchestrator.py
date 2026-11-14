from unittest.mock import patch

import respx
from httpx import Response

from app.aside_client import ASIDE_BASE_URL
from app.catalog import Product
from app.orchestrator import SessionState, handle_turn
from app.retrieval import RankedProduct, signal_contributions, total_score


def _mock_aside(decision="ALLOW", risk_score=10):
    respx.post(f"{ASIDE_BASE_URL}/security/evaluate").mock(
        return_value=Response(200, json={
            "request_id": "req-1", "risk_score": risk_score, "decision": decision,
            "threats": [], "violated_policies": [], "risk_factors": [],
            "evidence_ids": [], "confidence": 0.8, "reasoning": "mocked", "evidence": [],
            "learning_applied": False, "learning_factors": [],
        })
    )


def _new_state() -> SessionState:
    return SessionState(session_id="s1", agent_id="agent-1", user_id="user-1")


def test_full_conversation_reaches_a_shortlist_without_clarifying_when_slots_are_complete():
    state = _new_state()
    result = handle_turn(state, "Anniversary gift, gold, budget 7000, should last forever")
    assert state.stage == "PRESENTING"
    assert len(result.products) > 0
    assert result.trajectory["stated_intent"]


def test_conversation_asks_a_clarifying_question_when_slots_missing():
    state = _new_state()
    result = handle_turn(state, "I need a gift for someone")
    assert state.stage == "COLLECTING"
    assert "?" in result.reply

    result2 = handle_turn(state, "jewellery, budget 3000")
    assert state.stage == "PRESENTING"
    assert len(result2.products) > 0
    assert len(state.trajectory.clarifying_exchange) == 1


@respx.mock
def test_selecting_a_product_calls_aside_and_allows():
    _mock_aside(decision="ALLOW")
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    # j-603 (genuine reviews, no permanent/engraving tag) is deliberately the
    # non-escalating jewellery item -- see irreversibility escalation tests
    # below for the permanent/engraved cases.
    result = handle_turn(state, "buy j-603")
    # Fixed a real bug: the session used to dead-end into "This session has
    # ended" after exactly one evaluation, regardless of the decision. It
    # must stay in PRESENTING so the shopper can check or buy another item.
    assert state.stage == "PRESENTING"
    assert result.gate.decision == "ALLOW"
    assert "allowed" in result.reply.lower()
    assert state.trajectory.final_selection["id"] == "j-603"


@respx.mock
def test_selecting_a_product_calls_aside_and_reports_deny_without_ending_session():
    _mock_aside(decision="DENY", risk_score=95)
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    result = handle_turn(state, "buy j-603")
    assert state.stage == "PRESENTING"
    assert result.gate.decision == "DENY"
    assert "blocked" in result.reply.lower()


@respx.mock
def test_selecting_a_product_escalates_to_human_approval_and_session_stays_open():
    _mock_aside(decision="REQUIRE_HUMAN_APPROVAL", risk_score=75)
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    result = handle_turn(state, "buy j-603")
    assert state.stage == "PRESENTING"
    assert result.gate.decision == "REQUIRE_HUMAN_APPROVAL"


@respx.mock
def test_permanent_item_is_escalated_to_human_approval_even_when_aside_allows():
    """The fix for 'no human_verification_required flag for merchant-
    established but irreversible purchases': ASIDE has no reason to flag
    j-104 (established merchant, under the high-value threshold) beyond its
    own amount-based guardrail, but it's a permanent/personalized item --
    Agentra's own irreversibility layer must escalate it regardless of what
    ASIDE said."""
    _mock_aside(decision="ALLOW", risk_score=20)
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    result = handle_turn(state, "buy j-104")
    assert result.gate.decision == "REQUIRE_HUMAN_APPROVAL"
    assert "agentra_irreversibility_escalation" in result.gate.risk_factors
    assert "irreversibility" in result.gate.reasoning.lower()


@respx.mock
def test_deny_from_aside_is_never_downgraded_by_the_irreversibility_layer():
    _mock_aside(decision="DENY", risk_score=95)
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    result = handle_turn(state, "buy j-104")
    assert result.gate.decision == "DENY"


def _poisoned_shortlist(slots):
    """A shortlist where a genuinely well-fitting item is edged out for #1
    only by a mediocre-fit item's spoofed review score and self-declared
    'premium' brand tag -- see tests/test_counterfactual.py for the same
    fixture shape in isolation."""
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


@respx.mock
def test_counterfactual_divergence_escalates_the_item_that_only_won_on_gameable_signals():
    """Feature/regression guard for the Counterfactual Decision Checker
    (proposal §5's CDC, app/counterfactual.py): a shortlist item that only
    took the #1 spot once review score and brand tier are counted -- not on
    price fit, durability, or occasion match -- must get escalated to
    REQUIRE_HUMAN_APPROVAL even when ASIDE itself sees nothing wrong and the
    irreversibility layer alone wouldn't have escalated it either (electronics
    at this amount scores MEDIUM, not HIGH)."""
    _mock_aside(decision="ALLOW", risk_score=15)
    state = _new_state()
    with patch("app.orchestrator.retrieve_and_rank", side_effect=lambda slots, top_k=5: _poisoned_shortlist(slots)):
        handle_turn(state, "Gift, electronics, budget 10000, should last forever")
    assert state.counterfactual is not None
    assert state.counterfactual.diverged is True
    assert state.counterfactual.actual_top_id == "t-999"

    result = handle_turn(state, "buy t-999")
    assert result.gate.decision == "REQUIRE_HUMAN_APPROVAL"
    assert "agentra_counterfactual_escalation" in result.gate.risk_factors


@respx.mock
def test_fabricated_review_listing_is_escalated_to_human_approval():
    """Feature/regression guard for the Review Authenticity check
    (app/review_authenticity.py, the ADM component): e-602 from the mock
    MartVerse storefront (bias_testbed_storefront.html / app/storefront.py)
    is deliberately built with all three reviews landing in the same 3-day
    burst window behind a fabricated 4.9-star average. Buying it must be
    escalated even though ASIDE sees nothing wrong and irreversibility alone
    wouldn't flag an ordinary electronics item at this amount."""
    _mock_aside(decision="ALLOW", risk_score=10)
    state = _new_state()
    handle_turn(state, "Gift, electronics, budget 5000, should last forever")
    result = handle_turn(state, "buy e-602")
    assert result.gate.decision == "REQUIRE_HUMAN_APPROVAL"
    assert "agentra_review_authenticity_escalation" in result.gate.risk_factors
    assert "fabricated" in result.gate.reasoning.lower()


@respx.mock
def test_a_second_product_can_be_checked_after_the_first_without_a_new_session():
    """The core fix: the same session must support checking more than one
    shortlist item, one after another -- and each is judged on its own
    irreversibility, independent of the others."""
    _mock_aside(decision="ALLOW")
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    first = handle_turn(state, "buy j-603")   # plain ALLOW passes through
    second = handle_turn(state, "buy j-104")  # permanent item gets escalated
    assert first.gate.decision == "ALLOW"
    assert second.gate.decision == "REQUIRE_HUMAN_APPROVAL"
    assert state.stage == "PRESENTING"
    assert len(state.trajectory.purchase_history) == 2
    assert {p["product_id"] for p in state.trajectory.purchase_history} == {"j-104", "j-603"}


@respx.mock
def test_approving_a_pending_purchase_resolves_it_to_allow():
    """The fix for 'no additional process if a purchase requires human
    verification': a REQUIRE_HUMAN_APPROVAL result used to just sit there.
    Now a reviewer's 'approve <id>' actually resolves it."""
    _mock_aside(decision="ALLOW")
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    flagged = handle_turn(state, "buy j-104")
    assert flagged.gate.decision == "REQUIRE_HUMAN_APPROVAL"
    assert "j-104" in state.pending_approvals

    resolved = handle_turn(state, "approve j-104")
    assert resolved.gate.decision == "ALLOW"
    assert "j-104" not in state.pending_approvals
    assert state.trajectory.purchase_history[-1]["decision"] == "ALLOW"


@respx.mock
def test_rejecting_a_pending_purchase_resolves_it_to_deny():
    _mock_aside(decision="ALLOW")
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    handle_turn(state, "buy j-104")
    resolved = handle_turn(state, "reject j-104")
    assert resolved.gate.decision == "DENY"
    assert "j-104" not in state.pending_approvals


def test_approving_something_not_pending_is_a_no_op_message():
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 2000, should last forever")
    result = handle_turn(state, "approve j-102")
    assert "isn't waiting on an approval" in result.reply


@respx.mock
def test_multiple_products_can_be_checked_in_a_single_message():
    _mock_aside(decision="ALLOW")
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    result = handle_turn(state, "check j-104 and j-103")
    assert len(result.gates) == 2
    ids = {g["product_id"] for g in result.gates}
    assert ids == {"j-104", "j-103"}
    assert len(state.trajectory.purchase_history) == 2


def test_picking_an_id_not_in_the_shortlist_is_rejected_without_calling_aside():
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 2000, should last forever")
    result = handle_turn(state, "buy z-999")
    assert state.stage == "PRESENTING"
    assert "wasn't in the shortlist" in result.reply


@respx.mock
def test_new_search_after_a_purchase_restarts_intent_collection():
    _mock_aside(decision="ALLOW")
    state = _new_state()
    handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
    handle_turn(state, "buy j-104")
    result = handle_turn(state, "new search: electronics, budget 4000")
    assert state.stage in {"COLLECTING", "PRESENTING"}
    # A fresh search should not carry over the old jewellery slots.
    assert state.slots.category == "electronics"


def test_bare_budget_answer_is_understood_after_the_budget_question():
    state = _new_state()
    handle_turn(state, "I need a gift")  # triggers category question first
    handle_turn(state, "electronics")     # resolves category, triggers budget question
    result = handle_turn(state, "4000")   # bare number answering the budget question
    assert state.slots.budget_max == 4000.0
    assert all(p["price"] <= 4000 for p in result.products)


def test_new_search_marks_the_result_as_restarted_so_the_ui_can_clear_stale_state():
    """Regression guard: the front end used to keep showing the previous
    shortlist/chat after "new search" because there was no signal that a
    reset had happened -- the backend reset its own slots, but nothing told
    static/app.js to clear the old product cards and chat log. TurnResult
    must now say so."""
    _mock_aside_module = respx
    with respx.mock:
        _mock_aside(decision="ALLOW")
        state = _new_state()
        handle_turn(state, "Anniversary gift, jewellery, budget 7000, should last forever")
        result_normal = handle_turn(state, "jewellery, budget 3000")  # NOT a reset
        result_reset = handle_turn(state, "new search")
    assert result_normal.restarted is False
    assert result_reset.restarted is True


def test_repeated_unrecognized_category_answers_give_an_honest_failure_not_the_whole_catalogue():
    """Regression guard: once the clarifying-question budget is exhausted
    while the category is still unknown (e.g. every answer was gibberish
    like "a"), the old code fell through to retrieve_and_rank(category=None),
    which doesn't filter by category and silently returned 5 products from
    the ENTIRE catalogue with nonsensical "matches the category you asked
    about" justifications for a request the agent never understood. It must
    now say honestly that nothing matched, with no products at all."""
    state = _new_state()
    handle_turn(state, "a")  # triggers the category clarifying question
    handle_turn(state, "a")  # unrecognized again -- exhausts the 2-question cap
    result = handle_turn(state, "a")
    assert result.products == []
    assert "couldn't tell" in result.reply.lower()
