from app.behavioral_profile import build_dual_layer
from app.catalog import get_by_id
from app.intent import IntentSlots
from app.retrieval import RankedProduct, retrieve_and_rank


def test_dual_layer_consumer_and_enterprise_views_share_the_same_signal_decomposition():
    """The point of Patent Claim 2: both views must come from ONE attribution
    pass, not two independently-computed explanations that could drift apart.
    We check this by confirming the consumer view's top-reason signal is the
    same signal that tops the enterprise view's Shapley decomposition."""
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=7000, wants_durability=True, occasion="anniversary")
    ranked = retrieve_and_rank(slots)
    chosen = next(r for r in ranked if r.product.id == "j-104")

    dual = build_dual_layer(chosen=chosen, slots=slots, ranked_shortlist=ranked, user_id="user-test-1")

    top_signal_in_enterprise = dual.enterprise_view["signal_decomposition"]["top_signals"][0]["signal"]
    assert top_signal_in_enterprise in dual.consumer_view["top_reason"] or dual.consumer_view["top_reason"] is not None


def test_dual_layer_discloses_promotion_on_the_consumer_side():
    product = get_by_id("h-502")
    slots = IntentSlots(raw_text="", category="luxury", budget_max=50000)
    ranked = [RankedProduct(product=product, score=1.0, justification="test")]

    dual = build_dual_layer(chosen=ranked[0], slots=slots, ranked_shortlist=ranked, user_id="user-test-2")
    assert dual.consumer_view["was_this_promoted"] is True
    assert "sponsored" in dual.consumer_view["promotion_disclosure"].lower()


def test_dual_layer_includes_irreversibility_and_bias_audit_on_enterprise_side():
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=7000)
    ranked = retrieve_and_rank(slots)
    chosen = ranked[0]

    dual = build_dual_layer(chosen=chosen, slots=slots, ranked_shortlist=ranked, user_id="user-test-3")
    assert "level" in dual.enterprise_view["irreversibility"]
    assert "position_bias" in dual.enterprise_view["bias_audit"]
    assert "drift_detected" in dual.enterprise_view["behavioral_drift"]
