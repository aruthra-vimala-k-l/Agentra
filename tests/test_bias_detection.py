from app.attribution import attribute
from app.bias_detection import (
    choice_homogeneity,
    nudge_exposure_report,
    position_bias_risk,
    review_authenticity_report,
    run_bias_audit,
)
from app.intent import IntentSlots
from app.retrieval import retrieve_and_rank


def test_position_bias_risk_is_low_with_a_decisive_score_gap():
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=7000, wants_durability=True, occasion="anniversary")
    ranked = retrieve_and_rank(slots)
    report = position_bias_risk(ranked)
    assert report["risk"] in {"low", "medium", "high"}
    assert report["score_gap"] is not None


def test_position_bias_risk_handles_a_single_candidate():
    slots = IntentSlots(raw_text="", category="care")
    ranked = retrieve_and_rank(slots, top_k=1)
    report = position_bias_risk(ranked)
    assert report["risk"] == "low"
    assert report["score_gap"] is None


def test_nudge_exposure_report_flags_promoted_items_as_clean_when_they_earn_no_rank_weight():
    slots = IntentSlots(raw_text="", category="luxury", budget_max=50000)
    ranked = retrieve_and_rank(slots)
    attributions = {r.product.id: attribute(r.product, slots) for r in ranked}
    report = nudge_exposure_report(ranked, attributions)
    assert report["promoted_items_in_shortlist"] >= 1
    assert report["all_clean"] is True
    assert all(f["clean"] for f in report["flags"])


def test_choice_homogeneity_flags_a_single_merchant_shortlist():
    # budget_max=400 keeps this to the hand-authored diy items (all from
    # merchant-established) -- the MartVerse-ingested diy items
    # (app/storefront.py) start at ₹449, so they're deliberately excluded
    # here rather than diluting a test that's specifically about a
    # single-merchant shortlist.
    slots = IntentSlots(raw_text="", category="diy", budget_max=400)
    ranked = retrieve_and_rank(slots, top_k=5)
    report = choice_homogeneity(ranked)
    assert report["distinct_merchants"] == 1
    assert report["homogeneity_flag"] is True


def test_choice_homogeneity_improves_once_the_martverse_storefront_is_included():
    """The flip side: without the budget cap above, the diy shortlist now
    draws from multiple merchants (the hand-authored catalogue plus
    MartVerse) -- real choice, not just a longer list from one seller."""
    slots = IntentSlots(raw_text="", category="diy")
    ranked = retrieve_and_rank(slots, top_k=5)
    report = choice_homogeneity(ranked)
    assert report["distinct_merchants"] > 1


def test_run_bias_audit_returns_all_four_checks():
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=7000)
    ranked = retrieve_and_rank(slots)
    attributions = {r.product.id: attribute(r.product, slots) for r in ranked}
    audit = run_bias_audit(ranked, attributions)
    assert set(audit.keys()) == {
        "position_bias", "nudge_exposure", "choice_homogeneity", "review_authenticity",
    }


def test_review_authenticity_report_flags_the_fabricated_storefront_listing():
    slots = IntentSlots(raw_text="", category="electronics", budget_max=5000)
    ranked = retrieve_and_rank(slots)
    report = review_authenticity_report(ranked)
    flagged_ids = {f["product_id"] for f in report["flagged"]}
    assert "e-602" in flagged_ids
