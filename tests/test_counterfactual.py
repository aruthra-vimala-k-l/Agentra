from app.catalog import Product
from app.counterfactual import rederive
from app.intent import IntentSlots
from app.retrieval import RankedProduct, signal_contributions, total_score


def _ranked(products: list[Product], slots: IntentSlots) -> list[RankedProduct]:
    """Build a RankedProduct shortlist the same way retrieve_and_rank does,
    sorted by the real (non-clean) score -- so these fixtures behave exactly
    like a shortlist the orchestrator would have actually shown."""
    scored = [
        RankedProduct(product=p, score=total_score(signal_contributions(p, slots)), justification="test")
        for p in products
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


def test_rederive_with_no_shortlist_is_a_no_op():
    result = rederive([], IntentSlots(raw_text=""))
    assert result.diverged is False
    assert result.actual_top_id is None


def test_rederive_agrees_when_the_winner_is_the_same_on_trusted_signals_alone():
    """Two ordinary items, no review/brand shenanigans -- the actual top pick
    and the trusted-only re-derivation should simply agree."""
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=3000, wants_durability=True,
                         occasion="anniversary")
    strong = Product("t-strong", "Strong Fit", "jewellery", 1000, "merchant-a", False,
                      occasion_tags=["anniversary"], durability="high")
    weak = Product("t-weak", "Weak Fit", "jewellery", 2900, "merchant-a", False,
                    occasion_tags=[], durability="low")
    ranked = _ranked([strong, weak], slots)
    result = rederive(ranked, slots)
    assert result.diverged is False
    assert result.actual_top_id == "t-strong"
    assert result.clean_top_id == "t-strong"


def test_rederive_flags_divergence_when_the_actual_winner_only_leads_on_gameable_signals():
    """Regression/feature guard for the Counterfactual Decision Checker
    (proposal §5's CDC): craft a shortlist where a mediocre-fit item only
    edges out a clearly-better-fit item once review score and brand tier are
    counted -- the exact shape of a poisoned/spoofed listing. CDC must catch
    that the shortlist's #1 spot doesn't survive a trusted-only re-derivation."""
    slots = IntentSlots(raw_text="", category="electronics", budget_max=10000, wants_durability=True,
                         occasion="gift")
    # Wins decisively on trusted signals (occasion match, price fit), with no
    # review/brand boost at all.
    genuine = Product("t-998", "Genuinely Good Fit", "electronics", 6000, "merchant-a", False,
                       occasion_tags=["gift"], durability="low", review_score=3.0, brand_tier="standard")
    # Weaker on trusted signals (no occasion match), but maxed out on the two
    # gameable ones -- a spoofed 5-star review and a self-declared "premium"
    # brand tag -- enough to overtake the genuine fit on total score.
    poisoned = Product("t-999", "Poisoned Listing", "electronics", 7000, "merchant-a", False,
                        occasion_tags=[], durability="medium", review_score=5.0, brand_tier="premium")
    ranked = _ranked([genuine, poisoned], slots)
    # Sanity check the fixture actually reproduces the scenario under test:
    # the poisoned listing must be what the (unfixed) ranking surfaces first.
    assert ranked[0].product.id == "t-999"

    result = rederive(ranked, slots)
    assert result.diverged is True
    assert result.actual_top_id == "t-999"
    assert result.clean_top_id == "t-998"
    assert result.gameable_signal_share > 0
    assert "t-998" in result.note or "Genuinely Good Fit" in result.note


def test_rederive_result_serializes_to_a_dict():
    slots = IntentSlots(raw_text="", category="care")
    p = Product("t-1", "Test", "care", 100, "merchant-a", False)
    ranked = _ranked([p], slots)
    result = rederive(ranked, slots)
    d = result.as_dict()
    assert set(d.keys()) == {
        "actual_top_id", "clean_top_id", "diverged", "divergence_score", "gameable_signal_share", "note",
    }
