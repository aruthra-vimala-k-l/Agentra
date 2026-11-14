"""CDC v2: semantic divergence weighted by irreversibility (see app/counterfactual.py header for the formula)."""
from app.counterfactual import W_BLOCK, W_ESCALATE, assess_choice, attribute_distance, semantic_distance
from app.intent import IntentSlots
from app.product import Product
from app.retrieval import RankedProduct, signal_contributions, total_score

SLOTS = IntentSlots(raw_text="", category="electronics", budget_max=10000, wants_durability=True, occasion="gift")


def _ranked(products):
    r = [RankedProduct(p, total_score(signal_contributions(p, SLOTS)), "t") for p in products]
    return sorted(r, key=lambda x: -x.score)


def _genuine(price=6000):
    return Product("t-998", "Compact Bluetooth Speaker gift box", "electronics", price, "m", False, occasion_tags=["gift"],
                   durability="high", review_score=3.0, description="Compact bluetooth speaker, gift box, 12h battery.")


def _poisoned(price=7000):
    return Product("t-999", "Premium Audio Device limited offer", "electronics", price, "m", False, occasion_tags=[],
                   durability="low", review_score=5.0, brand_tier="premium",
                   description="Premium audio device, trusted brand, limited offer, fast shipping.")


def test_choosing_the_clean_winner_has_zero_divergence():
    r = _ranked([_genuine(), _poisoned()])
    a = assess_choice(r, SLOTS, "t-998", 70, "CRITICAL")
    assert a.weighted_divergence == 0 and a.level == "NONE"


def test_poisoned_choice_diverges_and_irreversibility_raises_the_weight():
    r = _ranked([_genuine(), _poisoned()])
    low = assess_choice(r, SLOTS, "t-999", 10, "LOW")
    high = assess_choice(r, SLOTS, "t-999", 45, "HIGH")
    crit = assess_choice(r, SLOTS, "t-999", 70, "CRITICAL")
    assert low.divergence == high.divergence == crit.divergence > 0      # same divergence ...
    assert low.weighted_divergence < high.weighted_divergence < crit.weighted_divergence   # ... heavier when irreversible
    assert high.level in ("ESCALATE", "BLOCK") and crit.level in ("ESCALATE", "BLOCK")


def test_block_level_is_reachable_for_a_strong_manipulation_on_a_hard_to_reverse_purchase():
    genuine = Product("t-998", "Waterproof Hiking Boots", "electronics", 1500, "m", False, occasion_tags=["gift"], durability="high",
                      review_score=2.5, description="Waterproof leather hiking boots with ankle support.")
    poisoned = Product("t-999", "Quantum Crypto Mining Rig", "electronics", 9500, "m", False, occasion_tags=[], durability="low",
                       review_score=5.0, brand_tier="premium", description="Industrial hashing hardware, enterprise power supply.")
    a = assess_choice(_ranked([genuine, poisoned]), SLOTS, "t-999", 70, "CRITICAL")
    assert a.weighted_divergence >= W_BLOCK and a.level == "BLOCK"
    # the same divergence on an easily reversible purchase is escalated, never blocked
    assert assess_choice(_ranked([genuine, poisoned]), SLOTS, "t-999", 10, "LOW").level != "BLOCK"


def test_near_identical_listings_are_immaterial():
    a = Product("a", "Wireless Earbuds Black", "electronics", 1500, "m", False, durability="high", review_score=4.9,
                brand_tier="premium", occasion_tags=["gift"], description="Wireless earbuds black bluetooth charging case.")
    b = Product("b", "Wireless Earbuds Black", "electronics", 1500, "m", False, durability="high", review_score=3.0,
                occasion_tags=["gift"], description="Wireless earbuds black bluetooth charging case.")
    assert semantic_distance(a, b) < 0.05 and attribute_distance(a, b) == 0
    assert assess_choice(_ranked([a, b]), SLOTS, "a", 70, "CRITICAL").level == "NONE"


def test_item_outside_the_shortlist_is_handled():
    r = _ranked([_genuine(), _poisoned()])
    assert assess_choice(r, SLOTS, "nope", 50, "HIGH").level == "NONE"


def test_thresholds_are_ordered():
    assert 0 < W_ESCALATE < W_BLOCK <= 2
