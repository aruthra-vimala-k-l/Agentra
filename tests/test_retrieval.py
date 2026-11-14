from app.catalog import CATALOGUE
from app.intent import IntentSlots
from app.retrieval import retrieve_and_rank


def test_retrieval_respects_budget_and_category():
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=2500, wants_durability=False)
    ranked = retrieve_and_rank(slots)
    assert ranked, "expected at least one match under budget"
    for r in ranked:
        assert r.product.category == "jewellery"
        assert r.product.price <= 2500


def test_durability_preference_ranks_high_durability_items_first():
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=7000, wants_durability=True)
    ranked = retrieve_and_rank(slots)
    assert ranked[0].product.durability == "high"


def test_justification_never_claims_an_attribute_the_product_lacks():
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=7000, wants_durability=True)
    ranked = retrieve_and_rank(slots)
    for r in ranked:
        if r.product.durability != "high":
            assert "high-durability" not in r.justification.lower()


def test_all_catalogue_products_are_reachable_by_category_filter():
    categories = {p.category for p in CATALOGUE}
    for cat in categories:
        slots = IntentSlots(raw_text="", category=cat)
        ranked = retrieve_and_rank(slots, top_k=50)
        assert len(ranked) > 0
