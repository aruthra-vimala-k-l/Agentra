from app.attribution import attribute, shapley_values
from app.catalog import get_by_id
from app.intent import IntentSlots


def test_shapley_values_satisfy_the_efficiency_property():
    """The defining correctness property of Shapley values: the contributions
    must sum exactly to the total payoff of switching every feature on
    (minus the empty-coalition payoff, which is 0 here). If this doesn't
    hold, the decomposition isn't actually a valid Shapley computation."""
    contributions = {"a": 2.0, "b": 1.0, "c": -0.5}

    def value_fn(active):
        total = sum(v for k, v in contributions.items() if k in active)
        if "a" in active and "b" in active:
            total += 1.0  # nonlinear synergy term
        return total

    features = list(contributions.keys())
    shap = shapley_values(features, value_fn)
    full_value = value_fn(frozenset(features))
    empty_value = value_fn(frozenset())
    assert abs(sum(shap.values()) - (full_value - empty_value)) < 1e-9


def test_shapley_gives_zero_weight_to_a_feature_the_value_function_ignores():
    contributions = {"a": 2.0, "b": 1.0, "ignored": 5.0}

    def value_fn(active):
        return sum(v for k, v in contributions.items() if k in active and k != "ignored")

    shap = shapley_values(list(contributions.keys()), value_fn)
    assert shap["ignored"] == 0.0


def test_attribute_on_a_real_product_sums_to_its_ranking_score():
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=7000, wants_durability=True, occasion="anniversary")
    product = get_by_id("j-104")
    result = attribute(product, slots)
    assert abs(sum(result.shapley_values.values()) - result.total_score) < 1e-6


def test_promoted_items_nudge_signal_earns_zero_shapley_weight():
    """h-502 is deliberately catalogued as promoted -- proves the ranking
    function's design rule (promotion never earns weight) holds under the
    actual Shapley computation, not just by inspection of the raw score."""
    slots = IntentSlots(raw_text="", category="luxury", budget_max=50000)
    product = get_by_id("h-502")
    result = attribute(product, slots)
    assert result.shapley_values["nudge_exposure"] == 0.0


def test_top_signals_are_sorted_by_magnitude():
    slots = IntentSlots(raw_text="", category="jewellery", budget_max=7000, wants_durability=True, occasion="anniversary")
    product = get_by_id("j-104")
    result = attribute(product, slots)
    magnitudes = [abs(v) for _, v in result.top_signals]
    assert magnitudes == sorted(magnitudes, reverse=True)
