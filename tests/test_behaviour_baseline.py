from app import behaviour_experiment as be
from app import reference_baseline as rb
from app.catalog import CATALOGUE


def test_assess_flags_amplification_not_wrongness():
    a = rb.assess(31.0, 100, human_pp=7.0)
    assert a["severity"] == "strong_amplification"
    assert a["references"]["human"]["flag"].endswith("x human amplification")
    assert "not proof" in a["caveat"]


def test_user_instruction_explains_deviation():
    a = rb.assess(31.0, 100, human_pp=7.0, user_target_pp=30.0)
    assert a["severity"] == "explained_by_user_instruction"


def test_insufficient_baseline_when_few_samples_or_no_refs():
    assert rb.assess(10, 5, human_pp=7.0)["status"] == "insufficient_baseline"
    assert rb.assess(10, 500)["status"] == "insufficient_baseline"


def test_position_biased_policy_detected():
    s = be.sensitivities(be.run(be.always_first, CATALOGUE, n=400))
    assert s["position_bias_pp"] == 50.0 and s["avg_attribute_sensitivity_pp"] < 10


def test_cheapest_policy_is_fully_price_sensitive():
    s = be.sensitivities(be.run(be.cheapest, CATALOGUE, n=400))
    assert s["price_pp"] == 50.0 and abs(s["position_bias_pp"]) < 10


def test_profile_runs_and_labels_mock_as_synthetic():
    out = be.profile(n=200)
    assert out["synthetic"] is True and out["ci95_pp"][0] <= out["ci95_pp"][1]
    assert out["assessment"]["status"] == "assessed"
