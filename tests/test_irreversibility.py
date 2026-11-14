from app.catalog import get_by_id
from app.irreversibility import classify


def test_low_value_diy_item_is_low_irreversibility():
    product = get_by_id("d-303")  # craft glue, ₹89
    result = classify(product)
    assert result["level"] == "LOW"


def test_permanent_jewellery_item_is_flagged_high_or_critical():
    product = get_by_id("j-104")  # 18K gold ring, "permanent" tag, new-ish price
    result = classify(product)
    assert result["level"] in {"HIGH", "CRITICAL"}
    assert any("permanent" in f.lower() or "personalized" in f.lower() for f in result["factors"])


def test_high_value_new_merchant_item_is_critical():
    product = get_by_id("h-502")  # ₹42,000, new merchant, luxury
    result = classify(product)
    assert result["level"] == "CRITICAL"
    assert any("new" in f.lower() and "merchant" in f.lower() for f in result["factors"])


def test_score_is_always_bounded_0_to_100():
    for pid in ["d-303", "j-104", "h-502", "c-401", "e-201"]:
        result = classify(get_by_id(pid))
        assert 0 <= result["score"] <= 100
