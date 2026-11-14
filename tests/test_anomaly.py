import os
from dataclasses import replace

import pytest

from app import anomaly
from app.product import Product

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "catalog", "products.jsonl")


def _p(i, price, cat="electronics", rating=4.3, reviews=500):
    return Product(id=f"x{i}", name=f"item {i}", category=cat, price=price, merchant="m", is_new_merchant=False,
                   review_score=rating, review_count=reviews, source="amazon_reviews_2023")


def _ref():
    # deterministic spread of normal-priced products across two categories
    prods = [_p(i, 800 + (i * 37) % 900, "electronics", 3.8 + (i % 10) / 10, 40 + i * 7) for i in range(80)]
    prods += [_p(100 + i, 300 + (i * 23) % 400, "care", 3.9 + (i % 8) / 10, 30 + i * 5) for i in range(60)]
    return anomaly.Reference.fit(prods)


def test_normal_product_is_normal():
    r = anomaly.assess(_p(999, 1200), _ref())
    assert r["level"] == "NORMAL" and r["score"] < 30


def test_too_cheap_is_flagged_high():
    r = anomaly.assess(_p(999, 12), _ref())
    assert r["level"] == "HIGH" and any("BELOW" in x for x in r["reasons"]) and r["price_robust_z"] < -3.5


def test_wildly_expensive_is_flagged():
    r = anomaly.assess(_p(999, 90000), _ref())
    assert r["level"] == "HIGH" and r["price_robust_z"] > 3.5


def test_perfect_rating_on_two_reviews_is_flagged():
    r = anomaly.assess(_p(999, 1200, rating=5.0, reviews=2), _ref())
    assert any("only 2 review" in x for x in r["reasons"]) and r["score"] >= 30


def test_isolation_forest_is_deterministic():
    a = anomaly.assess(_p(999, 5000, rating=2.0, reviews=3), _ref())
    b = anomaly.assess(_p(999, 5000, rating=2.0, reviews=3), _ref())
    assert a == b


def test_too_little_data_says_so():
    r = anomaly.assess(_p(1, 100), anomaly.Reference.fit([_p(2, 100)]))
    assert r["level"] == "NORMAL" and "Not enough" in r["reasons"][0]


@pytest.mark.skipif(not os.path.exists(DATA), reason="real catalogue not built on this machine")
def test_on_the_real_catalogue_few_products_are_flagged_high():
    """Sanity check on real data: the detector must not cry wolf on ordinary listings."""
    from app.real_catalog import load
    prods = load()
    ref = anomaly.Reference.fit(prods)
    high = sum(1 for p in prods if anomaly.assess(p, ref)["level"] == "HIGH")
    assert high / len(prods) < 0.10, f"{high}/{len(prods)} flagged HIGH"
