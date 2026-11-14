import json

from app import catalog, real_catalog
from app.catalog_build import build, derive_durability, normalise_amazon, parse_price


def _raw(**kw):
    base = {"parent_asin": "B0TEST1", "title": "Sterling Silver Ring Gift for Anniversary",
            "price": "$40.00", "average_rating": 4.5, "rating_number": 120,
            "store": "RingCo", "features": ["925 sterling silver band"], "details": {"Material": "Sterling Silver"}}
    base.update(kw)
    return base


def test_parse_price():
    assert parse_price("$1,299.50") == 1299.5
    assert parse_price(None) is None and parse_price("n/a") is None and parse_price(0) is None


def test_normalise_converts_and_derives():
    n = normalise_amazon(_raw(), "Clothing_Shoes_and_Jewelry", 80.0)
    assert n["price"] == 3200.0 and n["category"] == "jewellery"
    assert n["durability"] == "high" and "anniversary" in n["occasion_tags"] and "gift" in n["occasion_tags"]
    assert n["delivery_days"] is None and n["promoted"] is False
    assert n["source"] == "amazon_reviews_2023" and n["source_url"].endswith("B0TEST1")


def test_unusable_records_dropped_not_invented():
    assert normalise_amazon(_raw(price=None), "Electronics", 80) is None
    assert normalise_amazon(_raw(rating_number=2), "Electronics", 80) is None
    assert normalise_amazon(_raw(title="Running Shoes"), "Clothing_Shoes_and_Jewelry", 80) is None


def test_unknown_durability_stays_none():
    assert derive_durability("a generic thing") is None


def test_build_ids_and_tiers():
    raws = [_raw(parent_asin=f"B{i}", price=f"${10 + i * 10}") for i in range(8)]
    rows = build({"Clothing_Shoes_and_Jewelry": raws}, 80.0, 5)
    assert len(rows) == 5 and rows[0]["id"] == "j-10001"
    assert {r["brand_tier"] for r in rows} == {"premium", "standard"}


def test_real_loader_and_no_silent_mock_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOG_PATH", str(tmp_path / "missing.jsonl"))
    assert real_catalog.load() == []
    f = tmp_path / "products.jsonl"
    rows = build({"Electronics": [_raw(title="Wireless earbuds", parent_asin="B9")]}, 80.0, 3)
    f.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setenv("CATALOG_PATH", str(f))
    loaded = real_catalog.load()
    assert len(loaded) == 1 and loaded[0].source == "amazon_reviews_2023"


def test_status_flags_mock_as_synthetic():
    assert catalog.status()["synthetic"] is True


def test_review_normalisation_and_attachment_feeds_authenticity_check():
    from app.catalog_build import collect_reviews, count_reviews_per_asin, normalise_review
    from app.product import Product
    from app.review_authenticity import check
    raw = [{"parent_asin": "B1", "user_id": f"u{i}", "rating": 5.0, "timestamp": 1_600_000_000_000 + i * 86_400_000 * 30,
            "title": "Nice", "text": f"review number {i}", "verified_purchase": True} for i in range(12)]
    assert count_reviews_per_asin(raw) == {"B1": 12}
    got = collect_reviews(raw, {"B1"})["B1"]
    assert got[0]["timestamp"].startswith("2020-09") and got[0]["reviewer"] == "u0"
    assert normalise_review({"user_id": None, "rating": 5, "timestamp": 1}) is None
    p = Product("e-10001", "x", "electronics", 100, "m", False, reviews=got)
    res = check(p)           # real-shaped reviews run through the existing ADM without error
    assert res.review_count == 12


def test_build_attaches_reviews_by_asin():
    rev = {"B7": [{"reviewer": "u", "rating": 5.0, "timestamp": "2021-01-01T00:00:00", "text": "t"}]}
    rows = build({"Electronics": [_raw(title="Wireless earbuds", parent_asin="B7")]}, 80.0, 3, rev)
    assert rows[0]["reviews"] == rev["B7"]
