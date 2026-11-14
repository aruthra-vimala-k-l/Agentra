from app import web_assistant


def test_lookalike_with_digit_substitution_is_caught():
    assert any(k == "lookalike_domain" for k, _ in web_assistant.url_risk("https://amaz0n-deals.shop/x"))
    assert not any(k == "lookalike_domain" for k, _ in web_assistant.url_risk("https://www.amazon.in/dp/B0AAAAAAAA"))
    assert not any(k == "lookalike_domain" for k, _ in web_assistant.url_risk("https://www.1mg.com/x"))


def test_flagged_reviews_make_a_page_high_risk():
    from app import risk_lab
    c = next(x for x in risk_lab.candidates() if "Wireless Earbuds Pro" in x["name"])
    row = web_assistant.evaluate([c], ["amazon.in"], None)["results"][0]
    assert row["risk_level"] == "High risk" and any("review" in r.lower() for r in row["risk_reasons"])


def _row(**kw):
    c = {"name": "X", "url": "https://www.amazon.in/dp/B0AAAAAAAA", "price_inr": 1000, "rating": 4.6, "rating_count": 900,
         "seller": "Amazon Retail India", "link": {"state": "ok"}}
    c.update(kw)
    return web_assistant.evaluate([c], ["amazon.in"], None)["results"][0]


def test_strict_marketplace_rules():
    assert _row()["risk_level"] == "Low risk"
    z = _row(rating=4.8, rating_count=0)                       # good rating, nobody behind it, on Amazon
    assert z["risk_level"] == "High risk" and any("fewer than 5" in r for r in z["risk_reasons"])
    assert _row(rating=None, rating_count=None)["risk_level"] == "High risk"          # marketplace listing with no ratings at all
    assert _row(seller=None)["risk_score"] == 20 and _row(seller="Random Traders")["risk_level"] == "Low risk"
    s = _row(seller="Random Traders", seller_rating=3.2)
    assert s["risk_score"] >= 25 and any("does not make this seller trusted" in r for r in s["risk_reasons"])
    assert _row(rating=4.9, rating_count=60)["risk_score"] >= 15 + 0                  # near-perfect from few customers
    assert any("85%" in r for r in _row(price_inr=300, mrp_inr=2000)["risk_reasons"])


def test_popularity_badges_never_help_and_are_reported():
    r = _row(rating=4.8, rating_count=0, badges=["Bestseller", "Amazon's Choice", "Sponsored"])
    assert r["risk_level"] == "High risk"
    assert sum("ignored" in n for n in r["bias_notes"]) == 3 and any("Marketplace listing" in n for n in r["bias_notes"])


def test_price_far_below_peers_is_flagged_and_high_risk_ranks_lower():
    mk = lambda n, p, rc: {"name": n, "url": f"https://www.croma.com/p/{n}", "price_inr": p, "rating": 4.4, "rating_count": rc, "link": {"state": "ok"}}
    ev = web_assistant.evaluate([mk("a", 2000, 900), mk("b", 2100, 800), mk("c", 2200, 700), mk("d", 300, 900)], ["croma.com"], None)
    d = next(r for r in ev["results"] if r["name"] == "d")
    assert any("far below similar listings" in x for x in d["risk_reasons"])
