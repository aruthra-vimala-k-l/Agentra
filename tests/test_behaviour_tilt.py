from app import behaviour_experiment as be, catalog, price_tilt_study as pts, retrieval


def test_controls_bracket_ranker():
    prods = catalog.CATALOGUE
    cheap = pts.summarise(pts.run(pts.CONTROLS["cheapest"], prods, 100, 5, 0, "none")[0])
    dear = pts.summarise(pts.run(pts.CONTROLS["dearest"], prods, 100, 5, 0, "none")[0])
    first = pts.summarise(pts.run(pts.CONTROLS["always_first"], prods, 200, 5, 0, "none")[0])
    assert cheap["price_percentile_mean"][0] < 0.05 and dear["price_percentile_mean"][0] > 0.95
    assert 0.4 < first["price_percentile_mean"][0] < 0.6


def test_price_derived_tier_guard_makes_ranker_price_neutral_without_budget():
    prods = catalog.CATALOGUE
    if catalog.SOURCE == "mock":
        return                                               # guard only applies to real (price-derived) tiers
    on = pts.summarise(sum((pts.run(pts._mk(), prods, 300, 5, s, "none")[0] for s in (0, 1)), []))
    retrieval.GUARD_PRICE_DERIVED_TIER = False
    try:
        off = pts.summarise(sum((pts.run(pts._mk(), prods, 300, 5, s, "none")[0] for s in (0, 1)), []))
    finally:
        retrieval.GUARD_PRICE_DERIVED_TIER = True
    assert off["price_percentile_mean"][0] > 0.7             # the original defect
    assert 0.4 < on["price_percentile_mean"][0] < 0.62       # fixed: no hidden "prefer dearer" rule


def test_brand_tier_is_price_proxy_in_real_catalogue():
    if catalog.SOURCE == "mock":
        return
    conf = pts.confounding()
    assert all(v["spearman_price_vs_premium_brand"] > 0.5 for v in conf.values())


def test_comparison_shape():
    r = be.tilt_comparison("agentra", n=60)
    assert r["status"] == "ok" and len(r["agent_no_budget"]) == 3
