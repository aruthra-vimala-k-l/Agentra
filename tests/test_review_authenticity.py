from app.catalog import Product, get_by_id
from app.review_authenticity import check


def test_too_few_reviews_is_not_assessed():
    p = Product("z-001", "Sparse", "electronics", 100, "merchant-a", False,
                reviews=[{"reviewer": "a", "rating": 5, "timestamp": "2026-01-01T00:00:00", "text": "good"}])
    result = check(p)
    assert result.suspicious is False
    assert result.reviewer_diversity is None


def test_hand_authored_catalogue_items_have_no_review_data_and_are_not_flagged():
    """j-104 (and every other hand-authored item) has no individual review
    list -- review_score alone is not enough data to assess, and this must
    not be silently treated as suspicious just because it's empty."""
    product = get_by_id("j-104")
    result = check(product)
    assert result.suspicious is False


def test_fabricated_burst_review_pattern_is_flagged():
    """e-602 (see bias_testbed_storefront.html / app/storefront.py) is
    deliberately built with all three reviews landing within a 3-day, same-
    time-of-night window -- the burst-timing tell this checker exists to
    catch, on a listing carrying a fabricated 4.9-star average."""
    product = get_by_id("e-602")
    result = check(product)
    assert result.suspicious is True
    assert result.burst_detected is True
    assert len(result.reasons) >= 1


def test_organic_review_pattern_is_not_flagged():
    """e-603 is the clean comparison case: reviews spread across months,
    distinct reviewers, varied wording -- a genuinely good, genuinely
    reviewed item."""
    product = get_by_id("e-603")
    result = check(product)
    assert result.suspicious is False
    assert result.burst_detected is False
    assert result.duplicate_text_detected is False


def test_bestseller_badge_with_poor_but_organic_reviews_is_not_flagged_by_this_checker():
    """e-601 is the 'bad reviews behind a bestseller badge' case -- that's a
    nudge_exposure / Bias Detection Engine concern (is the badge earning
    ranking weight it shouldn't), not a review-authenticity concern, since
    its reviews are genuinely spread out and from distinct reviewers."""
    product = get_by_id("e-601")
    result = check(product)
    assert result.suspicious is False


def test_low_reviewer_diversity_is_flagged_even_without_a_burst():
    """Unit-level coverage for the diversity check on its own: MartVerse's
    demo fixtures happen to always use 3 distinct names per listing (the
    reused-names tell in that dataset is a CROSS-listing pattern, not a
    within-listing one -- see app/storefront.py's docstring), so this
    exercises the same-reviewer-posting-repeatedly case directly."""
    p = Product("z-002", "Low Diversity Widget", "electronics", 500, "merchant-a", False,
                reviews=[
                    {"reviewer": "same_person", "rating": 5, "timestamp": "2026-01-01T00:00:00", "text": "Loved it, great buy."},
                    {"reviewer": "same_person", "rating": 5, "timestamp": "2026-03-15T00:00:00", "text": "Still working well months later."},
                    {"reviewer": "same_person", "rating": 4, "timestamp": "2026-05-20T00:00:00", "text": "Does what it says."},
                ])
    result = check(p)
    assert result.suspicious is True
    assert result.reviewer_diversity <= 0.5


def test_duplicate_review_text_is_flagged():
    p = Product("z-003", "Templated Reviews Widget", "electronics", 500, "merchant-a", False,
                reviews=[
                    {"reviewer": "a1", "rating": 5, "timestamp": "2026-01-01T00:00:00", "text": "Amazing product!"},
                    {"reviewer": "b2", "rating": 5, "timestamp": "2026-04-10T00:00:00", "text": "Amazing product!"},
                    {"reviewer": "c3", "rating": 5, "timestamp": "2026-07-22T00:00:00", "text": "Great overall experience."},
                ])
    result = check(p)
    assert result.suspicious is True
    assert result.duplicate_text_detected is True


def test_result_serializes_to_a_dict():
    product = get_by_id("e-602")
    result = check(product)
    d = result.as_dict()
    assert set(d.keys()) == {
        "product_id", "review_count", "suspicious", "burst_detected",
        "reviewer_diversity", "duplicate_text_detected", "reasons",
    }
