"""Mock external storefront ingestion -- the Catalogue Ingestion Layer the
conversational-commerce brief describes ("normalizes arbitrary schemas"
from a real storefront into Agentra's format), reading from
bias_testbed_storefront.html (MartVerse), a standalone, self-contained
HTML/CSS/JS page built specifically as a bias-testbed: 25 products across 5
categories (electronics, clothing, jewelry, DIY & home, beauty), 5 per
category, each following one of five deliberate bias patterns:

  A: bestseller badge, low rating, genuine negative reviews  -- actually bad
  B: bestseller badge, high rating, FABRICATED reviews       -- actually bad
  C: bestseller badge, high rating, genuine positive reviews -- actually good
  D: no badge, a flawless 5.0 rating, FABRICATED reviews     -- actually bad
  E: no badge, a modest rating, genuine positive reviews     -- actually good, but buried

This module does NOT keep a second, hand-maintained copy of that catalogue.
It parses the exact JSON data island MartVerse's own page renders from
(`<script type="application/json" id="martverse-catalog-data">`), the same
way a real ingestion layer would parse a real site's embedded structured
data rather than re-typing its contents -- so the storefront page and what
the shopping agent "sees" can never drift out of sync with each other. If
you edit a product in the HTML file, this module picks up the change
automatically the next time it's read.

Field mapping from MartVerse's schema to Agentra's `app.catalog.Product`:
  - MartVerse's own id (already Agentra-ID-shaped, e.g. "e-601") is used
    directly, so a shopper can say "buy e-602" in chat and mean the exact
    item shown on the storefront page.
  - MartVerse's `cat` (its own nav category: electronics/clothing/jewelry/
    diy/beauty) is carried separately from `agentra_category`, which is
    what search()/retrieval actually filter on -- "jewelry" (American
    spelling, MartVerse's own nav label) maps to "jewellery" (Agentra's
    existing category, shared with the hand-authored catalogue's j-1xx
    items) rather than becoming a second, redundant category.
  - `reviews` is carried through verbatim -- this is the data
    app/review_authenticity.py actually analyzes; `rating`/`reviewCount`
    become Product.review_score/... implicitly via the average already
    baked into MartVerse's own `rating` field (kept as-is rather than
    re-averaged, since MartVerse's displayed rating is itself part of what
    a shopper -- and the agent -- sees and can be misled by).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.product import Product

# bias_testbed_storefront.html lives at the repo root, one level up from
# this file's `app/` directory -- not under `static/`, since it's a
# standalone artifact (openable directly in a browser, no server required)
# that also happens to double as this app's mock external storefront.
_STOREFRONT_HTML_PATH = Path(__file__).resolve().parent.parent / "bias_testbed_storefront.html"

_CATALOG_ISLAND_RE = re.compile(
    r'<script type="application/json" id="martverse-catalog-data">(.*?)</script>',
    re.DOTALL,
)


def _load_raw_listings() -> list[dict]:
    html = _STOREFRONT_HTML_PATH.read_text(encoding="utf-8")
    match = _CATALOG_ISLAND_RE.search(html)
    if not match:
        # Fail loud, not silent: an ingestion layer that quietly returns no
        # products because the source page's markup changed is exactly the
        # kind of bug that looks like "the agent found nothing" instead of
        # "the scraper broke."
        raise RuntimeError(
            f"Could not find MartVerse's #martverse-catalog-data JSON island in {_STOREFRONT_HTML_PATH}. "
            "Has the storefront page's markup changed?"
        )
    return json.loads(match.group(1))


def _normalize_reviews(raw_reviews: list[dict]) -> list[dict]:
    """MartVerse's own review shape (the page's JS reads {name, date, rating,
    text}, since that's what its own rendering code was written against) is
    not the same shape app/review_authenticity.py's contract expects
    ({reviewer, timestamp, rating, text}, matching the rest of Agentra's
    canonical Product.reviews usage). Translating field names is exactly the
    kind of normalization a real ingestion layer does for an arbitrary
    external schema -- review_authenticity.py stays agnostic of any one
    source site's naming."""
    return [
        {"reviewer": r["name"], "timestamp": r["date"], "rating": r["rating"], "text": r["text"]}
        for r in raw_reviews
    ]


def ingest_from_storefront() -> list[Product]:
    """Normalize MartVerse's raw listings into Products -- the contract a
    real ingestion layer fulfils for an actual external site's product and
    review pages."""
    products = []
    for listing in _load_raw_listings():
        products.append(Product(
            id=listing["id"],
            name=listing["name"],
            category=listing["agentra_category"],
            price=listing["price"],
            merchant=listing["merchant"],
            is_new_merchant=listing["is_new_merchant"],
            material=None,
            occasion_tags=listing["occasion_tags"],
            durability=listing["durability"],
            description=listing["desc"],
            delivery_days=listing["delivery_days"],
            review_score=listing["rating"],
            brand_tier=listing["brand_tier"],
            promoted=listing["bestseller"],
            reviews=_normalize_reviews(listing["reviews"]),
            source="martverse_testbed",
        ))
    return products
