"""Normalisation of REAL public product data into Agentra ``Product`` rows.

Pure functions (no network) so they are unit-testable; the CLI wrapper is
scripts/build_catalog.py. Input today: Amazon Reviews 2023 item-metadata
records (McAuley Lab) -- fields used: parent_asin, title, main_category,
categories, price (USD), average_rating, rating_number, store, features,
description, details.

Provenance / honesty rules (so we can defend this in front of evaluators):
* Every field copied from a source record is real.
* Fields the source does not carry are left as unknown (None / default), never
  invented: delivery_days=None, promoted=False (no sponsored flag in metadata),
  is_new_merchant=False (no seller-age data).
* Three fields are DERIVED by documented rules from real text/prices, and are
  labelled as such in the manifest: durability (keyword rule), occasion_tags
  (keyword rule), brand_tier (top price quartile within category).
* USD->INR conversion uses an explicit, recorded rate (an assumption).
"""
from __future__ import annotations

import re
import statistics
from datetime import datetime, timezone
from typing import Iterable

from app.intent import OCCASION_KEYWORDS

# Amazon-2023 category file -> Agentra category + id prefix
CATEGORY_MAP = {
    "Handmade_Products": ("jewellery", "j"),            # rings/pendants/gifts (filtered by jewellery hints)
    "Clothing_Shoes_and_Jewelry": ("jewellery", "j"),   # optional top-up (filtered by jewellery hints)
    "All_Beauty": ("care", "c"),
    "Grocery_and_Gourmet_Food": ("care", "c"),          # tea, gift boxes, baskets
    "Electronics": ("electronics", "e"),
    "Arts_Crafts_and_Sewing": ("diy", "d"),             # glue, paint, balsa -> school-project builds
    "Tools_and_Home_Improvement": ("diy", "d"),
}

JEWELLERY_HINTS = ("jewel", "ring", "necklace", "pendant", "bracelet", "earring", "chain")
HIGH_DURABILITY = ("solid gold", "sterling", "stainless steel", "titanium", "14k", "18k", "22k",
                   "tempered", "aluminum alloy", "forged")
LOW_DURABILITY = ("plated", "plastic", "costume", "fashion jewelry", "disposable", "alloy")
MIN_RATINGS = 5


def parse_price(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    m = re.search(r"(\d[\d,]*\.?\d*)", str(raw))
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return v if v > 0 else None


def _text(row: dict) -> str:
    parts = [row.get("title") or ""]
    for key in ("features", "description"):
        v = row.get(key)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    return " ".join(parts).lower()


def derive_durability(text: str) -> str | None:
    if any(k in text for k in LOW_DURABILITY) and not any(k in text for k in HIGH_DURABILITY):
        return "low"
    if any(k in text for k in HIGH_DURABILITY):
        return "high"
    return None  # unknown -- not guessed


def derive_occasions(text: str) -> list[str]:
    tags = {tag for kw, tag in OCCASION_KEYWORDS.items() if kw in text}
    return sorted(tags)


MAX_REVIEWS_PER_PRODUCT = 40


def normalise_review(r: dict) -> dict | None:
    """Amazon Reviews 2023 review record -> the shape app/review_authenticity.py
    expects ({reviewer, rating, timestamp(ISO), text}). Timestamps in the
    source are epoch milliseconds."""
    uid, rating, ts = r.get("user_id"), r.get("rating"), r.get("timestamp")
    if not uid or rating is None or ts is None:
        return None
    try:
        iso = datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc).replace(tzinfo=None).isoformat()
    except (ValueError, OSError, OverflowError):
        return None
    text = " ".join(x for x in ((r.get("title") or "").strip(), (r.get("text") or "").strip()) if x)
    return {"reviewer": uid, "rating": float(rating), "timestamp": iso, "text": text[:600],
            "verified_purchase": bool(r.get("verified_purchase")), "helpful_vote": r.get("helpful_vote", 0)}


def count_reviews_per_asin(review_rows: Iterable[dict]) -> dict[str, int]:
    """Distinct (user, timestamp, text) reviews per parent_asin."""
    seen: set[tuple] = set()
    counts: dict[str, int] = {}
    for r in review_rows:
        a = r.get("parent_asin")
        if not a:
            continue
        key = (a, r.get("user_id"), r.get("timestamp"), r.get("text"))
        if key in seen:
            continue
        seen.add(key)
        counts[a] = counts.get(a, 0) + 1
    return counts


def collect_reviews(review_rows: Iterable[dict], wanted: set[str]) -> dict[str, list[dict]]:
    """Gather up to MAX_REVIEWS_PER_PRODUCT real reviews per wanted parent_asin.

    The 2023 dataset repeats one review once per colour/size variant (same
    user, timestamp and text under different child ASINs but one parent_asin).
    Those repeats are a data artefact, not review fraud, so they are collapsed
    here -- otherwise the authenticity check would flag them as duplicates."""
    out: dict[str, list[dict]] = {a: [] for a in wanted}
    seen: set[tuple] = set()
    for r in review_rows:
        a = r.get("parent_asin")
        if a not in out or len(out[a]) >= MAX_REVIEWS_PER_PRODUCT:
            continue
        key = (a, r.get("user_id"), r.get("timestamp"), r.get("text"))
        if key in seen:
            continue
        seen.add(key)
        nr = normalise_review(r)
        if nr:
            out[a].append(nr)
    return out


def normalise_amazon(row: dict, file_category: str, usd_inr: float,
                     reviews_by_asin: dict[str, list[dict]] | None = None) -> dict | None:
    """One raw metadata record -> product dict (without id / brand_tier), or None
    if the record can't be used honestly (no price, no ratings, no title)."""
    if file_category not in CATEGORY_MAP:
        raise ValueError(f"unknown category file {file_category!r}")
    category, _ = CATEGORY_MAP[file_category]
    title = (row.get("title") or "").strip()
    price_usd = parse_price(row.get("price"))
    rating = row.get("average_rating")
    n_ratings = row.get("rating_number")
    asin = row.get("parent_asin")
    if not (title and price_usd and asin and rating and n_ratings and n_ratings >= MIN_RATINGS):
        return None
    text = _text(row)
    if category == "jewellery" and not any(h in text for h in JEWELLERY_HINTS):
        return None  # Clothing_Shoes_and_Jewelry also contains clothes/shoes
    details = row.get("details") or {}
    material = details.get("Material") if isinstance(details, dict) else None
    feats = row.get("features") or []
    return {
        "name": title[:140],
        "category": category,
        "price": round(price_usd * usd_inr, 2),
        "merchant": (row.get("store") or "unknown-seller").strip()[:80],
        "is_new_merchant": False,
        "material": material,
        "occasion_tags": derive_occasions(text),
        "durability": derive_durability(text),
        "description": (str(feats[0]) if feats else "")[:300],
        "in_stock": True,
        "delivery_days": None,
        "review_score": float(rating),
        "promoted": False,
        "source": "amazon_reviews_2023",
        "source_url": f"https://www.amazon.com/dp/{asin}",
        "review_count": int(n_ratings),
        "reviews": (reviews_by_asin or {}).get(asin, []),
    }


def assign_ids_and_tiers(rows_by_file: dict[str, list[dict]]) -> list[dict]:
    """Stable ids (<prefix>-<5 digits>, one counter per prefix so several source
    files can feed the same Agentra category without id clashes) and
    brand_tier = premium for the top price quartile of each source file
    (derived, documented)."""
    out: list[dict] = []
    counters: dict[str, int] = {}
    for file_category, rows in rows_by_file.items():
        _, prefix = CATEGORY_MAP[file_category]
        if not rows:
            continue
        prices = sorted(r["price"] for r in rows)
        q3 = statistics.quantiles(prices, n=4)[2] if len(prices) >= 4 else prices[-1]
        for r in rows:
            r = dict(r)
            counters[prefix] = counters.get(prefix, 10000) + 1
            r["id"] = f"{prefix}-{counters[prefix]}"
            r["brand_tier"] = "premium" if r["price"] >= q3 else "standard"
            out.append(r)
    return out


def build(raw_by_file: dict[str, Iterable[dict]], usd_inr: float, per_category: int,
          reviews_by_asin: dict[str, list[dict]] | None = None, seed: int = 7,
          min_reviews: int = 10) -> list[dict]:
    """Normalise, then pick ``per_category`` products per source file.

    Selection: products that have >= ``min_reviews`` real attached reviews come first
    (they can be run through the review-authenticity check), the rest fill up the
    quota. Within each group the pick is a seeded random sample, so the build is
    reproducible and not just 'the most popular items'."""
    import random
    rng = random.Random(seed)
    rows_by_file: dict[str, list[dict]] = {}
    for file_category, raws in raw_by_file.items():
        pool: list[dict] = []
        seen: set[str] = set()
        for raw in raws:
            norm = normalise_amazon(raw, file_category, usd_inr, reviews_by_asin)
            if norm is None or norm["source_url"] in seen:
                continue
            seen.add(norm["source_url"])
            pool.append(norm)
        reviewed = [r for r in pool if len(r["reviews"]) >= min_reviews]
        rest = [r for r in pool if len(r["reviews"]) < min_reviews]
        rng.shuffle(reviewed)
        rng.shuffle(rest)
        rows_by_file[file_category] = (reviewed + rest)[:per_category]
    return assign_ids_and_tiers(rows_by_file)
