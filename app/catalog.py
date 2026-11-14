"""Catalogue facade.

``CATALOG_SOURCE`` selects the backing store:

* ``real`` (default) -- real product records from ``data/catalog/products.jsonl``
  (see app/real_catalog.py, scripts/build_catalog.py). No synthetic fallback.
  Set ``CATALOG_INCLUDE_TESTBED=1`` to additionally inject the MartVerse bias-
  testbed listings (labelled source="martverse_testbed") as controlled
  attack/bias fixtures -- clearly separate from the real records.
* ``mock``           -- the hand-authored catalogue + MartVerse storefront
  (app/mock_catalog.py). Used by the unit tests (tests/conftest.py sets it).

Everything else imports ``Product``, ``CATALOGUE``, ``get_by_id``, ``search``
from here and is source-agnostic.
"""
from __future__ import annotations

import os

from app.product import Product

SOURCE = os.environ.get("CATALOG_SOURCE", "real").strip().lower()

if SOURCE == "mock":
    from app.mock_catalog import CATALOGUE
else:
    from app import real_catalog
    CATALOGUE = real_catalog.load()
    if os.environ.get("CATALOG_INCLUDE_TESTBED") == "1":
        from app.storefront import ingest_from_storefront
        CATALOGUE.extend(ingest_from_storefront())

_BY_ID = {p.id: p for p in CATALOGUE}


def get_by_id(product_id: str) -> Product | None:
    return _BY_ID.get(product_id)


def categories() -> list[str]:
    return sorted({p.category for p in CATALOGUE})


def search(category: str | None = None, max_price: float | None = None,
           occasion: str | None = None) -> list[Product]:
    results = [p for p in CATALOGUE if p.in_stock]
    if category:
        results = [p for p in results if p.category == category]
    if max_price is not None:
        results = [p for p in results if p.price <= max_price]
    if occasion:
        results = [p for p in results if occasion in p.occasion_tags]
    return results


def status() -> dict:
    """Provenance summary for /system/data-status."""
    sources: dict[str, int] = {}
    for p in CATALOGUE:
        sources[p.source] = sources.get(p.source, 0) + 1
    info = {
        "catalog_source": SOURCE,
        "product_count": len(CATALOGUE),
        "records_by_source": sources,
        "categories": categories(),
        "products_with_reviews": sum(1 for p in CATALOGUE if p.reviews),
        "synthetic": SOURCE == "mock" or any(
            p.source in ("mock", "martverse_testbed") for p in CATALOGUE),
    }
    if SOURCE != "mock":
        info["manifest"] = real_catalog.manifest()
        if not any(p.source not in ("mock", "martverse_testbed") for p in CATALOGUE):
            info["warning"] = ("Real catalogue is empty. Run scripts/build_catalog.py "
                               "(see DATA.md) -- there is intentionally no mock fallback.")
    return info
