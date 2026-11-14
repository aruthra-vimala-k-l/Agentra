"""Hand-authored MOCK catalogue + the MartVerse bias-testbed storefront --
tests and offline demos ONLY (CATALOG_SOURCE=mock).

Not used when CATALOG_SOURCE=real (the default). The planted h-502 "dark
pattern" listing and the MartVerse fabricated-review listings live here
because the unit tests depend on them.
"""
from __future__ import annotations

from app.product import Product


CATALOGUE: list[Product] = [
    # --- Jewellery / gifting ---------------------------------------------
    Product("j-101", "Solid Gold Band, 22K", "jewellery", 6800, "merchant-established", False,
            material="solid gold", occasion_tags=["anniversary", "wedding", "permanent"],
            durability="high", description="22K solid gold band, hallmarked, lifetime warranty.",
            delivery_days=4, review_score=4.4, brand_tier="premium"),
    Product("j-102", "Gold-Plated Couple Band (Pair)", "jewellery", 1499, "merchant-established", False,
            material="gold-plated brass", occasion_tags=["anniversary", "budget"],
            durability="low", description="Gold-plated brass, plating wears within 6-12 months.",
            delivery_days=2),
    Product("j-103", "Sterling Silver Engraved Pendant", "jewellery", 2200, "merchant-established", False,
            material="sterling silver", occasion_tags=["anniversary", "engraving"],
            durability="medium", description="925 sterling silver, free engraving, tarnish-resistant coating.",
            delivery_days=3),
    Product("j-104", "18K Gold Anniversary Ring", "jewellery", 6250, "merchant-premium", False,
            material="18K gold", occasion_tags=["anniversary", "permanent"],
            durability="high", description="18K gold, diamond-cut finish, includes certificate of authenticity.",
            delivery_days=5, review_score=4.6, brand_tier="premium"),
    Product("j-105", "Imitation Gold Bracelet", "jewellery", 699, "merchant-new-07", True,
            material="imitation alloy", occasion_tags=["anniversary", "budget"],
            durability="low", description="Imitation alloy, gold-tone finish, not hallmarked.",
            delivery_days=6),

    # --- Electronics -------------------------------------------------------
    Product("e-201", "Wireless Earbuds, ANC", "electronics", 3499, "merchant-established", False,
            occasion_tags=["gift", "everyday"], durability="medium",
            description="Active noise cancellation, 24h battery with case.", delivery_days=2),
    Product("e-202", "Smartwatch, Fitness Edition", "electronics", 5999, "merchant-established", False,
            occasion_tags=["gift", "fitness"], durability="medium",
            description="Heart-rate + SpO2 tracking, 7-day battery.", delivery_days=3,
            review_score=4.1, promoted=True),

    # --- DIY / model rocket -------------------------------------------------
    Product("d-301", "Cardboard Rocket Body Kit", "diy", 249, "merchant-established", False,
            occasion_tags=["school project", "overnight"], durability="low",
            description="Pre-cut cardboard body tube + nose cone, assembles in under an hour.",
            delivery_days=1),
    Product("d-302", "PVC Pipe Rocket Body (50cm)", "diy", 399, "merchant-established", False,
            occasion_tags=["school project"], durability="high",
            description="Rigid PVC body tube, needs a hacksaw to cut fins separately.",
            delivery_days=2),
    Product("d-303", "Craft Glue, Fast-Bond", "diy", 89, "merchant-established", False,
            occasion_tags=["school project", "overnight"], durability=None,
            description="Fast-setting craft adhesive, dries in 5 minutes.", delivery_days=1),
    Product("d-304", "Poster Paint Set (6 colours)", "diy", 149, "merchant-established", False,
            occasion_tags=["school project"], durability=None,
            description="Water-based poster paint, 6 colours, non-toxic.", delivery_days=1),
    Product("d-305", "Balsa Wood Fin Sheet Set", "diy", 179, "merchant-established", False,
            occasion_tags=["school project"], durability="medium",
            description="Pre-cut balsa wood fins, glue-ready edges.", delivery_days=2),

    # --- Get-well / care -----------------------------------------------------
    Product("c-401", "Get-Well Fruit Basket", "care", 449, "merchant-established", False,
            occasion_tags=["recovery", "same-day"], durability=None,
            description="Seasonal fruit basket, same-day delivery in metro areas.", delivery_days=0),
    Product("c-402", "Herbal Tea Gift Box", "care", 399, "merchant-established", False,
            occasion_tags=["recovery"], durability=None,
            description="Caffeine-free herbal tea assortment, gift-boxed.", delivery_days=1),

    # --- High-value / attack-surface demo items ------------------------------
    Product("h-501", "Premium Noise-Cancelling Headphones", "electronics", 28500, "merchant-premium", False,
            occasion_tags=["gift"], durability="high",
            description="Flagship over-ear headphones, 30h battery.", delivery_days=3,
            review_score=4.7, brand_tier="premium"),
    Product("h-502", "Imported Designer Watch", "luxury", 42000, "merchant-new-19", True,
            occasion_tags=["gift", "luxury"], durability="high",
            description="Imported chronograph watch, limited stock.", delivery_days=7,
            # Deliberately a "dark pattern" demo case: heavily promoted despite mediocre
            # reviews and a brand-new, unverified merchant -- the Bias Detection Engine
            # (app/bias_detection.py) is built to catch exactly this shape of listing.
            review_score=3.2, brand_tier="premium", promoted=True),
]

# Additional listings "ingested" from the mock external storefront (see
# app/storefront.py) -- carries individual, timestamped reviews rather than
# a single pre-aggregated score, for the Review Authenticity check
# (app/review_authenticity.py) to analyze. Imported here, after Product is
# already defined, rather than the other way around, so storefront.py can
# import Product from this module without a circular import.
from app.storefront import ingest_from_storefront as _ingest_from_storefront  # noqa: E402

CATALOGUE.extend(_ingest_from_storefront())
