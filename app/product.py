"""Product record shared by the mock and real catalogues."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Product:
    id: str
    name: str
    category: str
    price: float
    merchant: str
    is_new_merchant: bool
    material: str | None = None
    occasion_tags: list[str] = field(default_factory=list)
    durability: str | None = None  # "high" | "medium" | "low"
    description: str = ""
    in_stock: bool = True
    delivery_days: int | None = 3  # None = unknown (real catalogues rarely state it)
    # Added for the signal-decomposition / bias-detection layer (see
    # app/attribution.py, app/bias_detection.py) -- these are the extra
    # commerce signals the original proposal named (review influence, brand
    # preference, nudge sensitivity) that the mock catalogue didn't carry
    # fields for yet.
    review_score: float = 4.0       # out of 5
    brand_tier: str = "standard"    # "standard" | "premium"
    promoted: bool = False          # sponsored/boosted placement, independent of fit
    # Provenance -- every real record says where it came from (evaluator
    # requirement: no unlabeled/synthetic data in the real pipeline).
    source: str = "mock"            # "mock" | "open_food_facts" | "amazon_reviews_2023" | ...
    source_url: str = ""
    review_count: int | None = None
    # Individual reviews {reviewer, rating, timestamp(ISO), text} when the source has them;
    # app/review_authenticity.py analyses these.
    reviews: list[dict] = field(default_factory=list)
