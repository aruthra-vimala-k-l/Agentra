"""Review Authenticity Check -- the first concrete implementation of the
Anomaly Detection Module (ADM) the proposal names ("flags statistically
unusual purchases -- outlier prices, unusual merchants, velocity anomalies,
category mismatches") but never actually built as its own component:
velocity/amount anomalies are handled by ASIDE's own guardrails, and nothing
on Agentra's side looked at the LISTING itself for signs of manipulation.

This is a distinct concern from app/bias_detection.py's nudge_exposure
check. Nudge exposure asks "did a sponsored badge earn ranking weight it
shouldn't have" -- it's about the AGENT's own scoring being fair. This
module asks a different question entirely: "does this listing's review data
look organic at all," independent of whether the agent's ranking used it
correctly. A perfectly fair ranking algorithm can still be fed a fabricated
5-star average -- catching that requires looking at the reviews themselves,
which nothing else in this codebase does (Product.review_score is a single
pre-aggregated float; only listings ingested via app/storefront.py carry the
underlying Product.reviews list this module needs).

Three checks, each look-at-face-value simple on purpose -- these are the
same tells a human trust & safety reviewer would name, not a learned model,
for the same auditability reason app/irreversibility.py is rule-based:

  - burst_detected: every review landing inside a short window is a classic
    review-farming tell -- organic reviews accumulate over the item's whole
    time on sale, not in a multi-day sprint.
  - low reviewer diversity: very few distinct names behind many reviews
    (or the same name posting more than once) means the review count is
    inflated relative to how many actual customers it represents.
  - duplicate_text_detected: near-identical wording repeated across
    "different" reviewers is a template-fraud tell.

None of these alone is proof of fraud -- a genuinely popular item launched
last week could have a real burst of early reviews. That is why this stays
a disclosed signal (like every other Agentra component) rather than an
automatic block: it feeds the gate's escalation logic and the enterprise
dashboard, the same way irreversibility and counterfactual divergence do,
never a silent DENY on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.catalog import Product

# Fewer reviews than this and there simply isn't enough data to say anything
# meaningful about the pattern behind them -- reporting "suspicious" off two
# reviews would be noise, not a signal.
MIN_REVIEWS_TO_ASSESS = 3

# All reviews landing within this many days of each other is treated as a
# burst -- organic review accumulation for an item that's been listed for a
# while does not look like this.
BURST_WINDOW_DAYS = 5

# unique reviewers / review count at or below this ratio means the review
# count is materially inflated relative to how many distinct customers it
# actually represents.
LOW_DIVERSITY_RATIO = 0.5


@dataclass
class ReviewAuthenticityResult:
    product_id: str
    review_count: int
    suspicious: bool
    burst_detected: bool
    reviewer_diversity: float | None
    duplicate_text_detected: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "review_count": self.review_count,
            "suspicious": self.suspicious,
            "burst_detected": self.burst_detected,
            "reviewer_diversity": self.reviewer_diversity,
            "duplicate_text_detected": self.duplicate_text_detected,
            "reasons": self.reasons,
        }


def check(product: Product) -> ReviewAuthenticityResult:
    reviews = product.reviews or []
    if len(reviews) < MIN_REVIEWS_TO_ASSESS:
        return ReviewAuthenticityResult(
            product_id=product.id, review_count=len(reviews), suspicious=False,
            burst_detected=False, reviewer_diversity=None, duplicate_text_detected=False,
            reasons=[] if reviews else ["No individual review data available for this listing."],
        )

    reasons: list[str] = []

    timestamps = sorted(datetime.fromisoformat(r["timestamp"]) for r in reviews)
    burst_detected = (timestamps[-1] - timestamps[0]) <= timedelta(days=BURST_WINDOW_DAYS)
    if burst_detected:
        reasons.append(
            f"All {len(reviews)} reviews were posted within {BURST_WINDOW_DAYS} days of each other -- "
            "unusual for organic review accumulation on an established listing."
        )

    reviewers = [r.get("reviewer", "") for r in reviews]
    reviewer_diversity = round(len(set(reviewers)) / len(reviews), 2)
    if reviewer_diversity <= LOW_DIVERSITY_RATIO:
        reasons.append(
            f"Only {len(set(reviewers))} distinct reviewer name(s) behind {len(reviews)} reviews "
            f"(diversity {reviewer_diversity}) -- the review count may not represent that many "
            "actual customers."
        )

    texts = [r.get("text", "").strip().lower() for r in reviews]
    duplicate_text_detected = len(set(texts)) < len(texts)
    if duplicate_text_detected:
        reasons.append("Near-identical review text repeats across multiple reviews -- a template-fraud tell.")

    return ReviewAuthenticityResult(
        product_id=product.id, review_count=len(reviews), suspicious=bool(reasons),
        burst_detected=burst_detected, reviewer_diversity=reviewer_diversity,
        duplicate_text_detected=duplicate_text_detected, reasons=reasons,
    )
