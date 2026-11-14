"""Retrieval + ranking + grounded explanation.

Non-negotiable design principle carried over from the conversational-commerce
brief: "never hallucinate an attribute the catalogue doesn't state." Every
justification string below is built by reading fields off the Product
dataclass -- there is no free-text generation step that could invent a
material or a durability claim.

`signal_contributions()` is the single scoring function both the ranker and
the SHAP-style attribution module (app/attribution.py) read from -- one
scoring pass, two consumers, so the number a product is ranked by and the
number the decomposition explains can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.catalog import Product, search
from app.intent import IntentSlots

# Signals that are allowed to move a product's rank. `nudge_exposure` is
# intentionally excluded: a sponsored/boosted placement (Product.promoted)
# is tracked as a signal for transparency and for the Bias Detection Engine,
# but by design it earns zero ranking weight -- promotion is disclosed, not
# rewarded. This is a business rule, not an oversight; app/bias_detection.py
# specifically checks that this holds.
RANK_AFFECTING_SIGNALS = (
    "price_fit", "durability_match", "occasion_match",
    "review_influence", "brand_preference", "new_merchant_caution",
)

# A durable item that also matches the stated occasion is worth more than the
# sum of the two signals in isolation (a durable-but-irrelevant item and a
# relevant-but-flimsy item are both worse gifts than one that's both) -- this
# nonlinear interaction is what makes the Shapley decomposition in
# app/attribution.py meaningful rather than a relabeling of the raw terms.
# Construct-validity guard (found by app/price_tilt_study.py): in the REAL catalogue, ``brand_tier`` is built as "top price
# quartile within category" (app/catalog_build.py), so it is a price rank, not brand information (Spearman with price
# 0.67-0.75). Letting it earn ranking weight made the ranker prefer expensive items (price percentile 0.81 with no budget,
# 0.52 once the signal is removed). The signal therefore only counts on data where the tier is an independent field (mock).
GUARD_PRICE_DERIVED_TIER = True

# price_fit shape. "flat" = every affordable item fits the budget equally (price-neutral inside the budget, like the live
# assistant's evaluate()); "spend_to_budget" = the original shape that paid 1.5 to items using >= half the budget, which made
# the ranker prefer dearer items (price percentile 0.87 under a loose budget, app/price_tilt_study.py).
PRICE_FIT_MODE = "flat"

SYNERGY_BONUS = 0.5
SYNERGY_PAIR = ("durability_match", "occasion_match")


@dataclass
class RankedProduct:
    product: Product
    score: float
    justification: str


def signal_contributions(p: Product, slots: IntentSlots) -> dict[str, float]:
    """Every commerce signal contributing to this product's score, named and
    isolated -- this dict IS the raw material the attribution/bias layer
    decomposes. Keep every new ranking factor introduced here, not just
    folded silently into a single number.
    """
    c: dict[str, float] = {}

    if slots.budget_max is not None:
        headroom = slots.budget_max - p.price
        if headroom < 0:
            c["price_fit"] = -2.0  # shouldn't normally reach here post-filter; defensive
        elif PRICE_FIT_MODE == "flat" or headroom <= slots.budget_max * 0.5:
            c["price_fit"] = 1.5
        else:
            c["price_fit"] = 0.3
    else:
        c["price_fit"] = 0.0

    if slots.wants_durability:
        c["durability_match"] = {"high": 3.0, "medium": 1.0}.get(p.durability, 0.0)
    else:
        c["durability_match"] = 0.0

    c["occasion_match"] = 2.0 if (slots.occasion and slots.occasion in p.occasion_tags) else 0.0
    c["review_influence"] = round((p.review_score - 3.0) * 0.5, 3)
    tier_is_price = GUARD_PRICE_DERIVED_TIER and p.source != "mock"
    c["brand_preference"] = 1.0 if (p.brand_tier == "premium" and not tier_is_price) else 0.0
    c["nudge_exposure"] = 1.0 if p.promoted else 0.0
    c["new_merchant_caution"] = -0.5 if p.is_new_merchant else 0.0
    return c


def coalition_value(active: frozenset[str], contributions: dict[str, float]) -> float:
    """The value function the Shapley calculation is run over: sum of
    whichever rank-affecting signals are 'switched on' in this coalition,
    plus the durability/occasion synergy bonus when both are present and
    both are actually contributing something positive.
    """
    total = sum(v for k, v in contributions.items() if k in active and k in RANK_AFFECTING_SIGNALS)
    a, b = SYNERGY_PAIR
    if a in active and b in active and contributions.get(a, 0) > 0 and contributions.get(b, 0) > 0:
        total += SYNERGY_BONUS
    return total


def total_score(contributions: dict[str, float]) -> float:
    return coalition_value(frozenset(contributions.keys()), contributions)


def _justify(p: Product, slots: IntentSlots) -> str:
    reasons = []
    if slots.wants_durability and p.durability == "high":
        reasons.append("you said it should last, and this is solid material rated high-durability")
    elif slots.wants_durability and p.durability in ("low", None):
        reasons.append("note: you asked for something durable, but this item is lower-durability")
    if slots.budget_max is not None and p.price <= slots.budget_max:
        reasons.append(f"fits your ₹{slots.budget_max:,.0f} budget at ₹{p.price:,.0f}")
    if slots.occasion and slots.occasion in p.occasion_tags:
        reasons.append(f"tagged for {slots.occasion}")
    if p.is_new_merchant:
        reasons.append("from a merchant new to this catalogue -- flagged for the security gate")
    if p.promoted:
        reasons.append("sponsored placement (disclosed) -- did not earn extra ranking weight")
    if not reasons:
        reasons.append(f"matches the {p.category} category you asked about")
    return "; ".join(reasons).capitalize() + "."


def retrieve_and_rank(slots: IntentSlots, top_k: int = 5) -> list[RankedProduct]:
    candidates = search(category=slots.category, max_price=slots.budget_max, occasion=slots.occasion)
    if not candidates and slots.category:
        # Relax the occasion filter before giving up -- availability validation
        # before surfacing anything (brief §4.3): don't return nothing when a
        # slightly broader match exists.
        candidates = search(category=slots.category, max_price=slots.budget_max)

    ranked: list[RankedProduct] = []
    for p in candidates:
        contributions = signal_contributions(p, slots)
        score = total_score(contributions)
        ranked.append(RankedProduct(product=p, score=score, justification=_justify(p, slots)))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k]
