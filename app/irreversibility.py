"""Irreversibility Classifier -- standalone component.

In the original proposal this scoring lived only implicitly inside ASIDE's
guardrail amount-threshold checks (a purchase over ₹5,000/₹25,000 gets more
scrutiny, but "irreversibility" itself was never scored or explained on its
own). This module pulls it out as an independent, explainable classifier
that runs on this agent's side of the boundary -- it doesn't touch or
replace anything in ASIDE; it produces one more signal that gets attached to
the Canonical Decision Object and shown to both the consumer and the
enterprise dashboard.

This is a rule-based classifier, not a learned model -- deliberately: the
factors are the same ones a human reviewer would name to justify a
"should this even be one-click autonomous" judgment, so it needs to be
auditable line-by-line, not a black box on top of a black box.

Calibration note: `app/orchestrator.py` now actually *acts* on a HIGH/
CRITICAL result (escalating an ALLOW to REQUIRE_HUMAN_APPROVAL) rather than
only displaying it, so the category base scores below are deliberately kept
below the HIGH threshold on their own -- an ordinary, non-personalized,
established-merchant jewellery purchase should NOT be flagged just for
being jewellery. What pushes a specific item into HIGH/CRITICAL is the
combination of factors (personalization, a new/unverified merchant, a large
amount), the same way a human reviewer would actually reason about it.
"""
from __future__ import annotations

from app.catalog import Product

CATEGORY_BASE_SCORE = {
    "jewellery": 30,
    "luxury": 35,
    "electronics": 20,
    "diy": 10,
    "care": 5,
}


def classify(product: Product, amount: float | None = None) -> dict:
    amount = product.price if amount is None else amount
    base = CATEGORY_BASE_SCORE.get(product.category, 20)
    score = base
    factors = [f"Base category risk for '{product.category}': {base}/100"]

    if "permanent" in product.occasion_tags or "engraving" in product.occasion_tags:
        score += 20
        factors.append("Personalized or explicitly 'permanent' item -- typically non-returnable once customized.")

    if product.is_new_merchant:
        score += 15
        factors.append("New/unverified merchant -- return process is untested.")

    if amount >= 25000:
        score += 20
        factors.append("Above the high-value approval threshold -- financial reversal is materially harder.")
    elif amount >= 5000:
        score += 5
        factors.append("Above the autonomous-spend threshold -- moderate financial exposure if this needs reversing.")

    if product.delivery_days == 0:
        score += 5
        factors.append("Same-day/perishable delivery -- little to no return window.")

    score = max(0, min(100, score))
    if score >= 70:
        level = "CRITICAL"
    elif score >= 45:
        level = "HIGH"
    elif score >= 20:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"score": score, "level": level, "factors": factors, "amount": amount}


def classify_listing(amount: float, seller_unverified: bool = False) -> dict:
    """Same rule set for a live web listing, where the catalogue fields (category tags, merchant age, delivery days) do not
    exist. Only the factors a listing can actually show are used: amount and whether the seller could be verified."""
    score, factors = 20, ["Base risk for an online purchase of an unspecified category: 20/100"]
    if seller_unverified:
        score += 15
        factors.append("Seller could not be verified -- return process is untested.")
    if amount >= 25000:
        score += 20
        factors.append("Above the high-value approval threshold -- financial reversal is materially harder.")
    elif amount >= 5000:
        score += 5
        factors.append("Above the autonomous-spend threshold -- moderate financial exposure if this needs reversing.")
    level = "CRITICAL" if score >= 70 else "HIGH" if score >= 45 else "MEDIUM" if score >= 20 else "LOW"
    return {"score": score, "level": level, "factors": factors, "amount": amount}
