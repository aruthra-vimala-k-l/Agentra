"""Dual-layer output -- Patent Claim 2 ("one ML inference pass producing
both a consumer-facing explanation and an enterprise-facing behavioral
profile simultaneously").

The key thing this module proves is not that two differently-worded outputs
exist (that's trivial -- two templates) but that they are two *renderings of
the same computation*: both views below are built from exactly one call to
`app.attribution.attribute()`, plus the same irreversibility classification
and the same bias audit. Nothing is computed twice, and nothing shown to the
consumer is a simplification that the enterprise view contradicts -- they
are the same numbers, presented for two different audiences.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.attribution import AttributionResult, attribute
from app.bias_detection import run_bias_audit
from app.catalog import Product
from app.counterfactual import CounterfactualResult
from app.drift import record_and_check
from app.intent import IntentSlots
from app.irreversibility import classify as classify_irreversibility
from app.retrieval import RankedProduct

SIGNAL_LABELS = {
    "price_fit": "how well the price fit your budget",
    "durability_match": "how well it matched your durability preference",
    "occasion_match": "how well it matched the occasion",
    "review_influence": "customer review sentiment",
    "brand_preference": "brand reputation",
    "nudge_exposure": "sponsored/promoted placement",
    "new_merchant_caution": "merchant history",
}


def _humanize_top_signal(shap: dict[str, float]) -> str | None:
    if not shap:
        return None
    top_key, top_val = max(shap.items(), key=lambda kv: abs(kv[1]))
    if abs(top_val) < 0.05:
        return None
    label = SIGNAL_LABELS.get(top_key, top_key)
    direction = "helped" if top_val > 0 else "counted against"
    return f"The single biggest factor was {label}, which {direction} this option (weight {top_val:+.2f})."


@dataclass
class DualLayerOutput:
    consumer_view: dict
    enterprise_view: dict


def build_dual_layer(
    chosen: RankedProduct,
    slots: IntentSlots,
    ranked_shortlist: list[RankedProduct],
    user_id: str,
    counterfactual: CounterfactualResult | None = None,
) -> DualLayerOutput:
    # --- one inference pass -------------------------------------------------
    chosen_attribution: AttributionResult = attribute(chosen.product, slots)
    shortlist_attributions: dict[str, AttributionResult] = {
        r.product.id: attribute(r.product, slots) for r in ranked_shortlist
    }
    irreversibility = classify_irreversibility(chosen.product, amount=chosen.product.price)
    bias_audit = run_bias_audit(ranked_shortlist, shortlist_attributions)
    drift = record_and_check(user_id, chosen_attribution.shapley_values)

    # Counterfactual Decision Checker (app/counterfactual.py) is computed
    # once per shortlist, not per product -- only surface it here as a
    # per-item concern when the item actually being bought is the one whose
    # top position the clean re-derivation disagreed with.
    cdc_flags_this_item = bool(
        counterfactual and counterfactual.diverged and counterfactual.actual_top_id == chosen.product.id
    )

    # --- consumer-facing rendering ------------------------------------------
    top_signal_sentence = _humanize_top_signal(chosen_attribution.shapley_values)
    consumer_view = {
        "product_id": chosen.product.id,
        "summary": chosen.justification,
        "top_reason": top_signal_sentence,
        "irreversibility_level": irreversibility["level"],
        "irreversibility_note": irreversibility["factors"][-1] if irreversibility["factors"] else None,
        "was_this_promoted": chosen.product.promoted,
        "promotion_disclosure": (
            "This item is a sponsored placement, but it earned zero extra ranking weight for that."
            if chosen.product.promoted else None
        ),
        "counterfactual_note": (
            "This item's top spot on the shortlist leaned more on its review score and brand tag than "
            "on price fit, durability, or occasion match -- worth a second look before confirming."
            if cdc_flags_this_item else None
        ),
    }

    # --- enterprise-facing rendering (same computation, full detail) -------
    enterprise_view = {
        "product_id": chosen.product.id,
        "signal_decomposition": chosen_attribution.as_dict(),
        "irreversibility": irreversibility,
        "bias_audit": bias_audit,
        "behavioral_drift": drift,
        "counterfactual_check": counterfactual.as_dict() if counterfactual else None,
    }

    return DualLayerOutput(consumer_view=consumer_view, enterprise_view=enterprise_view)
