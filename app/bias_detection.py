"""Bias Detection Engine -- standalone component, per the original proposal's
citation of position-bias and choice-architecture research (MIT/Columbia).

Three checks, each over the shortlist actually shown to the user this turn:

- position_bias_risk: how much of the #1 spot is explained by an actual
  score gap vs. an arbitrary tie -- a small gap means display order, not
  fit, is doing the persuading.
- nudge_exposure_report: cross-checks Product.promoted against the Shapley
  decomposition's `nudge_exposure` term (see app/attribution.py) to prove --
  per shortlist, not just as a design claim -- that sponsored placement
  earned zero ranking credit.
- choice_homogeneity: how much real variety is behind an apparently long
  shortlist (distinct merchants/materials), since a list of five near-
  identical options from one merchant isn't meaningful choice.

This audits *this agent's own* shortlist -- it is a self-check, not a claim
that the underlying catalogue or ASIDE are bias-free.
"""
from __future__ import annotations

from app.attribution import AttributionResult
from app.retrieval import RankedProduct
from app.review_authenticity import check as check_review_authenticity


def position_bias_risk(ranked: list[RankedProduct]) -> dict:
    if len(ranked) < 2:
        return {"risk": "low", "score_gap": None,
                "note": "Fewer than two candidates -- position bias isn't applicable."}
    scores = [r.score for r in ranked]
    gap = round(scores[0] - scores[1], 3)
    if gap < 0.5:
        risk, note = "high", (
            f"Top two options are separated by a score gap of only {gap} -- display order, "
            "not fit, is likely driving which one gets picked. Consider disclosing the near-tie "
            "or randomizing presentation order among tied items."
        )
    elif gap < 1.5:
        risk, note = "medium", f"Score gap of {gap} between #1 and #2 -- some position-order influence likely."
    else:
        risk, note = "low", f"Score gap of {gap} is decisive -- ranking, not position, explains the top choice."
    return {"risk": risk, "score_gap": gap, "note": note}


def nudge_exposure_report(ranked: list[RankedProduct], attributions: dict[str, AttributionResult]) -> dict:
    flags = []
    for r in ranked:
        if not r.product.promoted:
            continue
        a = attributions.get(r.product.id)
        earned_weight = a.shapley_values.get("nudge_exposure", None) if a else None
        if earned_weight == 0:
            flags.append({
                "product_id": r.product.id, "name": r.product.name, "clean": True,
                "note": "Sponsored placement earned zero ranking weight -- shown on merit, not paid placement.",
            })
        else:
            flags.append({
                "product_id": r.product.id, "name": r.product.name, "clean": False,
                "note": f"Sponsored placement appears to carry ranking weight ({earned_weight}) -- investigate.",
            })
    return {"promoted_items_in_shortlist": len(flags), "flags": flags,
            "all_clean": all(f["clean"] for f in flags) if flags else True}


def choice_homogeneity(ranked: list[RankedProduct]) -> dict:
    if not ranked:
        return {"distinct_merchants": 0, "diversity_score": 0.0, "homogeneity_flag": False,
                "note": "No candidates to assess."}
    merchants = {r.product.merchant for r in ranked}
    materials = {r.product.material for r in ranked if r.product.material}
    prices = [r.product.price for r in ranked]
    diversity_score = round(len(merchants) / len(ranked), 2)
    homogeneous = diversity_score < 0.4 and len(ranked) >= 3
    return {
        "distinct_merchants": len(merchants),
        "distinct_materials": len(materials),
        "price_range": [min(prices), max(prices)],
        "diversity_score": diversity_score,
        "homogeneity_flag": homogeneous,
        "note": ("Shortlist is dominated by very few merchants -- limited real choice behind an "
                 "apparently long list." if homogeneous else
                 "Shortlist draws from a reasonably diverse set of merchants."),
    }


def review_authenticity_report(ranked: list[RankedProduct]) -> dict:
    """Runs app/review_authenticity.py (the Anomaly Detection Module) over
    every item in the shortlist -- a distinct check from nudge_exposure
    above: this looks at whether the underlying review DATA looks organic at
    all, not at whether the ranking algorithm used it fairly."""
    results = [check_review_authenticity(r.product) for r in ranked]
    flagged = [r.as_dict() for r in results if r.suspicious]
    return {"checked": len(results), "flagged_count": len(flagged), "flagged": flagged}


def run_bias_audit(ranked: list[RankedProduct], attributions: dict[str, AttributionResult]) -> dict:
    return {
        "position_bias": position_bias_risk(ranked),
        "nudge_exposure": nudge_exposure_report(ranked, attributions),
        "choice_homogeneity": choice_homogeneity(ranked),
        "review_authenticity": review_authenticity_report(ranked),
    }
