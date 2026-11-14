"""Anomaly Detection Module (ADM) -- the proposal's "does this purchase look statistically unusual? Wrong price range,
unusual merchant, category mismatch" check, fitted on the REAL catalogue (no synthetic training data).

Two independent detectors, combined into one 0-100 score with plain-language reasons:

1. Robust statistics (explainable): price is compared with the other products of the SAME category using the robust
   z-score  z = 0.6745 * (ln price - median) / MAD  (Iglewicz & Hoaglin, 1993). |z| >= 2.5 is "unusual", >= 3.5 is "very
   unusual" (the standard modified-z cut-offs). A very LOW price is the classic too-good-to-be-true listing tell; a very
   high one is a price-inflation / wrong-category tell. A perfect rating resting on very few reviews is flagged too.
2. Isolation Forest (Liu, Ting & Zhou, ICDM 2008), scikit-learn, fixed seed (deterministic). Features: ln price,
   in-category robust z, rating, ln(1+review count). The product's anomaly score is reported as a percentile of the
   catalogue's own scores (e.g. "more unusual than 98% of real listings"). Skipped if scikit-learn is missing.

What this is NOT: a fraud label. "Unusual" means statistically rare in the real catalogue; rare is not the same as bad
(a genuinely premium item is rare too). Like every Agentra signal it is disclosed to the gate and the user, which
decide what to do with it. Merchant velocity / spend limits stay with ASIDE's guardrails.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

try:  # optional: the robust-statistics detector works without it
    import numpy as np
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - exercised only when scikit-learn is absent
    np = None
    IsolationForest = None

MIN_FIT = 30            # need at least this many products to say what "usual" looks like
Z_UNUSUAL, Z_VERY = 2.5, 3.5
PCTL_UNUSUAL, PCTL_VERY = 97.0, 99.0
ELEVATED_AT, HIGH_AT = 30, 60


def _ln(x: float) -> float:
    return math.log(max(x, 1.0))


def _features(price: float, z: float, rating: float, reviews: int | None) -> list[float]:
    return [_ln(price), z, rating, math.log1p(reviews or 0)]


@dataclass
class Reference:
    """What 'usual' looks like, learned from the real catalogue."""
    by_category: dict[str, tuple[float, float]] = field(default_factory=dict)   # category -> (median ln price, MAD)
    forest: object | None = None
    train_scores: list[float] = field(default_factory=list)
    size: int = 0

    @classmethod
    def fit(cls, products) -> "Reference":
        products = [p for p in products if p.price and p.price > 0]
        ref = cls(size=len(products))
        cats: dict[str, list[float]] = {}
        for p in products:
            cats.setdefault(p.category, []).append(_ln(p.price))
        for c, vals in cats.items():
            med = statistics.median(vals)
            mad = statistics.median([abs(v - med) for v in vals]) or 1e-6
            ref.by_category[c] = (med, mad)
        if IsolationForest is not None and len(products) >= MIN_FIT:
            X = np.array([_features(p.price, ref.price_z(p.category, p.price), p.review_score, p.review_count)
                          for p in products])
            ref.forest = IsolationForest(n_estimators=200, random_state=0).fit(X)
            ref.train_scores = sorted((-ref.forest.score_samples(X)).tolist())
        return ref

    def price_z(self, category: str, price: float) -> float:
        med, mad = self.by_category.get(category, (None, None))
        if med is None:
            return 0.0
        return 0.6745 * (_ln(price) - med) / mad


def _percentile(sorted_scores: list[float], value: float) -> float:
    if not sorted_scores:
        return 0.0
    lo = 0
    for s in sorted_scores:
        if s <= value:
            lo += 1
    return 100.0 * lo / len(sorted_scores)


def assess(product, ref: Reference) -> dict:
    """Score one product against the reference. Returns score (0-100), level, reasons and the raw numbers."""
    reasons: list[str] = []
    pts = 0
    z = ref.price_z(product.category, product.price)
    if ref.size < MIN_FIT:
        return {"score": 0, "level": "NORMAL", "price_robust_z": round(z, 2), "iforest_percentile": None,
                "reasons": ["Not enough catalogue data to judge what is usual."], "reference_size": ref.size}
    if z <= -Z_VERY:
        pts += 60; reasons.append(f"Price is far BELOW what is usual for {product.category} (robust z {z:.1f}) -- a too-good-to-be-true tell.")
    elif z <= -Z_UNUSUAL:
        pts += 35; reasons.append(f"Price is unusually low for {product.category} (robust z {z:.1f}).")
    elif z >= Z_VERY:
        pts += 60; reasons.append(f"Price is far ABOVE what is usual for {product.category} (robust z {z:.1f}).")
    elif z >= Z_UNUSUAL:
        pts += 35; reasons.append(f"Price is unusually high for {product.category} (robust z {z:.1f}).")
    if product.review_score >= 4.8 and (product.review_count is not None and product.review_count < 10):
        pts += 35; reasons.append(f"A near-perfect {product.review_score} rating rests on only {product.review_count} review(s).")
    pct = None
    if ref.forest is not None:
        s = float(-ref.forest.score_samples(np.array([_features(product.price, z, product.review_score, product.review_count)]))[0])
        pct = round(_percentile(ref.train_scores, s), 1)
        if pct >= PCTL_VERY:
            pts += 50; reasons.append(f"Isolation Forest: more unusual than {pct}% of real listings.")
        elif pct >= PCTL_UNUSUAL:
            pts += 30; reasons.append(f"Isolation Forest: more unusual than {pct}% of real listings.")
    score = min(100, pts)
    level = "HIGH" if score >= HIGH_AT else "ELEVATED" if score >= ELEVATED_AT else "NORMAL"
    return {"score": score, "level": level, "price_robust_z": round(z, 2), "iforest_percentile": pct,
            "reasons": reasons or ["Price, rating and review volume are in the usual range for this category."],
            "reference_size": ref.size}


_REF: Reference | None = None


def reference() -> Reference:
    """Reference fitted once from the live catalogue (real products only when CATALOG_SOURCE=real)."""
    global _REF
    if _REF is None:
        from app.catalog import CATALOGUE
        _REF = Reference.fit([p for p in CATALOGUE if p.source != "martverse_testbed"])
    return _REF


def assess_product(product) -> dict:
    return assess(product, reference())
