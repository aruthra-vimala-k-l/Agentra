"""SHAP-style signal decomposition -- the ML attribution layer the original
proposal calls the Behavioral Decision Attributor (BDA) and Patent Claim 1
("game-theoretic decomposition of commerce signals into price weight,
position bias, review influence, brand preference, nudge sensitivity").

This is a genuine Shapley-value computation, not a relabeling of the raw
per-signal scores: `app/retrieval.signal_contributions()` includes a
nonlinear synergy term (a durable item that also matches the occasion is
worth more than the two signals added separately -- see
`app/retrieval.SYNERGY_BONUS`), so the exact per-signal contribution to the
final score cannot be read off directly. Shapley values are the
game-theoretically unique way to split credit for a value function like that
fairly across its inputs: attribute to each signal the average of its
marginal contribution across every possible order in which the signals could
have been "switched on."

`shapley_values()` is a small, exact (not approximate/sampled) Shapley
calculator -- feasible here because the shopping agent only has a handful of
named commerce signals (<= 10), so the full 2^n coalition space is trivial to
enumerate. A production system with hundreds of raw features would need
KernelSHAP-style sampling instead; exact computation is a deliberate,
disclosed choice for this scale, not a corner cut.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

from app.catalog import Product
from app.intent import IntentSlots
from app.retrieval import coalition_value, signal_contributions


def shapley_values(feature_names: list[str], value_fn: Callable[[frozenset[str]], float]) -> dict[str, float]:
    """Exact Shapley value of every feature in `feature_names` under
    `value_fn`, a set function mapping a coalition (subset of features
    switched on) to a real-valued payoff.

    phi_i = sum over subsets S not containing i of:
        |S|! * (n - |S| - 1)! / n!  *  (value_fn(S u {i}) - value_fn(S))

    Satisfies the efficiency property: sum(phi_i) == value_fn(full set) -
    value_fn(empty set) -- checked in tests/test_attribution.py rather than
    just asserted here.
    """
    n = len(feature_names)
    if n == 0:
        return {}
    result: dict[str, float] = {}
    for feature in feature_names:
        others = [f for f in feature_names if f != feature]
        total = 0.0
        for r in range(len(others) + 1):
            for subset in combinations(others, r):
                s = frozenset(subset)
                marginal = value_fn(s | {feature}) - value_fn(s)
                weight = (math.factorial(r) * math.factorial(n - r - 1)) / math.factorial(n)
                total += weight * marginal
        result[feature] = round(total, 4)
    return result


@dataclass
class AttributionResult:
    product_id: str
    total_score: float
    raw_signal_contributions: dict[str, float]
    shapley_values: dict[str, float]
    top_signals: list[tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "total_score": round(self.total_score, 4),
            "raw_signal_contributions": self.raw_signal_contributions,
            "shapley_values": self.shapley_values,
            "top_signals": [{"signal": s, "value": v} for s, v in self.top_signals],
        }


def attribute(product: Product, slots: IntentSlots) -> AttributionResult:
    """One inference pass over a product's ranking signals, decomposed. This
    same AttributionResult is what feeds *both* the consumer-facing
    explanation and the enterprise-facing behavioral profile in
    app/behavioral_profile.py -- one computation, two renderings, which is
    the mechanism Patent Claim 2 (dual-layer output) actually depends on.
    """
    contributions = signal_contributions(product, slots)
    feature_names = list(contributions.keys())

    def value_fn(active: frozenset[str]) -> float:
        return coalition_value(active, contributions)

    shap = shapley_values(feature_names, value_fn)
    total = value_fn(frozenset(feature_names))
    top = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return AttributionResult(
        product_id=product.id,
        total_score=total,
        raw_signal_contributions=contributions,
        shapley_values=shap,
        top_signals=top,
    )
