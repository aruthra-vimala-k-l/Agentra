"""Counterfactual Decision Checker (CDC) -- the "Secure" stage component the
proposal (§5) names Counterfactual Re-Derivation: "a sandboxed, injection-
resistant model instance re-derives what an unmanipulated agent would have
decided given the same original intent and clean product data. Material
divergence from the actual decision = manipulation signal."

Deliberately a rule-based, fully-deterministic re-derivation rather than a
second live model call for this build, for two reasons that matter for a
capstone/buildathon demo specifically:

  1. Determinism. A live model instance re-deriving a decision would make the
     divergence check itself non-reproducible in front of a panel -- exactly
     the "8/10 in rehearsal is a failed demo" problem the rest of this build
     was scoped to avoid.
  2. IP boundary. The real semantic-divergence-weighted-by-irreversibility
     mechanism (the actual patent-relevant internals, and the MELON prior-art
     differentiation) is not something to hand-wave into a hand-off deck --
     it belongs in front of the patent attorney before it appears in any
     public artifact. This module implements the same CONTRACT (trusted
     re-derivation -> divergence score -> manipulation signal) with a fully
     transparent, inspectable scoring function standing in for the sandboxed
     model, so the *pipeline shape* is real and demoable without exposing
     that mechanism.

How it actually re-derives "cleanly": `app/retrieval.signal_contributions()`
already names every signal that can move a product's rank (see
RANK_AFFECTING_SIGNALS). Two of those are populated from fields a poisoned or
spoofed listing could plausibly fake -- `review_influence` (a review score)
and `brand_preference` (a brand tier tag) -- while the rest are either
directly stated by the shopper (occasion, budget, durability preference) or a
structural catalogue fact cross-checked against the shopper's own words
(price, merchant novelty). Re-ranking the SAME shortlist using only the
latter, "trusted" signals and comparing its winner against the shortlist that
was actually shown is the re-derivation: if they agree, the shortlist's top
position rests on real fit; if they disagree by more than noise, the actual
top pick only wins once the gameable fields are counted, which is exactly
the shape of a poisoned-listing attack (see catalog.py's h-502, "deliberately
a dark pattern demo case").
"""
from __future__ import annotations

from dataclasses import dataclass

from app.intent import IntentSlots
from app.retrieval import RankedProduct, coalition_value, signal_contributions

# Signals a shopper's own words or a hard catalogue fact ground -- not
# something an attacker-controlled listing can plausibly spoof on its own.
TRUSTED_SIGNALS = frozenset({"price_fit", "durability_match", "occasion_match", "new_merchant_caution"})

# Signals populated from externally-suppliable listing fields (a review
# score, a self-declared brand tier) -- exactly what a poisoned or spoofed
# listing would inflate.
GAMEABLE_SIGNALS = ("review_influence", "brand_preference")

# A clean-score gap below this is treated as noise/a near-tie, not a
# manipulation signal -- otherwise every shortlist with any review spread at
# all would "diverge." Calibrated so a rank flip driven by a single strong
# gameable signal (e.g. a maxed-out review score plus a premium brand tag,
# worth up to ~2.0 combined) clears it, while a fractional review-score
# difference alone does not.
DIVERGENCE_THRESHOLD = 0.75


@dataclass
class CounterfactualResult:
    actual_top_id: str | None
    clean_top_id: str | None
    diverged: bool
    divergence_score: float
    gameable_signal_share: float
    note: str

    def as_dict(self) -> dict:
        return {
            "actual_top_id": self.actual_top_id,
            "clean_top_id": self.clean_top_id,
            "diverged": self.diverged,
            "divergence_score": self.divergence_score,
            "gameable_signal_share": self.gameable_signal_share,
            "note": self.note,
        }


def rederive(ranked: list[RankedProduct], slots: IntentSlots) -> CounterfactualResult:
    """Re-rank the shortlist actually shown this turn using only trusted
    signals, and compare its winner against the shortlist's real #1. Runs
    once per shortlist (at retrieval time), independent of which item the
    shopper eventually picks -- the question CDC asks is "was the shortlist
    itself steered," not "is this particular purchase suspicious."
    """
    if not ranked:
        return CounterfactualResult(
            actual_top_id=None, clean_top_id=None, diverged=False,
            divergence_score=0.0, gameable_signal_share=0.0,
            note="No shortlist to check.",
        )

    actual_top_id = ranked[0].product.id
    actual_order = [r.product.id for r in ranked]
    contributions_by_id = {r.product.id: signal_contributions(r.product, slots) for r in ranked}
    clean_scores = {pid: coalition_value(TRUSTED_SIGNALS, c) for pid, c in contributions_by_id.items()}

    # Tie-break deterministically toward whichever product the ACTUAL ranking
    # preferred, so a genuine tie under trusted-only signals is never reported
    # as divergence just because of dict/iteration order.
    clean_top_id = max(clean_scores, key=lambda pid: (clean_scores[pid], -actual_order.index(pid)))

    divergence_score = round(clean_scores[clean_top_id] - clean_scores.get(actual_top_id, 0.0), 3)
    diverged = clean_top_id != actual_top_id and divergence_score >= DIVERGENCE_THRESHOLD

    actual_contributions = contributions_by_id[actual_top_id]
    gameable_share = round(sum(max(0.0, actual_contributions.get(k, 0.0)) for k in GAMEABLE_SIGNALS), 3)

    if diverged:
        clean_name = next(r.product.name for r in ranked if r.product.id == clean_top_id)
        actual_name = next(r.product.name for r in ranked if r.product.id == actual_top_id)
        note = (
            f"Clean re-derivation (price fit, durability, occasion match, merchant history only) "
            f"would have surfaced {clean_name!r} ({clean_top_id}) instead of the actual top pick, "
            f"{actual_name!r} ({actual_top_id}) -- the actual pick only leads once review score and "
            f"brand tier are counted, worth {gameable_share:.2f} points here. That is a manipulation "
            f"signal, not proof of an attack: this shortlist's #1 spot is more sensitive to "
            f"externally-suppliable fields than to core fit."
        )
    else:
        note = "Clean re-derivation agrees with the actual shortlist's top pick -- no manipulation signal."

    return CounterfactualResult(
        actual_top_id=actual_top_id, clean_top_id=clean_top_id, diverged=diverged,
        divergence_score=divergence_score if diverged else 0.0,
        gameable_signal_share=gameable_share, note=note,
    )


# ---------------------------------------------------------------------------------------------------------------------
# CDC v2 -- semantic divergence weighted by irreversibility  (the proposal's Counterfactual Re-Derivation, Stage 3)
#
# The question: "if the agent had decided from trusted information only, would it have bought something materially
# different -- and how costly would it be to be wrong, given that this purchase may not be reversible?"
#
#   1. Re-derivation      clean_top = argmax over the shown shortlist of the TRUSTED-signal score (above).
#   2. Score gap     S    = clamp((clean(clean_top) - clean(chosen)) / SCORE_NORM, 0, 1)
#                           how much the choice only wins thanks to gameable fields (review score, brand tag).
#   3. Materiality   M    = 0.5 * d_sem + 0.5 * d_attr          (both in [0, 1])
#                           d_sem  = 1 - cosine(TF-IDF(chosen text), TF-IDF(clean_top text))   semantic distance
#                           d_attr = 0.5 * min(1, |ln(price_chosen / price_clean)| / ln 3) + 0.5 * [category differs]
#                           A rank flip between two near-identical listings is immaterial; between different things
#                           it is not. TF-IDF is fitted on the real catalogue, deterministic, no model download.
#   4. Divergence    D    = S * (0.25 + 0.75 * M)
#   5. Weighted      W    = D * (1 + IRC / 100)                 IRC = irreversibility score 0-100 (app/irreversibility)
#                           The same divergence is twice as serious when the purchase cannot be undone.
#
#   W >= W_ESCALATE -> manipulation suspected, human review.   W >= W_BLOCK and IRC level HIGH/CRITICAL -> block.
#   Thresholds are calibrated in tests/test_counterfactual_v2.py against a known poisoned listing and benign shortlists.
# ---------------------------------------------------------------------------------------------------------------------
import math
import re as _re

SCORE_NORM = 3.0
W_ESCALATE = 0.30
W_BLOCK = 0.60
_TOKEN = _re.compile(r"[a-z0-9]{3,}")
_IDF: dict[str, float] | None = None


def _doc(p) -> list[str]:
    return _TOKEN.findall(f"{p.name} {p.description[:600]} {p.category}".lower())


def _idf() -> dict[str, float]:
    global _IDF
    if _IDF is None:
        from app.catalog import CATALOGUE
        df: dict[str, int] = {}
        for p in CATALOGUE:
            for t in set(_doc(p)):
                df[t] = df.get(t, 0) + 1
        n = max(1, len(CATALOGUE))
        _IDF = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
    return _IDF


def _vec(p) -> dict[str, float]:
    idf, counts = _idf(), {}
    for t in _doc(p):
        counts[t] = counts.get(t, 0) + 1
    default = math.log(1 + len(idf)) + 1.0
    return {t: (1 + math.log(c)) * idf.get(t, default) for t, c in counts.items()}


def semantic_distance(a, b) -> float:
    """1 - cosine similarity of the two products' TF-IDF vectors (0 identical wording ... 1 nothing in common)."""
    va, vb = _vec(a), _vec(b)
    dot = sum(w * vb.get(t, 0.0) for t, w in va.items())
    na, nb = math.sqrt(sum(w * w for w in va.values())), math.sqrt(sum(w * w for w in vb.values()))
    return 1.0 if not na or not nb else round(1.0 - dot / (na * nb), 4)


def attribute_distance(a, b) -> float:
    price = min(1.0, abs(math.log(max(a.price, 1) / max(b.price, 1))) / math.log(3))
    return round(0.5 * price + 0.5 * (0.0 if a.category == b.category else 1.0), 4)


@dataclass
class CDCAssessment:
    chosen_id: str
    clean_top_id: str | None
    score_gap: float            # S
    semantic_distance: float    # d_sem
    attribute_distance: float   # d_attr
    materiality: float          # M
    divergence: float           # D
    irreversibility_score: int  # IRC
    weighted_divergence: float  # W
    level: str                  # NONE | ESCALATE | BLOCK
    note: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def assess_choice(ranked: list[RankedProduct], slots: IntentSlots, chosen_id: str, irreversibility_score: int,
                  irreversibility_level: str = "LOW") -> CDCAssessment:
    """Full CDC v2 for the item actually being bought."""
    by_id = {r.product.id: r.product for r in ranked}
    if chosen_id not in by_id:   # item chosen from outside the shown shortlist: nothing to re-derive against
        return CDCAssessment(chosen_id, None, 0, 0, 0, 0, 0, irreversibility_score, 0.0, "NONE",
                             "Chosen item was not part of the shortlist, so no clean comparison was possible.")
    actual_order = [r.product.id for r in ranked]
    clean = {r.product.id: coalition_value(TRUSTED_SIGNALS, signal_contributions(r.product, slots)) for r in ranked}
    clean_top = max(clean, key=lambda pid: (clean[pid], -actual_order.index(pid)))
    chosen = by_id[chosen_id]
    if clean_top == chosen_id:
        return CDCAssessment(chosen_id, clean_top, 0, 0, 0, 0, 0, irreversibility_score, 0.0, "NONE",
                             "The chosen item is also what a trusted-information-only re-derivation would pick.")
    S = max(0.0, min(1.0, (clean[clean_top] - clean[chosen_id]) / SCORE_NORM))
    d_sem = semantic_distance(chosen, by_id[clean_top])
    d_attr = attribute_distance(chosen, by_id[clean_top])
    M = 0.5 * d_sem + 0.5 * d_attr
    D = S * (0.25 + 0.75 * M)
    W = round(D * (1 + irreversibility_score / 100), 4)
    level = "NONE"
    if W >= W_BLOCK and irreversibility_level in ("HIGH", "CRITICAL"):
        level = "BLOCK"
    elif W >= W_ESCALATE:
        level = "ESCALATE"
    note = (f"Trusted-only re-derivation prefers {by_id[clean_top].name!r}; the chosen {chosen.name!r} trails it by "
            f"{S:.2f} (normalised) on trusted signals, the two differ by materiality {M:.2f} "
            f"(semantic {d_sem:.2f}, attributes {d_attr:.2f}), irreversibility {irreversibility_score}/100 "
            f"-> weighted divergence {W:.2f} ({level}).")
    return CDCAssessment(chosen_id, clean_top, round(S, 4), d_sem, d_attr, round(M, 4), round(D, 4),
                         irreversibility_score, W, level, note)
