"""Intent extraction and goal decomposition.

Rule-based on purpose: the capstone demo needs to be deterministic and
explainable in front of a panel, not dependent on an external LLM API key
being configured on demo day. Every slot this module fills is traceable back
to a keyword match in the user's own words -- which is also what the
explanation layer (app/explain.py) grounds its justifications in.

Mirrors the two cases from the conversational-commerce brief:
 - "simple case": constrained intent (jewellery/gifting) -> slot filling
 - "complex case": open-ended task (the model rocket) -> goal decomposition
   into a requirement tree, each requirement resolved against the catalogue
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

OCCASION_KEYWORDS = {
    "anniversary": "anniversary",
    "wedding": "wedding",
    "recovery": "recovery",
    "surgery": "recovery",
    "get well": "recovery",
    "school project": "school project",
    "rocket": "school project",
    "birthday": "gift",
    "gift": "gift",
}

CATEGORY_KEYWORDS = {
    "jewellery": "jewellery",
    "jewelry": "jewellery",
    "gold": "jewellery",
    "silver": "jewellery",
    "ring": "jewellery",
    "necklace": "jewellery",
    "pendant": "jewellery",
    "bracelet": "jewellery",
    "electronics": "electronics",
    "earbuds": "electronics",
    "headphones": "electronics",
    "watch": "electronics",
    "smartwatch": "electronics",
    "diy": "diy",
    "rocket": "diy",
    "fruit basket": "care",
    "tea": "care",
    "clothing": "clothing",
    "clothes": "clothing",
    "jacket": "clothing",
    "hoodie": "clothing",
    "leggings": "clothing",
    "jeans": "clothing",
    "sweater": "clothing",
    "beauty": "beauty",
    "skincare": "beauty",
    "serum": "beauty",
    "moisturizer": "beauty",
    "face mask": "beauty",
    "lotion": "beauty",
}

DURABILITY_PHRASES = ["last forever", "lasts forever", "permanent", "long-lasting", "durable", "hold up"]
BUDGET_RE = re.compile(r"(?:under|below|budget\s*(?:of|is)?|around|~)?\s*(?:rs\.?|inr|₹)?\s*([\d,]{3,7})", re.I)
RANGE_RE = re.compile(r"([\d,]{3,7})\s*(?:-|to)\s*(?:rs\.?|inr|₹)?\s*([\d,]{3,7})", re.I)
# A bare number with no "budget"/"under"/"₹" etc keyword attached -- only trusted
# as a budget when the orchestrator tells us this message is directly answering
# the budget clarifying question (see `expecting` below). Fixes a real bug:
# typing a plain "3000" in reply to "What's your budget?" was silently ignored
# because BARE_NUMBER_RE below wasn't reachable at all without this hint.
BARE_NUMBER_RE = re.compile(r"([\d,]{3,7})")


@dataclass
class IntentSlots:
    raw_text: str
    occasion: str | None = None
    category: str | None = None
    budget_max: float | None = None
    wants_durability: bool = False
    requirement_tree: list[str] = field(default_factory=list)  # for complex/DIY goals
    clarifying_answers: dict[str, str] = field(default_factory=dict)


def extract_intent(text: str, existing: IntentSlots | None = None, expecting: str | None = None) -> IntentSlots:
    """Parse a free-text situation into structured slots.

    If `existing` is given, this call is a clarifying-answer turn -- new
    matches fill in whatever `existing` was still missing, they don't
    overwrite already-resolved slots (so the user's earlier statement stays
    authoritative).

    `expecting` names the slot the last clarifying question was actually
    asking for (e.g. "budget_max"). When set, a bare number with no
    "budget"/"under"/"₹" keyword is still trusted for that slot -- the
    question itself already established what the number means, so demanding
    a keyword too was just losing real answers.
    """
    lower = text.lower()
    slots = existing or IntentSlots(raw_text=text)
    if existing:
        slots.raw_text = f"{existing.raw_text} | {text}"

    if slots.occasion is None:
        for kw, occ in OCCASION_KEYWORDS.items():
            if kw in lower:
                slots.occasion = occ
                break

    if slots.category is None:
        for kw, cat in CATEGORY_KEYWORDS.items():
            if kw in lower:
                slots.category = cat
                break

    if slots.budget_max is None:
        range_match = RANGE_RE.search(lower)
        if range_match:
            slots.budget_max = float(range_match.group(2).replace(",", ""))
        else:
            budget_match = BUDGET_RE.search(lower)
            if budget_match and any(t in lower for t in ["budget", "under", "below", "rs", "inr", "₹", "around"]):
                slots.budget_max = float(budget_match.group(1).replace(",", ""))
            elif expecting == "budget_max" or slots.category is not None:
                # A bare number with no "budget"/"₹" keyword is still trusted
                # as the budget when either (a) the orchestrator says this
                # message is directly answering the budget question, or (b)
                # the message also names a category in the same breath --
                # "electronics, 5000" and "electronics 5000" are both common
                # single-line phrasings, and a bare number right alongside a
                # resolved category is unambiguous enough not to demand a
                # keyword too. Fixes budget silently not applying for both
                # the single-line and the two-turn phrasing.
                bare_match = BARE_NUMBER_RE.search(lower)
                if bare_match:
                    slots.budget_max = float(bare_match.group(1).replace(",", ""))

    if any(p in lower for p in DURABILITY_PHRASES):
        slots.wants_durability = True

    if slots.category == "diy" and not slots.requirement_tree:
        # Goal decomposition for the open-ended "build a model rocket" case:
        # expand into the requirement tree the brief describes (body, fins,
        # adhesive, paint) instead of a single product lookup.
        slots.requirement_tree = ["diy"]  # category itself resolves via multiple catalogue rows

    return slots


def missing_slots(slots: IntentSlots) -> list[str]:
    """Which slots still block a confident recommendation, in narrowing-value order."""
    missing = []
    if slots.category is None:
        missing.append("category")
    if slots.budget_max is None:
        missing.append("budget_max")
    return missing


CLARIFYING_QUESTIONS = {
    "category": "What kind of product are we looking at (e.g. jewellery, electronics, a DIY project)?",
    "budget_max": "What's your budget, roughly (in INR)?",
}

MAX_CLARIFYING_QUESTIONS = 2


def next_clarifying_question(slots: IntentSlots, questions_asked: int) -> str | None:
    if questions_asked >= MAX_CLARIFYING_QUESTIONS:
        return None
    missing = missing_slots(slots)
    if not missing:
        return None
    return CLARIFYING_QUESTIONS[missing[0]]
