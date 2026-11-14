"""Risk-signal weights, loaded from data/references/risk_weights.json so every number has a recorded source.

Each weight is one of:
  fitted      points = (20 / ln 2) * ln(likelihood ratio) from labelled data (scripts/calibrate_url_weights.py)
  ahp         relative importance from Analytic Hierarchy Process pairwise judgments by several raters
              (scripts/ahp_aggregate.py), scaled to points using the data-fitted signals as the unit anchor
  provisional the original hand-set value, used until the AHP ratings have been collected
The file also records the unit (20 points = odds doubled) and the High-risk threshold (40 points = odds x4)."""
from __future__ import annotations

import json
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "data" / "references" / "risk_weights.json"

DEFAULTS = {"unverified_source": 40, "dead_link": 45, "review_flagged": 40, "thin_ratings": 25, "no_rating_evidence": 40,
            "too_good_few": 15, "seller_unknown": 20, "seller_unproven": 25, "extreme_discount": 15, "price_too_low": 25,
            "no_rating": 10, "no_price": 10, "low_rating": 10, "over_budget": 5, "prompt_injection": 60, "not_product_page": 10,
            "wrong_region": 10, "insecure_url": 30, "odd_url": 60, "lookalike_domain": 45, "risky_tld": 15, "shortener": 14}
DEFAULT_SOURCE = {"odd_url": "fitted", "shortener": "fitted"}


def _load() -> dict:
    try:
        d = json.loads(FILE.read_text(encoding="utf-8"))
        w = {k: float(v["points"]) for k, v in d["weights"].items()}
        if set(w) >= set(DEFAULTS):
            return d
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {"status": "provisional", "threshold": 40,
            "weights": {k: {"points": v, "source": DEFAULT_SOURCE.get(k, "provisional")} for k, v in DEFAULTS.items()}}


_DATA = _load()


def points() -> dict[str, int]:
    return {k: int(round(v["points"])) for k, v in _DATA["weights"].items()}


def threshold() -> int:
    return int(_DATA.get("threshold", 40))


def provenance() -> dict:
    """How each weight was obtained, for the dashboard and the methods note."""
    src = {}
    for k, v in _DATA["weights"].items():
        src.setdefault(v.get("source", "provisional"), []).append(k)
    return {"status": _DATA.get("status", "provisional"), "threshold": threshold(), "sources": src, "meta": _DATA.get("meta", {})}
