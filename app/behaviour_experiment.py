"""Two-alternative forced-choice (2AFC) behaviour experiment on REAL products.

Method follows the ACES / Cherep et al. design: draw two real products from the
same category, randomise which one is shown first, ask a *policy* (the agent
under test) to pick, then measure how much each attribute moves the choice.

Attribute sensitivity (pp) for attribute X = 50 * (P(A chosen | A better on X)
- P(A chosen | A worse on X)) in percentage points -- i.e. how far the choice
probability swings (per side) when X flips in favour of an option. Averaged
over attributes (price, rating). Position bias = P(first shown chosen) - 0.5.
Bootstrap 95% CIs. This is an approximation of the published LPM metric; the
exact regression variant should be validated before quoting numbers as equal.

No synthetic data: pairs come from ``app.catalog`` (real when
CATALOG_SOURCE=real). Policies are plain callables so any agent (rule-based,
LLM) can be plugged in.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from app import catalog
from app.intent import IntentSlots
from app.product import Product
from app import retrieval

Policy = Callable[[Product, Product], int]  # returns 0 (first shown) or 1 (second)


def agentra_policy(first: Product, second: Product) -> int:
    """The Agentra ranker as a chooser (budget set just above the pricier item)."""
    slots = IntentSlots(raw_text="", category=first.category,
                        budget_max=max(first.price, second.price) * 1.1)
    s0 = retrieval.total_score(retrieval.signal_contributions(first, slots))
    s1 = retrieval.total_score(retrieval.signal_contributions(second, slots))
    return 0 if s0 >= s1 else 1


def always_first(first: Product, second: Product) -> int:
    return 0


def cheapest(first: Product, second: Product) -> int:
    return 0 if first.price <= second.price else 1


def make_pairs(products: list[Product], n: int, seed: int = 0) -> list[tuple[Product, Product]]:
    rng = random.Random(seed)
    by_cat: dict[str, list[Product]] = {}
    for p in products:
        by_cat.setdefault(p.category, []).append(p)
    cats = [c for c, v in by_cat.items() if len(v) >= 2]
    pairs = []
    for _ in range(n if cats else 0):
        a, b = rng.sample(by_cat[rng.choice(cats)], 2)
        pairs.append((a, b))
    return pairs


@dataclass
class Trial:
    price_a_better: bool | None
    rating_a_better: bool | None
    chose_a: bool
    chose_first: bool


def run(policy: Policy, products: list[Product] | None = None, n: int = 400, seed: int = 0) -> list[Trial]:
    products = products if products is not None else catalog.CATALOGUE
    rng = random.Random(seed + 1)
    trials: list[Trial] = []
    for a, b in make_pairs(products, n, seed):
        a_first = rng.random() < 0.5            # randomised presentation order
        first, second = (a, b) if a_first else (b, a)
        pick = policy(first, second)
        chose_a = (pick == 0) == a_first
        trials.append(Trial(
            price_a_better=None if a.price == b.price else a.price < b.price,
            rating_a_better=None if a.review_score == b.review_score else a.review_score > b.review_score,
            chose_a=chose_a, chose_first=(pick == 0)))
    return trials


def _swing(trials: list[Trial], attr: str) -> float | None:
    better = [t.chose_a for t in trials if getattr(t, attr) is True]
    worse = [t.chose_a for t in trials if getattr(t, attr) is False]
    if not better or not worse:
        return None
    return 50.0 * (sum(better) / len(better) - sum(worse) / len(worse))


def sensitivities(trials: list[Trial]) -> dict:
    vals = {a: _swing(trials, f"{a}_a_better") for a in ("price", "rating")}
    used = [abs(v) for v in vals.values() if v is not None]
    avg = sum(used) / len(used) if used else 0.0
    pos = (sum(t.chose_first for t in trials) / len(trials) - 0.5) * 100 if trials else 0.0
    return {"price_pp": vals["price"], "rating_pp": vals["rating"],
            "avg_attribute_sensitivity_pp": avg, "position_bias_pp": pos, "n": len(trials)}


def bootstrap_ci(trials: list[Trial], reps: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        sample = [rng.choice(trials) for _ in trials]
        stats.append(sensitivities(sample)["avg_attribute_sensitivity_pp"])
    stats.sort()
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


POLICIES = {"agentra": agentra_policy, "first": always_first, "cheapest": cheapest}


def profile(policy: Policy = agentra_policy, n: int = 400, seed: int = 0, policy_name: str = "agentra") -> dict:
    """Run the experiment and compare to the reference points."""
    from app import reference_baseline as rb
    trials = run(policy, n=n, seed=seed)
    if not trials:
        return {"status": "no_data", "detail": "Catalogue has no category with >= 2 products.",
                "data_source": catalog.SOURCE}
    s = sensitivities(trials)
    lo, hi = bootstrap_ci(trials, seed=seed)
    ref = rb.load_reference()
    return {"policy": policy_name, "data_source": catalog.SOURCE, "synthetic": catalog.SOURCE == "mock",
            **s, "ci95_pp": [round(lo, 2), round(hi, 2)],
            "assessment": rb.assess(s["avg_attribute_sensitivity_pp"], s["n"], human_pp=rb.human_value(ref)),
            "reference_source": (ref or {}).get("source")}


# ---------------------------------------------------------------- price tilt (like-for-like with the OPeRA human baseline)
def _choose_from(policy: Policy, options: list[Product]) -> Product:
    """Pairwise policy -> pick from a bigger set by knockout (winner stays on)."""
    best = options[0]
    for nxt in options[1:]:
        best = best if policy(best, nxt) == 0 else nxt
    return best


def price_tilt(policy: Policy = agentra_policy, products: list[Product] | None = None, n: int = 300,
               k: int = 5, seed: int = 0) -> dict:
    """Same two numbers as data/references/opera_human_sensitivity.json: price percentile of the chosen item within the
    set it chose from (0 = cheapest, 0.5 = no price tilt) and the share of choices cheaper than the set median."""
    import statistics
    products = products if products is not None else catalog.CATALOGUE
    rng = random.Random(seed)
    by_cat: dict[str, list[Product]] = {}
    for p in products:
        by_cat.setdefault(p.category, []).append(p)
    cats = [c for c, v in by_cat.items() if len(v) >= 3]
    if not cats:
        return {"status": "no_data", "detail": "Need a category with at least 3 products."}
    pct, cheap = [], []
    for _ in range(n):
        pool = by_cat[rng.choice(cats)]
        opts = rng.sample(pool, min(k, len(pool)))
        rng.shuffle(opts)
        pick = _choose_from(policy, opts)
        prices = sorted(o.price for o in opts)
        pct.append(sum(x < pick.price for x in prices) / (len(prices) - 1))
        cheap.append(pick.price < statistics.median(prices))
    reps_p, reps_c = [], []
    for _ in range(1000):
        idx = [rng.randrange(len(pct)) for _ in pct]
        reps_p.append(sum(pct[i] for i in idx) / len(idx))
        reps_c.append(sum(cheap[i] for i in idx) / len(idx))
    q = lambda v, f: sorted(v)[int(f * (len(v) - 1))]
    return {"status": "ok", "data_source": catalog.SOURCE, "synthetic": catalog.SOURCE == "mock", "n": n, "options_per_choice": k,
            "price_percentile_mean": [round(sum(pct) / n, 3), round(q(reps_p, .025), 3), round(q(reps_p, .975), 3)],
            "cheaper_than_median_rate": [round(sum(cheap) / n, 3), round(q(reps_c, .025), 3), round(q(reps_c, .975), 3)]}


def human_price_tilt() -> dict | None:
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "data" / "references" / "opera_human_sensitivity.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def tilt_comparison(policy_name: str = "agentra", n: int = 300, seed: int = 0) -> dict:
    """Dashboard summary: Agentra (guarded ranker) with and without a stated budget, the pre-guard ranker, controls, humans."""
    from app import price_tilt_study as pts, retrieval
    if policy_name != "agentra":
        pol = POLICIES[policy_name]
        return {"policy": policy_name, "agent": price_tilt(pol, n=n, seed=seed)}
    prods = catalog.CATALOGUE
    cell = lambda ch, reg: (pts.summarise(sum((pts.run(ch, prods, n, 5, seed + j, reg)[0] for j in range(3)), [])) or {}).get("price_percentile_mean")
    guarded = {reg: cell(pts._mk(), reg) for reg in ("none", "loose(1.1x max)")}
    live = (pts.summarise(pts.run(lambda o, b, r: pts.assistant_pick(o, b), prods, max(n // 3, 60), 5, seed, "none")[0]) or {}).get("price_percentile_mean")
    retrieval.PRICE_FIT_MODE = "spend_to_budget"
    try:
        legacy_fit = {reg: cell(pts._mk(), reg) for reg in ("none", "loose(1.1x max)")}
    finally:
        retrieval.PRICE_FIT_MODE = "flat"
    retrieval.GUARD_PRICE_DERIVED_TIER = False
    try:
        before = {reg: cell(pts._mk(), reg) for reg in ("none", "loose(1.1x max)")}
    finally:
        retrieval.GUARD_PRICE_DERIVED_TIER = True
    human = human_price_tilt()
    out = {"policy": "agentra", "data_source": catalog.SOURCE, "synthetic": catalog.SOURCE == "mock", "n": n, "options_per_choice": 5,
           "agent_no_budget": guarded["none"], "agent_with_budget": guarded["loose(1.1x max)"],
           "agent_before_guard": before, "agent_legacy_price_fit": legacy_fit, "live_assistant_ranking": live,
           "controls": {nm: cell(f, "none") for nm, f in pts.CONTROLS.items() if nm in ("always_first", "cheapest")},
           "human_opera": human,
           "caveat": "Browsing-trace tilt, not the Cherep 2AFC pp metric. Agent choices come from random same-category sets of the "
                     "catalogue; human choices from what each OPeRA shopper had seen. Directional comparison only. Full study: "
                     "data/references/price_tilt_study.json (python -m app.price_tilt_study)."}
    if human and human.get("last_search_page", {}).get("price_percentile_mean"):
        h = human["last_search_page"]["price_percentile_mean"]
        a = guarded["none"]
        mine = ("agent is essentially price-neutral with no stated budget (within 0.05 of 0.5)" if abs(a[0] - 0.5) < 0.05 else
                "agent leans to dearer items with no stated budget" if a[1] > 0.5 else "agent leans to cheaper items with no stated budget")
        theirs = ("humans lean cheaper" if h[2] < 0.5 else "humans lean dearer" if h[1] > 0.5 else "humans are price-neutral")
        out["reading"] = f"{mine}; {theirs}"
    out["status"] = "ok"
    return out
