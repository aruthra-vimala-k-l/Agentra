"""Price-tilt study: does the ranker systematically favour cheaper or dearer items, and WHY?

Metric (identical to scripts/human_baseline_from_opera.py --sensitivity): price percentile of the chosen item within the
choice set it was chosen from. 0 = always the cheapest, 0.5 = no price tilt, 1 = always the dearest.

Design (each step answers one threat to validity):
  * Choice sets: k same-category catalogue products drawn at random, so set composition does not depend on the ranker.
  * Budget regimes: the tilt of a budget-aware ranker depends on the budget it is told, so the budget is varied explicitly
    (none / loose / median / tight) instead of being hidden inside the experiment. Over-budget items are filtered exactly as
    ``retrieval.retrieve_and_rank`` filters them; trials left with < 2 affordable items are dropped and counted.
  * Ablation: each rank-affecting signal is switched off in turn; the change in tilt is that signal's contribution.
  * Confounding: within-category Spearman correlation of price with review score and with premium-brand status, because a
    ranker that rewards reviews or brand will inherit a price tilt from the catalogue even if it never reads price.
  * Controls: always-first (position only) ~0.5; cheapest = 0; dearest = 1; uniform random ~0.5.
  * Uncertainty: bootstrap 95% CI over choice sets, repeated over several seeds, reported with the seed spread.
  * Set size: k = 3, 5, 8.
Human reference: OPeRA (data/references/opera_human_sensitivity.json). Different data-generating processes (browsing traces
vs random catalogue sets), so the comparison is directional, and the metric is NOT the Cherep 2AFC percentage-point metric.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from app import catalog
from app.intent import IntentSlots
from app import retrieval

ROOT = Path(__file__).resolve().parent.parent
SIGNALS = ("price_fit", "review_influence", "brand_preference", "new_merchant_caution")
REGIMES = {"none": None, "loose(1.1x max)": 1.1, "median": "median", "tight(0.75x median)": 0.75}


def _by_cat(products):
    d: dict[str, list] = {}
    for p in products:
        d.setdefault(p.category, []).append(p)
    return {c: v for c, v in d.items() if len(v) >= 3}


def budget_for(regime: str, prices: list[float]) -> float | None:
    rule = REGIMES[regime]
    if rule is None:
        return None
    if rule == "median":
        return statistics.median(prices)
    if regime.startswith("tight"):
        return rule * statistics.median(prices)
    return rule * max(prices)


def ranker_pick(options, budget, drop: frozenset = frozenset(), rng=None):
    slots = IntentSlots(raw_text="", category=options[0].category, budget_max=budget)
    best, best_s = None, None
    for p in options:
        if budget is not None and p.price > budget:
            continue
        c = retrieval.signal_contributions(p, slots)
        s = sum(v for k, v in c.items() if k in retrieval.RANK_AFFECTING_SIGNALS and k not in drop)
        s += (rng.random() * 1e-9 if rng else 0)        # random tie-break, never "first in list"
        if best_s is None or s > best_s:
            best, best_s = p, s
    return best


def _pct(pick, options):
    prices = sorted(o.price for o in options)
    return sum(x < pick.price for x in prices) / (len(prices) - 1)


def run(chooser, products=None, n=400, k=5, seed=0, regime="none"):
    """chooser(options, budget, rng) -> Product | None. Returns list of percentiles + dropped count."""
    cats = _by_cat(products if products is not None else catalog.CATALOGUE)
    names = sorted(cats)
    rng = random.Random(seed)
    out, dropped = [], 0
    for _ in range(n):
        pool = cats[rng.choice(names)]
        opts = rng.sample(pool, min(k, len(pool)))
        rng.shuffle(opts)
        b = budget_for(regime, [o.price for o in opts])
        affordable = [o for o in opts if b is None or o.price <= b]
        if len(affordable) < 2:
            dropped += 1
            continue
        pick = chooser(opts, b, rng)
        out.append((_pct(pick, opts), pick.price < statistics.median(o.price for o in opts)))
    return out, dropped


def summarise(trials, reps=1000, seed=0):
    rng = random.Random(seed)
    pct = [t[0] for t in trials]
    chp = [t[1] for t in trials]
    if not pct:
        return None
    bp, bc = [], []
    for _ in range(reps):
        idx = [rng.randrange(len(pct)) for _ in pct]
        bp.append(sum(pct[i] for i in idx) / len(idx))
        bc.append(sum(chp[i] for i in idx) / len(idx))
    q = lambda v, f: sorted(v)[int(f * (len(v) - 1))]
    return {"n": len(pct), "price_percentile_mean": [round(sum(pct) / len(pct), 3), round(q(bp, .025), 3), round(q(bp, .975), 3)],
            "cheaper_than_median_rate": [round(sum(chp) / len(chp), 3), round(q(bc, .025), 3), round(q(bc, .975), 3)]}


def assistant_pick(options, budget, rng=None):
    """The LIVE assistant's deterministic ranking stage (web_assistant.evaluate: rating + budget fit + grounding - risk
    penalties) applied to catalogue products. Excludes only the model/web steps (no network in a study)."""
    from app import web_assistant
    cands = [{"name": o.name, "url": f"https://www.croma.com/p/{o.id}", "price_inr": float(o.price), "rating": float(o.review_score),
              "rating_count": o.review_count or 50, "link": {"state": "ok"}, "why": "Seller: Croma Retail"} for o in options]
    res = web_assistant.evaluate(cands, ["croma.com"], budget)["results"]
    ok = [r for r in res if budget is None or r["price_inr"] <= budget]
    top = ok[0]
    return next(o for o in options if o.name == top["name"] and float(o.price) == top["price_inr"])


def _mk(drop=frozenset()):
    return lambda opts, b, rng: ranker_pick(opts, b, drop, rng)


CONTROLS = {
    "always_first": lambda o, b, r: next((x for x in o if b is None or x.price <= b), o[0]),
    "uniform_random": lambda o, b, r: r.choice([x for x in o if b is None or x.price <= b] or o),
    "cheapest": lambda o, b, r: min((x for x in o if b is None or x.price <= b), key=lambda x: x.price),
    "dearest": lambda o, b, r: max((x for x in o if b is None or x.price <= b), key=lambda x: x.price),
}


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def confounding(products=None):
    cats = _by_cat(products if products is not None else catalog.CATALOGUE)
    res = {}
    for c, v in cats.items():
        res[c] = {"n": len(v),
                  "spearman_price_vs_review_score": _spearman([p.price for p in v], [p.review_score for p in v]),
                  "spearman_price_vs_premium_brand": _spearman([p.price for p in v], [1.0 if p.brand_tier == "premium" else 0.0 for p in v]),
                  "premium_share": round(sum(p.brand_tier == "premium" for p in v) / len(v), 3)}
    return res


def study(n=400, seeds=(0, 1, 2, 3, 4), ks=(3, 5, 8)):
    prods = catalog.CATALOGUE
    out: dict = {"data_source": catalog.SOURCE, "synthetic": catalog.SOURCE == "mock", "products": len(prods),
                 "n_per_cell": n, "seeds": list(seeds), "confounding": confounding(prods)}
    # 1. regimes x set size, full ranker, pooled over seeds + seed spread
    grid = {}
    for k in ks:
        for reg in REGIMES:
            per_seed, pooled, dropped = [], [], 0
            for s in seeds:
                t, d = run(_mk(), prods, n, k, s, reg)
                dropped += d
                pooled += t
                if t:
                    per_seed.append(round(sum(x[0] for x in t) / len(t), 3))
            grid[f"k={k}|{reg}"] = {**(summarise(pooled) or {}), "per_seed_means": per_seed, "trials_dropped_lt2_affordable": dropped}
    out["ranker_grid"] = grid
    # 1b. the ranker as it was BEFORE the guard (brand_tier = price rank allowed to score)
    retrieval.GUARD_PRICE_DERIVED_TIER = False
    try:
        out["ranker_before_guard_k5"] = {reg: summarise(sum((run(_mk(), prods, n, 5, s, reg)[0] for s in seeds), []))
                                         for reg in ("none", "loose(1.1x max)", "median")}
    finally:
        retrieval.GUARD_PRICE_DERIVED_TIER = True
    # 1c. legacy price_fit shape (spend-to-budget) for before/after
    retrieval.PRICE_FIT_MODE = "spend_to_budget"
    try:
        out["ranker_legacy_price_fit_k5"] = {reg: summarise(sum((run(_mk(), prods, n, 5, s, reg)[0] for s in seeds), []))
                                             for reg in ("none", "loose(1.1x max)", "median")}
    finally:
        retrieval.PRICE_FIT_MODE = "flat"
    # 1d. the live assistant's ranking stage
    out["live_assistant_ranking_k5"] = {reg: summarise(sum((run(lambda o, b, r: assistant_pick(o, b), prods, n, 5, s, reg)[0] for s in seeds), []))
                                        for reg in ("none", "loose(1.1x max)", "median")}
    # 1e. the metric must be set-size invariant for a price-blind chooser (placebo)
    out["placebo_uniform_random_by_k"] = {f"k={k}": summarise(sum((run(CONTROLS["uniform_random"], prods, n, k, s, "none")[0] for s in seeds), []))["price_percentile_mean"] for k in (3, 5, 8, 15)}
    # 2. ablation (k=5)
    abl = {}
    for reg in ("none", "loose(1.1x max)", "median"):
        base = summarise(sum((run(_mk(), prods, n, 5, s, reg)[0] for s in seeds), []))
        row = {"full_ranker": base}
        for sig in SIGNALS:
            r = summarise(sum((run(_mk(frozenset({sig})), prods, n, 5, s, reg)[0] for s in seeds), []))
            row[f"without_{sig}"] = {**r, "shift_vs_full": round(r["price_percentile_mean"][0] - base["price_percentile_mean"][0], 3)}
        abl[reg] = row
    out["ablation_k5"] = abl
    # 3. controls (k=5, no budget)
    out["controls_k5_none"] = {nm: summarise(sum((run(f, prods, n, 5, s, "none")[0] for s in seeds), [])) for nm, f in CONTROLS.items()}
    # 4. human reference
    hf = ROOT / "data" / "references" / "opera_human_sensitivity.json"
    out["human_opera"] = json.loads(hf.read_text(encoding="utf-8")) if hf.exists() else None
    return out


if __name__ == "__main__":
    import sys
    res = study(n=int(sys.argv[1]) if len(sys.argv) > 1 else 400, seeds=tuple(range(int(sys.argv[2]))) if len(sys.argv) > 2 else (0, 1, 2, 3, 4))
    path = ROOT / "data" / "references" / "price_tilt_study.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("confounding", "controls_k5_none")}, indent=1))
    print("\nLEGACY price_fit:", {k: v["price_percentile_mean"] for k, v in res["ranker_legacy_price_fit_k5"].items()})
    print("LIVE assistant ranking:", {k: v["price_percentile_mean"] for k, v in res["live_assistant_ranking_k5"].items()})
    print("placebo by k:", res["placebo_uniform_random_by_k"])
    print("\nBEFORE guard:", {k: v["price_percentile_mean"] for k, v in res["ranker_before_guard_k5"].items()})
    for reg, row in res["ablation_k5"].items():
        print("\n==", reg, "full:", row["full_ranker"]["price_percentile_mean"])
        for k, v in row.items():
            if k != "full_ranker":
                print("  ", k, v["price_percentile_mean"][0], "shift", v["shift_vs_full"])
    print("\n== grid")
    for k, v in res["ranker_grid"].items():
        print(k, v.get("price_percentile_mean"), "dropped", v["trials_dropped_lt2_affordable"])
    print("->", path)
