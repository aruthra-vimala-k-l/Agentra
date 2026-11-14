"""SIMULATED validation of the price-tilt method. Everything in this module is synthetic and labelled so; it validates the
measurement, it is never evidence about real shoppers.

A. Planted-tilt recovery: a synthetic chooser with a KNOWN price preference (softmax over price percentile, strength beta) is
   run through the same pipeline on real catalogue sets. If the measured percentile tracks the planted one, the metric works.
B. Power: how many choices are needed before a true human-like tilt (~0.38) is distinguishable from 0.5.
C. Real human choice sets + synthetic quality: Agentra's rankers choose from the ACTUAL price lists OPeRA shoppers had (from
   data/references/opera_choice_sets.json, created by ``--export-sets``), with synthetic ratings whose correlation with price is
   set explicitly (rho = -0.4, 0, +0.4). This removes the "different choice-set process" limitation: same sets as humans.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from app import catalog, price_tilt_study as pts, retrieval
from app.product import Product

ROOT = Path(__file__).resolve().parent.parent


def softmax_chooser(beta: float):
    """beta > 0 prefers cheaper (percentile < 0.5); beta < 0 dearer; 0 = random."""
    def pick(opts, budget, rng):
        pool = [o for o in opts if budget is None or o.price <= budget]
        prices = sorted(o.price for o in opts)
        z = [sum(x < o.price for x in prices) / (len(prices) - 1) for o in pool]
        w = [math.exp(-beta * v) for v in z]
        return rng.choices(pool, weights=w, k=1)[0]
    return pick


def planted_recovery(n=600, seeds=(0, 1, 2), betas=(-3, -1.5, 0, 1.5, 3, 4.5)):
    prods = catalog.CATALOGUE
    rows = []
    for b in betas:
        t = sum((pts.run(softmax_chooser(b), prods, n, 5, s, "none")[0] for s in seeds), [])
        rows.append({"planted_beta": b, "measured_price_percentile": pts.summarise(t)["price_percentile_mean"]})
    return rows


def calibrate_beta(target=0.38, k=5, n=800):
    lo, hi = 0.0, 8.0
    prods = catalog.CATALOGUE
    for _ in range(12):
        mid = (lo + hi) / 2
        v = pts.summarise(pts.run(softmax_chooser(mid), prods, n, k, 0, "none")[0])["price_percentile_mean"][0]
        lo, hi = (mid, hi) if v > target else (lo, mid)
    return round((lo + hi) / 2, 2)


def power(beta, ns=(30, 60, 123, 250, 500), sims=300, k=5, seed=0):
    """P(95% CI of the mean percentile excludes 0.5) for a chooser with planted tilt beta, at sample size n."""
    prods = catalog.CATALOGUE
    rng = random.Random(seed)
    big, _ = pts.run(softmax_chooser(beta), prods, 4000, k, seed, "none")
    pct = [t[0] for t in big]
    out = {}
    for n in ns:
        hit = 0
        for _ in range(sims):
            s = [pct[rng.randrange(len(pct))] for _ in range(n)]
            m = statistics.mean(s)
            se = statistics.stdev(s) / math.sqrt(n)
            hit += abs(m - 0.5) > 1.96 * se
        out[str(n)] = round(hit / sims, 3)
    return out


def _product(i, price_usd, rating):
    return Product(id=f"o{i}", name=f"item{i}", category="opera", price=price_usd * 83.0, merchant="m", is_new_merchant=False,
                   review_score=rating, source="synthetic", review_count=200)


def real_sets_synthetic_quality(rhos=(-0.4, 0.0, 0.4), seed=0, max_sets=None):
    f = ROOT / "data" / "references" / "opera_choice_sets.json"
    if not f.exists():
        return {"status": "needs_export", "how": "py scripts/human_baseline_from_opera.py --export-sets"}
    sets = json.loads(f.read_text(encoding="utf-8"))["sets"][:max_sets]
    rng = random.Random(seed)
    res = {"status": "ok", "n_sets": len(sets), "label": "SIMULATED quality attributes on REAL human price sets", "by_rho": {}}
    human = [sum(x < s["prices_usd"][s["chosen"]] for x in s["prices_usd"]) / (len(s["prices_usd"]) - 1) for s in sets if len(s["prices_usd"]) > 1]
    res["humans_on_these_sets"] = pts.summarise([(h, h < 0.5) for h in human])
    for rho in rhos:
        per = {"ranker_flat_price_fit": [], "live_assistant_ranking": [], "cheapest": [], "uniform_random": []}
        for si, s in enumerate(sets):
            prices = s["prices_usd"]
            if len(prices) < 2:
                continue
            srt = sorted(prices)
            zs = [(sum(x < p for x in srt) / (len(srt) - 1) - 0.5) * 3.46 for p in prices]      # unit-variance-ish
            opts = [_product(i, p, min(5.0, max(1.0, 4.0 + 0.5 * (rho * z + math.sqrt(1 - rho * rho) * rng.gauss(0, 1)))))
                    for i, (p, z) in enumerate(zip(prices, zs))]
            for nm, ch in (("ranker_flat_price_fit", lambda o, b, r: pts.ranker_pick(o, b, rng=r)),
                           ("live_assistant_ranking", lambda o, b, r: pts.assistant_pick(o, b)),
                           ("cheapest", pts.CONTROLS["cheapest"]), ("uniform_random", pts.CONTROLS["uniform_random"])):
                try:
                    pick = ch(opts, None, rng)
                except Exception:
                    continue
                per[nm].append((sum(x < pick.price for x in sorted(o.price for o in opts)) / (len(opts) - 1), pick.price < statistics.median(o.price for o in opts)))
        res["by_rho"][str(rho)] = {k: (pts.summarise(v) or {}).get("price_percentile_mean") for k, v in per.items()}
    return res


if __name__ == "__main__":
    out = {"label": "SIMULATED / method validation only", "planted_recovery": planted_recovery()}
    beta = calibrate_beta()
    out["beta_for_human_like_tilt_0.38"] = beta
    out["power_at_planted_human_tilt"] = power(beta)
    out["real_sets_synthetic_quality"] = real_sets_synthetic_quality()
    path = ROOT / "data" / "references" / "price_tilt_validation.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("->", path)
