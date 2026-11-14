"""Turn AHP rater judgments into risk-signal weights (writes data/references/risk_weights.json).

Method (Saaty's Analytic Hierarchy Process):
  1. Each rater compares signals/families pairwise on the 1-9 scale (scripts/ahp_rater_form.html -> one JSON file per rater).
  2. Raters are combined by the element-wise geometric mean (the aggregation that keeps reciprocal consistency).
  3. Priorities = principal eigenvector; consistency ratio CR = ((lambda_max - n)/(n-1)) / RI(n). CR < 0.10 is acceptable.
  4. Global weight of a signal = family weight x within-family weight.
  5. Unit anchor: points = s * global weight, with s chosen so the two DATA-FITTED signals (odd_url 136, shortener 14 points,
     from scripts/calibrate_url_weights.py) are matched in log space. The script prints how well the experts' relative
     importance of those two agrees with the data (a check of the experts against data).
  6. Fitted signals keep their fitted points; every other weight is AHP-derived. Caps at 100.
Usage:  py scripts/ahp_aggregate.py ratings/*.json            (add --dry-run to print without writing)
"""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

FAMILIES = {
    "source": ["unverified_source", "dead_link", "not_product_page", "wrong_region"],
    "url": ["insecure_url", "odd_url", "lookalike_domain", "risky_tld", "shortener"],
    "seller": ["seller_unknown", "seller_unproven"],
    "evidence": ["review_flagged", "no_rating_evidence", "thin_ratings", "too_good_few", "low_rating", "no_rating"],
    "price": ["price_too_low", "extreme_discount", "no_price", "over_budget"],
    "agent": ["prompt_injection"],
}
FITTED = {"odd_url": 136.0, "shortener": 14.0}      # points from data (data/references/url_calibration.json)
RI = {1: 0, 2: 0, 3: .58, 4: .90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
OUT = Path(__file__).resolve().parent.parent / "data" / "references" / "risk_weights.json"


def matrix(items: list[str], judg: dict) -> np.ndarray:
    n = len(items)
    A = np.ones((n, n))
    for i, j in combinations(range(n), 2):
        v = judg[f"{items[i]}|{items[j]}"]
        A[i, j], A[j, i] = v, 1 / v
    return A


def priorities(A: np.ndarray) -> tuple[np.ndarray, float]:
    n = len(A)
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    cr = 0.0 if n <= 2 else ((vals[k].real - n) / (n - 1)) / RI[n]
    return w, float(cr)


def run(files: list[str], write: bool = True) -> dict:
    raters = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    groups = {"families": list(FAMILIES), **{f: s for f, s in FAMILIES.items() if len(s) > 1}}
    crs, W = {}, {}
    for g, items in groups.items():
        key = "families" if g == "families" else g
        mats = [matrix(items, r[key]) for r in raters]
        for r, M in zip(raters, mats):
            crs.setdefault(g, {})[r.get("rater", "?")] = round(priorities(M)[1], 3)
        geo = np.exp(np.mean([np.log(M) for M in mats], axis=0))
        w, cr = priorities(geo)
        W[g] = dict(zip(items, w))
        crs[g]["group"] = round(cr, 3)
    glob = {}
    for fam, sigs in FAMILIES.items():
        for s in sigs:
            glob[s] = W["families"][fam] * (W[fam][s] if len(sigs) > 1 else 1.0)
    logs = [math.log(FITTED[s] / glob[s]) for s in FITTED]
    scale = math.exp(sum(logs) / len(logs))
    exp_ratio = glob["odd_url"] / glob["shortener"]
    agreement = {"experts_odd_url_over_shortener": round(exp_ratio, 1), "data_odd_url_over_shortener": round(FITTED["odd_url"] / FITTED["shortener"], 1)}
    weights = {}
    for s, g in glob.items():
        if s in FITTED:
            weights[s] = {"points": min(100, round(FITTED[s])), "source": "fitted", "ahp_implied": round(min(100, scale * g), 1)}
        else:
            weights[s] = {"points": min(100, round(scale * g)), "source": "ahp"}
    bad = {g: v for g, d in crs.items() for r, v in d.items() if v > 0.10}
    res = {"status": "ahp", "threshold": 40,
           "unit": "20 points = odds of an unsafe listing doubled; High risk at 40 points = odds x4",
           "weights": weights,
           "meta": {"raters": len(raters), "consistency_ratio": crs, "cr_above_0.10": bad, "expert_vs_data": agreement,
                    "scale_points_per_unit_weight": round(scale, 1), "family_weights": {k: round(float(v), 3) for k, v in W["families"].items()}}}
    print(json.dumps(res["meta"], indent=1))
    if bad:
        print("WARNING: inconsistent judgments (CR > 0.10) in:", bad, "-> ask those raters to revisit, then re-run.")
    if write:
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", OUT)
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    run(args, write="--dry-run" not in sys.argv)
