"""Attribution fidelity of the Shapley decomposition (app/attribution.py) against the REAL ranker, on the REAL catalogue.

Protocol (intervention test, the standard faithfulness check for attributions):
  for every (query, recommended product) and every signal s:
      neutralise the *source attribute* of s (e.g. promoted=False, review_score=3.0, occasion_tags=[], budget removed),
      re-run the real scoring function (signal_contributions -> total_score, the same code retrieve_and_rank uses),
      actual_drop_s = score_before - score_after.
  Compare phi_s (Shapley) with actual_drop_s.
Reported: efficiency error, mean |phi - actual_drop|, Spearman(phi, actual_drop) per instance, top-1 agreement,
and the same numbers for two baselines (raw signal value = what a naive explanation shows; random permutation of phi),
plus rank-position impact of ablating the top vs the bottom attributed signal.
Notes: the ranker is additive + one synergy term, so Shapley is expected to be near-exact; the synergy (0.5) is split between
durability_match and occasion_match, so an intervention on one of them removes MORE than its phi -- that residual is the
only systematic gap and is reported, not hidden. This tests the attribution layer, not human comprehension.
"""
import json, math, os, random, statistics, sys
from dataclasses import replace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.attribution import attribute  # noqa: E402
from app.catalog import search  # noqa: E402
from app.intent import extract_intent  # noqa: E402
from app.retrieval import RANK_AFFECTING_SIGNALS, retrieve_and_rank, signal_contributions, total_score  # noqa: E402

QUERIES = [
    "a durable gold necklace for my mother's birthday under 3000",
    "a gift for my wife's anniversary, jewellery under 5000",
    "I need something that will last, wedding gift jewellery under 8000",
    "a sturdy drill set for home repairs under 4000",
    "diy tools for a weekend project under 2500",
    "durable hardware tools under 6000",
    "skin care for dry skin under 1500",
    "a gift set from the personal care category under 2000",
    "a long lasting hair care product under 1200",
    "wireless earbuds under 2500",
    "a durable phone accessory under 1500",
    "electronics gift for a birthday under 4000",
]
NEUTRAL = {
    "price_fit": lambda p, s: (p, replace(s, budget_max=None)),
    "durability_match": lambda p, s: (p, replace(s, wants_durability=False)),
    "occasion_match": lambda p, s: (replace(p, occasion_tags=[]), s),
    "review_influence": lambda p, s: (replace(p, review_score=3.0), s),
    "brand_preference": lambda p, s: (replace(p, brand_tier="standard"), s),
    "new_merchant_caution": lambda p, s: (replace(p, is_new_merchant=False), s),
    "nudge_exposure": lambda p, s: (replace(p, promoted=False), s),
}


def score(p, s): return total_score(signal_contributions(p, s))


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rk(a), rk(b); ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def rank_of(target_id, cands, s, focal_override=None):
    sc = []
    for p in cands:
        pp, ss = focal_override if (focal_override and p.id == target_id) else (p, s)
        sc.append((score(pp, ss), p.id))
    sc.sort(key=lambda x: -x[0])
    return [i for _, i in sc].index(target_id) + 1


def main():
    rnd = random.Random(0); inst = []; eff = []; rank_top = []; rank_bot = []; n_q = 0
    for q in QUERIES:
        s = extract_intent(q)
        top = retrieve_and_rank(s, top_k=3)
        if not top: continue
        n_q += 1
        cands = search(category=s.category, max_price=s.budget_max, occasion=s.occasion) or search(category=s.category, max_price=s.budget_max)
        for rp in top:
            p = rp.product; at = attribute(p, s)
            assert abs(at.total_score - score(p, s)) < 1e-6 and abs(at.total_score - rp.score) < 1e-6
            eff.append(abs(sum(at.shapley_values.values()) - at.total_score))
            base = score(p, s); sigs = list(at.shapley_values); phi = [at.shapley_values[k] for k in sigs]
            drop = []
            for k in sigs:
                pp, ss = NEUTRAL[k](p, s); drop.append(base - score(pp, ss))
            raw = [at.raw_signal_contributions[k] for k in sigs]; perm = phi[:]; rnd.shuffle(perm)
            inst.append({"phi": phi, "drop": drop, "raw": raw, "perm": perm, "sigs": sigs})
            ranked = sorted(sigs, key=lambda k: -abs(at.shapley_values[k]))
            nz = [k for k in ranked if abs(at.shapley_values[k]) > 1e-9]
            if len(nz) >= 2:
                for lst, k in ((rank_top, nz[0]), (rank_bot, nz[-1])):
                    rank_n = rank_of(p.id, cands, s, NEUTRAL[k](p, s))
                    lst.append(rank_n - rank_of(p.id, cands, s))
    def stats(key):
        mae = statistics.mean(abs(a - b) for i in inst for a, b in zip(i[key], i["drop"]))
        rhos = [r for i in inst if (r := spearman(i[key], i["drop"])) is not None]
        t1 = [max(range(len(i["drop"])), key=lambda j: abs(i["drop"][j])) == max(range(len(i[key])), key=lambda j: abs(i[key][j])) for i in inst if any(abs(d) > 1e-9 for d in i["drop"])]
        return {"mean_abs_error": round(mae, 4), "mean_spearman": round(statistics.mean(rhos), 3), "top1_agreement": round(sum(t1) / len(t1), 3)}
    syn = [abs(a - b) for i in inst for k, a, b in zip(i["sigs"], i["phi"], i["drop"]) if k in ("durability_match", "occasion_match")]
    other = [abs(a - b) for i in inst for k, a, b in zip(i["sigs"], i["phi"], i["drop"]) if k not in ("durability_match", "occasion_match")]
    out = {"queries_used": n_q, "instances(query x product)": len(inst), "signals_tested": list(NEUTRAL),
           "efficiency_max_abs_error": max(eff), "shapley": stats("phi"), "baseline_raw_signal_value": stats("raw"),
           "baseline_random_permutation": stats("perm"),
           "residual_on_synergy_pair_signals": round(statistics.mean(syn), 4) if syn else None,
           "residual_on_other_signals": round(statistics.mean(other), 4),
           "rank_change_when_top_signal_ablated(mean positions)": round(statistics.mean(rank_top), 3) if rank_top else None,
           "rank_change_when_bottom_signal_ablated(mean positions)": round(statistics.mean(rank_bot), 3) if rank_bot else None}
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1: json.dump(out, open(sys.argv[1], "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
