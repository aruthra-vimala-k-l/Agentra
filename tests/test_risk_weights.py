import itertools
import json
import random

from app import risk_weights
from scripts import ahp_aggregate as ahp


def test_every_signal_has_a_weight_and_a_source():
    p = risk_weights.provenance()
    assert set(risk_weights.points()) == set(risk_weights.DEFAULTS)
    assert sum(len(v) for v in p["sources"].values()) == len(risk_weights.DEFAULTS)
    assert risk_weights.threshold() == 40                       # 40 points = odds x4 (20 points double the odds)


def test_ahp_recovers_planted_weights_and_checks_consistency(tmp_path):
    base = risk_weights.DEFAULTS
    fam = {"source": 50, "url": 40, "seller": 30, "evidence": 45, "price": 25, "agent": 60}
    rng, files = random.Random(3), []
    for k in range(3):
        r = {"rater": f"r{k}", "families": {f"{a}|{b}": min(9, max(1 / 9, fam[a] / fam[b] * rng.uniform(.9, 1.1)))
                                           for a, b in itertools.combinations(ahp.FAMILIES, 2)}}
        for f, s in ahp.FAMILIES.items():
            if len(s) > 1:
                r[f] = {f"{a}|{b}": min(9, max(1 / 9, base[a] / base[b] * rng.uniform(.9, 1.1))) for a, b in itertools.combinations(s, 2)}
        fp = tmp_path / f"r{k}.json"
        fp.write_text(json.dumps(r))
        files.append(str(fp))
    res = ahp.run(files, write=False)
    assert not res["meta"]["cr_above_0.10"]                     # consistent raters pass the CR < 0.10 check
    w = {k: v["points"] for k, v in res["weights"].items()}
    assert w["dead_link"] > w["no_rating"] and w["prompt_injection"] > w["over_budget"]   # planted ordering is recovered
    assert res["weights"]["odd_url"]["source"] == "fitted" and res["weights"]["shortener"]["points"] == 14
