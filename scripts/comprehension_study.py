"""Comprehension-study kit (instrument + analysis). It does NOT contain results: that needs human participants.

  python scripts/comprehension_study.py stimuli  > data/eval/comprehension_stimuli.json   # real Agentra outputs, 2 conditions
  python scripts/comprehension_study.py analyse responses.csv                              # after you collect data
  python scripts/comprehension_study.py selftest                                           # checks the analysis code only

responses.csv columns: participant,condition(A|B),scenario,q1,q2,q3,q4,seconds,trust_1to7
 q1..q4 = 1 if answered correctly else 0 (answer key is in the stimuli file).
Condition A = the plain one-line justification the agent already shows. Condition B = Agentra's explanation:
the top Shapley signals with their share of the score + the risk tag. Questions are generated from the *real* attribution, so
the answer key is computed, not written by hand.
"""
import csv, json, math, os, random, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCENARIOS = ["a durable gold necklace for my mother's birthday under 3000", "a sturdy drill set for home repairs under 4000",
             "skin care for dry skin under 1500", "wireless earbuds under 2500"]


def stimuli():
    from app.attribution import attribute
    from app.intent import extract_intent
    from app.retrieval import retrieve_and_rank
    out = []
    for q in SCENARIOS:
        s = extract_intent(q); top = retrieve_and_rank(s, top_k=1)
        if not top: continue
        rp = top[0]; at = attribute(rp.product, s)
        pos = {k: v for k, v in at.shapley_values.items() if abs(v) > 1e-9}
        tot = sum(abs(v) for v in pos.values()) or 1
        lead = max(pos, key=lambda k: abs(pos[k]))
        out.append({"scenario": q, "product": rp.product.name[:90], "price": rp.product.price,
                    "condition_A_text": rp.justification,
                    "condition_B_text": "Why this was chosen: " + "; ".join(f"{k.replace('_', ' ')} {abs(v)/tot:.0%}" for k, v in sorted(pos.items(), key=lambda kv: -abs(kv[1]))),
                    "questions": [
                        {"id": "q1", "text": "Which factor mattered MOST in choosing this product?", "answer": lead.replace("_", " ")},
                        {"id": "q2", "text": "Did the seller's promotion/sponsorship change the ranking? (yes/no)", "answer": "no"},
                        {"id": "q3", "text": "Was price a factor in the choice? (yes/no)", "answer": "yes" if "price_fit" in pos else "no"},
                        {"id": "q4", "text": "Name one factor that did NOT affect the choice.", "answer": sorted(k.replace('_', ' ') for k in at.shapley_values if k not in pos) or ["(none)"]}]})
    return out


def welch(a, b):
    ma, mb, va, vb = statistics.mean(a), statistics.mean(b), statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b)); t = (ma - mb) / se if se else float("nan")
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    sp = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return {"mean_B": round(ma, 3), "mean_A": round(mb, 3), "t": round(t, 3), "df": round(df, 1), "cohens_d": round((ma - mb) / sp, 3) if sp else None}


def mann_whitney_p(a, b, iters=20000, seed=0):
    rnd = random.Random(seed); obs = abs(statistics.mean(a) - statistics.mean(b)); pool = list(a) + list(b); hit = 0
    for _ in range(iters):
        rnd.shuffle(pool); x, y = pool[:len(a)], pool[len(a):]
        hit += abs(statistics.mean(x) - statistics.mean(y)) >= obs
    return (hit + 1) / (iters + 1)  # permutation p-value


def analyse(path):
    rows = list(csv.DictReader(open(path))); by = {}
    for r in rows:
        score = sum(int(r[f"q{i}"]) for i in range(1, 5)) / 4
        by.setdefault((r["participant"], r["condition"]), []).append((score, float(r["seconds"]), float(r["trust_1to7"])))
    per = {k: tuple(statistics.mean(x[i] for x in v) for i in range(3)) for k, v in by.items()}
    res = {"participants": {c: sum(1 for k in per if k[1] == c) for c in "AB"}}
    for i, name in enumerate(["comprehension_score(0-1)", "seconds_per_scenario", "trust_1to7"]):
        A = [v[i] for k, v in per.items() if k[1] == "A"]; B = [v[i] for k, v in per.items() if k[1] == "B"]
        res[name] = {**welch(B, A), "permutation_p": round(mann_whitney_p(B, A), 4)}
    return res


def selftest():
    rnd = random.Random(1); path = "/tmp/_ct.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["participant", "condition", "scenario", "q1", "q2", "q3", "q4", "seconds", "trust_1to7"])
        for p in range(40):
            c = "AB"[p % 2]
            for s in range(4): w.writerow([p, c, s] + [int(rnd.random() < (0.8 if c == "B" else 0.5)) for _ in range(4)] + [rnd.randint(20, 60), rnd.randint(3, 7)])
    print("SELF-TEST ON SIMULATED DATA (checks the code only, not a result):"); print(json.dumps(analyse(path), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stimuli"
    if cmd == "stimuli": print(json.dumps(stimuli(), indent=1))
    elif cmd == "analyse": print(json.dumps(analyse(sys.argv[2]), indent=1))
    else: selftest()
