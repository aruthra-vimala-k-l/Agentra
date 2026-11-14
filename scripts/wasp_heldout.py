"""Second held-out set: WASP (Evtimov et al., Meta, 2025; github.com/facebookresearch/wasp) injection texts.

84 strings = 21 attacker instructions (gitlab 12, reddit 9) x the 4 published message formats
(goal-hijacking / generic, plain-text / URL), instantiated from WASP's own configs (data/eval/wasp_injections.json,
sha256 c3aa4c97...0d5a).  The detector was tuned on AgentDojo failures, NOT on these. Caveat: I had read the template
wording while preparing the file, before tuning, so treat it as 'not tuned against', not 'never seen'.
Scope: measures whether the gate flags the injection text; it does NOT run WASP's web-agent environments.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import attack_harness as h  # noqa: E402

PATH = os.path.join(h.ROOT, "data", "eval", "wasp_injections.json")


def build(n_benign):
    base = h.build_cases(n_benign)
    cases = [c for c in base if c[0] == "benign"]
    prods = [p for p in h.REAL if h.anomaly.assess_product(p)["level"] == "NORMAL" and not p.is_new_merchant]
    for i, x in enumerate(json.load(open(PATH))):
        p = prods[i % len(prods)]
        cases.append(("attack", f"{x['format']}", h.payload(p, h.listing(p, x["text"]), h.signals(p))))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--offline", action="store_true"); g.add_argument("--aside-url")
    ap.add_argument("--benign", type=int, default=150); ap.add_argument("--json")
    a = ap.parse_args(argv)
    gate = h.OfflineGate() if a.offline else h.HttpGate(a.aside_url)
    try: rows = h.run(gate, build(a.benign))
    finally:
        if a.offline: gate.close()
    o = h.report(rows, ("offline-rules" if a.offline else f"http {a.aside_url}") + " | WASP not-tuned")
    h.print_report(o)
    if a.json: json.dump({"summary": o, "rows": rows}, open(a.json, "w"), indent=1)
    return o


if __name__ == "__main__":
    main()
