#!/usr/bin/env python3
"""Scripted live demo of Agentra against a RUNNING stack (Agentra :8100 + ASIDE :8000).

    python scripts/demo_run.py [--base http://localhost:8100]

Four scenes, printed as a run sheet you can narrate over:
  1  clean purchase      - shortlist -> gate ALLOW -> order created -> audit entry
  2  flagged purchase    - gate escalates -> human types "approve" -> order created
  3  blocked purchase    - high-risk item never reaches payment
  4  tamper test         - edit one audit line in a COPY of the log; the chain check goes red

Outcomes depend on the live ASIDE gate and on the catalogue, so the script reports what actually
happened and never fakes a result. Scenes that cannot be staged with the current data say so.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Demo:
    def __init__(self, base: str):
        self.c = httpx.Client(base_url=base, timeout=60)

    def say(self, msg: str, sid: str):
        r = self.c.post("/session/message", json={"session_id": sid, "message": msg})
        r.raise_for_status()
        return r.json()

    def session(self) -> str:
        return self.c.post("/session/start").json()["session_id"]


def hr(title: str):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def show(label: str, d: dict):
    print(f"  you> {label}\n  agent> {d['reply']}")
    for g in d.get("gates") or []:
        print(f"     gate: {g['product_id']} -> {g['decision']} (risk {g['risk_score']})")


def try_buy(demo: Demo, queries: list[str], want: set[str], max_tries: int = 6):
    """Search with each query and try shortlist items until the gate gives one of `want`."""
    for q in queries:
        sid = demo.session()
        d = demo.say(q, sid)
        for _ in range(2):  # allow answering one clarifying question
            if d["products"] or d["stage"] != "COLLECTING":
                break
            d = demo.say(q + " budget 5000", sid)
        for p in d["products"][:max_tries]:
            r = demo.say(f"buy {p['id']}", sid)
            decisions = {g["decision"] for g in r.get("gates") or []}
            if decisions & want:
                return sid, p, r
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8100")
    args = ap.parse_args()
    demo = Demo(args.base)

    hr("Preflight")
    ds = demo.c.get("/system/data-status").json()
    print(f"  catalogue: {ds['catalog_source']} | {ds['product_count']} products | "
          f"{ds['products_with_reviews']} with real reviews | synthetic={ds['synthetic']}")
    print(f"  categories: {', '.join(ds['categories']) or '--'}")
    print(f"  payment mode: {demo.c.get('/system/payment-mode').json()['mode']}")
    aside = demo.c.get("/system/aside-status").json()
    print(f"  ASIDE gate connected: {aside['connected']}")
    if ds["product_count"] == 0:
        print("  !! No products. Build the real catalogue first (see DATA.md).")
        return 1

    cats = ds["categories"]
    first = cats[0]

    hr("Scene 1 - clean purchase")
    sid, p, r = try_buy(demo, [f"{c} gift, budget 3000" for c in cats], {"ALLOW"})
    if r:
        print(f"  Shortlist item chosen: {p['name']} (₹{p['price']:,.0f})")
        print(f"  agent> {r['reply']}")
    else:
        print("  Could not find an item the gate allows outright (is ASIDE running and reachable?).")

    hr("Scene 2 - flagged purchase, human approval, then purchase")
    sid, p, r = try_buy(demo, [f"{c} budget 60000" for c in ["electronics"] + cats],
                        {"REQUIRE_HUMAN_APPROVAL", "REQUIRE_VERIFIABLE_APPROVAL"})
    if r:
        print(f"  agent> {r['reply']}")
        d = demo.say(f"approve {p['id']}", sid)
        print(f"  you> approve {p['id']}\n  agent> {d['reply']}")
    else:
        print("  No item was escalated by the gate with the current catalogue/ASIDE.")

    hr("Scene 3 - blocked before payment")
    sid, p, r = try_buy(demo, [f"{c} budget 500000" for c in cats], {"DENY"})
    if r:
        print(f"  agent> {r['reply']}")
        print("  -> no order was created for this item.")
    else:
        print("  The gate did not DENY any item here. Tip: restart with CATALOG_INCLUDE_TESTBED=1 to add the")
        print("  MartVerse bias-testbed listings (e.g. h-502 style dark patterns), or ask for a very high budget.")

    hr("Scene 4 - tamper test on a COPY of the audit log")
    chain = demo.c.get("/audit/verify").json()
    print(f"  live chain: valid={chain['valid']} entries={chain['entries']}")
    from app import audit_log
    src = audit_log.log_path()
    if not src.exists() or chain["entries"] == 0:
        print("  (no audit entries yet)")
        return 0
    tmp = Path(tempfile.mkdtemp()) / "purchases_copy.jsonl"
    shutil.copy(src, tmp)
    lines = tmp.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["payload"]["tampered"] = True
    lines[0] = json.dumps(entry, ensure_ascii=False)
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["AUDIT_LOG_PATH"] = str(tmp)
    res = audit_log.verify()
    print(f"  edited entry 1 in the copy -> valid={res['valid']} first_bad_seq={res.get('first_bad_seq')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
