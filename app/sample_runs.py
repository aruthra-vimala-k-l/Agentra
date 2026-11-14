"""Run the catalogue pipeline (the ASIDE-gated agent: retrieval, irreversibility, bias audit, drift, counterfactual check) on real
catalogue products so the research panels of the dashboard have data. Each session is a genuine execution of the pipeline through
the orchestrator (no hand-written numbers); only the shopper messages are scripted. Sessions live in memory like any other and
are tagged user_id='sample-run'."""
from __future__ import annotations

import random

from app.orchestrator import handle_turn
from app.session_store import create_session

QUERIES = [
    "a silver necklace as a birthday gift for my sister, budget under 3000",
    "an anniversary gift, gold earrings, under 8000",
    "a cordless drill for home repairs, under 4000",
    "a toolkit for DIY work around the house under 2500",
    "a face wash and moisturiser for dry skin under 800",
    "a gentle shampoo for daily use under 600",
    "wireless earbuds for travel under 2000",
    "a power bank for my phone under 1500",
    "a smart watch for fitness under 5000",
    "a personalised engraved bracelet for my partner under 6000",
]


def run(n: int = 12, seed: int = 0) -> dict:
    n = max(1, min(int(n), 40))
    rng = random.Random(seed)
    done, skipped, decisions = 0, 0, {}
    for i in range(n):
        s = create_session(user_id="sample-run")
        try:
            r = handle_turn(s, QUERIES[(i + seed) % len(QUERIES)])
            ids = [p["id"] for p in (r.products or [])][:3]
            if not ids:
                skipped += 1
                continue
            r = handle_turn(s, f"I want to buy {rng.choice(ids)}")
            for g in (r.gates or []):
                d = g.get("decision") if isinstance(g, dict) else None
                if d:
                    decisions[d] = decisions.get(d, 0) + 1
            done += 1
        except Exception:                      # a bad session must not stop the batch
            skipped += 1
    return {"sessions_run": done, "skipped": skipped, "decisions": decisions,
            "note": "Real runs of the catalogue pipeline on catalogue products; shopper messages are scripted. In-memory: cleared on restart."}
