from __future__ import annotations

from app import envload as _envload

_envload.load()

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import audit_log, autonomous, risk_lab, web_assistant, web_checkout, behaviour_experiment, catalog, payments
from app.aside_client import get_decision_policy
from app.dashboard_service import build_dashboard
from app.orchestrator import handle_turn
from app.review_authenticity import check as check_review_authenticity
from app.session_store import create_session, get_session
from app.storefront import ingest_from_storefront

app = FastAPI(
    title="Agentra Shopping Agent",
    version="0.1.0",
    description=(
        "Option A demo: intent capture -> catalogue retrieval -> ASIDE Pre-Transaction "
        "Gate, composed as a pipeline (see conversational-commerce brief §6)."
    ),
)


class StartSessionResponse(BaseModel):
    session_id: str
    greeting: str


class MessageRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "agentra-shopping-agent"}


@app.get("/system/aside-status")
def aside_status():
    policy = get_decision_policy()
    return {"connected": policy is not None, "policy": policy}


@app.get("/system/data-status")
def data_status():
    """Provenance of the catalogue the agent is buying from (real vs mock)."""
    return catalog.status()


@app.get("/system/payment-mode")
def payment_mode():
    return {"mode": payments.mode(), "hard_cap_inr": payments.hard_cap()}


@app.get("/enterprise/behaviour-profile")
def behaviour_profile(n: int = 400, seed: int = 0, policy: str = "agentra"):
    """policy: agentra (rule-based) | llm (needs GROQ_API_KEY or ANTHROPIC_API_KEY, capped at 200 pairs) | first | cheapest."""
    if policy == "llm":
        from app import llm_policy
        if not llm_policy.available():
            raise HTTPException(status_code=400, detail="No LLM key set: put GROQ_API_KEY (or ANTHROPIC_API_KEY) in .env, never in chat.")
        try:
            return behaviour_experiment.profile(llm_policy.llm_policy, n=min(max(n, 30), 200), seed=seed,
                                                policy_name=f"llm:{llm_policy.model()}")
        except llm_policy.LLMUnavailable as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    if policy not in behaviour_experiment.POLICIES:
        raise HTTPException(status_code=400, detail=f"unknown policy {policy!r}")
    return behaviour_experiment.profile(behaviour_experiment.POLICIES[policy], n=min(max(n, 30), 5000),
                                        seed=seed, policy_name=policy)


@app.get("/enterprise/price-tilt")
def price_tilt(policy: str = "agentra", n: int = 300, seed: int = 0):
    """Agent price percentile vs the OPeRA human browsing baseline (same two numbers)."""
    if policy not in behaviour_experiment.POLICIES:
        raise HTTPException(status_code=400, detail=f"unknown policy {policy!r}")
    return behaviour_experiment.tilt_comparison(policy, n=min(max(n, 30), 2000), seed=seed)


@app.get("/audit/verify")
def audit_verify():
    """Re-compute the hash chain of the purchase audit log."""
    return audit_log.verify()


@app.post("/session/start", response_model=StartSessionResponse)
def start_session():
    state = create_session()
    return StartSessionResponse(
        session_id=state.session_id,
        greeting="Tell me what you're shopping for -- who it's for, the occasion, your budget.",
    )


@app.post("/session/message")
def post_message(req: MessageRequest):
    state = get_session(req.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Start a new session.")
    result = handle_turn(state, req.message)
    return {
        "reply": result.reply,
        "stage": result.stage,
        "products": result.products,
        "gate": asdict(result.gate) if result.gate else None,
        "gates": result.gates,  # every product evaluated THIS turn -- supports checking/buying several at once
        "trajectory": result.trajectory,
        "dual_layer": result.dual_layer,
        "restarted": result.restarted,
    }


@app.get("/enterprise/dashboard")
def enterprise_dashboard(page: int = 1, page_size: int = 20):
    """Aggregated behavioral profiles, bias trends, and audit rollup across
    every purchase evaluated in every in-memory session -- the 'Govern'
    stage of the pipeline. Summary stats (avg risk, decision breakdown,
    etc.) cover ALL evaluated purchases; `recent_profiles` is paginated via
    `page`/`page_size` since a session can now have many. Backs
    static/dashboard.html."""
    return build_dashboard(page=page, page_size=page_size)


@app.get("/consumer/decision/{session_id}")
def consumer_decision(session_id: str):
    """The consumer-facing half of the dual-layer output for one session --
    what the embeddable Consumer Widget (static/widget.js) fetches and
    renders at checkout time."""
    state = get_session(session_id)
    if state is None or state.trajectory is None or state.trajectory.consumer_view is None:
        raise HTTPException(status_code=404, detail="No completed decision for this session yet.")
    return {
        "consumer_view": state.trajectory.consumer_view,
        "gate_decision": state.trajectory.gate_decision,
        "final_selection": state.trajectory.final_selection,
    }


@app.get("/storefront/listings")
def storefront_listings():
    """The mock external storefront's own product+review feed -- what a real
    Catalogue Ingestion Layer would crawl/parse. Also runs the Review
    Authenticity check (app/review_authenticity.py) per listing so
    static/storefront.html can show, right on the product page, which
    listings look organically reviewed and which don't -- the same
    transparency principle as the rest of Agentra's disclosed-signal design."""
    listings = []
    for product in ingest_from_storefront():
        authenticity = check_review_authenticity(product)
        listings.append({
            "id": product.id, "name": product.name, "price": product.price,
            "merchant": product.merchant, "is_new_merchant": product.is_new_merchant,
            "promoted": product.promoted, "brand_tier": product.brand_tier,
            "review_score": product.review_score, "reviews": product.reviews,
            "review_authenticity": authenticity.as_dict(),
        })
    return {"listings": listings}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    # The product is the assistant + the dashboard. The old chat page (static/index.html) is kept on disk, not served.
    return RedirectResponse("/assistant")


@app.post("/enterprise/sample-runs")
def enterprise_sample_runs(n: int = 12, seed: int = 0):
    """Run the catalogue pipeline on real catalogue products so the research panels fill (in memory, cleared on restart)."""
    from app import sample_runs
    return sample_runs.run(n=n, seed=seed)


@app.get("/enterprise/risk-weights")
def risk_weights_provenance():
    """Where every risk-signal weight comes from (fitted / ahp / provisional), the unit and the High-risk threshold."""
    from app import risk_weights
    return {**risk_weights.provenance(), "points": risk_weights.points()}


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    return FileResponse("static/dashboard.html")


@app.get("/agent", include_in_schema=False)
def agent_page():
    # Retired page (static/agent.html kept on disk). The API behind it (POST /agent/run) still works for tests/scripts.
    return RedirectResponse("/assistant")


@app.get("/widget-demo", include_in_schema=False)
def widget_demo_page():
    return FileResponse("static/widget-demo.html")


@app.get("/storefront", include_in_schema=False)
def storefront_page():
    # bias_testbed_storefront.html (MartVerse) lives at the repo root -- it's
    # a standalone artifact openable directly in a browser with no server at
    # all, and app/storefront.py's ingestion reads the exact same file, so
    # serving it here too means the in-app link and a bare double-click show
    # byte-identical content.
    return FileResponse("bias_testbed_storefront.html")


class AgentRunRequest(BaseModel):
    goal: str
    max_candidates: int = 5


@app.post("/agent/run")
def agent_run(req: AgentRunRequest):
    """Autonomous mode: goal in, verified order (or an honest stop) out."""
    return autonomous.run(req.goal, max(1, min(req.max_candidates, 10)))


@app.get("/agent/orders")
def agent_orders():
    return {"payment_mode": payments.mode(), "orders": payments.issued_orders()}


@app.get("/agent/orders/{order_id}/verify")
def agent_verify_order(order_id: str):
    order = next((o for o in payments.issued_orders() if o["order_id"] == order_id), None)
    if order is None:
        raise HTTPException(status_code=404, detail="Unknown order_id for this server run.")
    return payments.verify_order(order)


class AssistantRequest(BaseModel):
    message: str
    session_id: str | None = None


@app.post("/assistant/message")
def assistant_message(req: AssistantRequest):
    """Conversational web shopping assistant (Groq + web search) with Agentra's evaluation layer."""
    try:
        return web_assistant.converse(req.session_id, req.message)
    except web_assistant.AssistantUnavailable as exc:
        msg = str(exc)
        code = 400 if "not set" in msg else 429 if "HTTP 429" in msg else 503 if "did not answer in time" in msg else 502
        raise HTTPException(status_code=code, detail=msg)


class BuyRequest(BaseModel):
    session_id: str
    rank: int = 0
    url: str | None = None


class DecisionRequest(BaseModel):
    order_id: str
    approve: bool


def _lookup(fn, *a):
    try:
        return fn(*a)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


class DemoRequest(BaseModel):
    session_id: str | None = None


@app.post("/assistant/demo")
def assistant_demo(req: DemoRequest):
    """Risk lab: SYNTHETIC listings evaluated by the real risk checks and gate (no network, no model)."""
    return risk_lab.run(req.session_id)


@app.post("/assistant/buy")
def assistant_buy(req: BuyRequest):
    """Buy -> gate (Allow / Escalate / Deny) -> demo-wallet settlement. Simulated money; see app/web_checkout.py."""
    return _lookup(web_checkout.buy, req.session_id, req.rank, req.url)


@app.post("/assistant/approve")
def assistant_approve(req: DecisionRequest):
    """The human's Approve / Reject on an escalated order."""
    return _lookup(web_checkout.decide, req.order_id, req.approve)


@app.post("/assistant/orders/{order_id}/refund")
def assistant_order_refund(order_id: str):
    return _lookup(web_checkout.refund, order_id)


@app.get("/assistant/wallet")
def assistant_wallet():
    return {**web_checkout.wallet(), "orders": web_checkout.orders()[-10:], "decisions": web_checkout.gate_counts()}


@app.get("/assistant/log")
def assistant_log(limit: int = 100, session_id: str | None = None):
    """Activity log of the web assistant: every message, shortlist, link check and error, from the hash-chained audit log."""
    return web_assistant.activity_log(limit, session_id)


@app.get("/assistant", include_in_schema=False)
def assistant_page():
    return FileResponse("static/assistant.html")
