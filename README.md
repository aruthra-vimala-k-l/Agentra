# Agentra Shopping Agent — Option A (Capture + Explain layer)

# Agentra: final system (what to run and show)

**Two pages.** `/assistant` is the product; `/dashboard` is the governance view. `/` redirects to `/assistant`.

1. **Assistant (`/assistant`)**: Groq (`openai/gpt-oss-*`, `qwen/qwen3.8-27b`) converses; Tavily (or DuckDuckGo) finds real product pages; Agentra reads each page itself and evaluates it in code (grounded source, transparent score, review authenticity, URL/lookalike risk). Hard filters from what the user says: **budget** and **brand origin** (reviewed lookup; unknown brands are labelled, never guessed). Pasted links, with or without `https://`, get the same page check.
2. **Buy -> gate -> approve -> demo wallet.** Buy runs a local deterministic gate (`app/web_checkout.py`): **ALLOW** settles, **REQUIRE_HUMAN_APPROVAL** waits for Approve/Reject, **DENY** never settles. Rules: price known, https, working link, no instruction-like text aimed at an AI agent, per-purchase cap, daily cap, wallet balance, hard cap, human approval above a threshold, High-risk page, unconfirmed page, unknown merchant. **The wallet is simulated: no real money moves**; product links and prices are real. Balance is derived from the audit chain, so editing the log breaks the hash chain.
3. **Dashboard (`/dashboard`)**: wallet, orders, gate-decision counts, the assistant activity log (every message, shortlist, link check, gate decision, human decision, debit, refund, error) and the hash-chain integrity check. The lower "research modules" sections belong to the earlier agent pipeline.

Run: `uvicorn app.main:app --port 8100`, open `http://127.0.0.1:8100/assistant`. Keys live in `.env` only (`GROQ_API_KEY`, `TAVILY_API_KEY`). Check them first with `python scripts/groq_check.py`. Limits and caps: `DEMO_WALLET_START_INR`, `DEMO_PER_PURCHASE_CAP_INR`, `DEMO_DAILY_CAP_INR`, `DEMO_APPROVAL_ABOVE_INR`. Tests: `python -m pytest tests -q`.

**Bias controls.** A trusted marketplace is not evidence: a listing with no confirmed ratings is High risk even on Amazon; a star rating with fewer than 5 raters counts for nothing; a near-perfect rating from under 100 customers, an unknown or unproven third-party seller, an extreme "discount" and a price far below similar listings each add risk; Bestseller / Amazon's Choice / Sponsored badges are ignored and reported; High-risk items are also ranked lower.

**Honest limits.** The web gate is a rule set, not the external ASIDE service (that pipeline remains in the repo, behind `/agent/run`). Shops that block bots give no price, and the gate then refuses to pay. Reviews are a visible sample read by a model. A flag is a signal, not proof. Brand origin covers about 100 common brands. Payments are simulated.

---

# Earlier pipeline (ASIDE-based, API only)


This is the "Capture" half of Agentra's Option A shopping agent: intent
capture → catalogue retrieval → the ASIDE Pre-Transaction Gate ("Explain &
Secure"), composed as a pipeline rather than merged into one codebase. It
also now carries the standalone attribution, bias-detection, irreversibility
and dual-layer-output modules described below -- the "Explain" half of the
pipeline that previously only existed as a design in the proposal.

## What's new since the first Review 2 build

Added directly in response to a second round of review feedback ("build the
SHAP-style decomposition," "the Bias Detection Engine / Irreversibility
Classifier / Consumer Widget / Enterprise Dashboard are all absent," "I need
at least 1 patent claim fulfilled," "address the vulnerabilities"):

| Module | What it does | Patent claim it demonstrates |
|---|---|---|
| `app/attribution.py` | Exact Shapley-value decomposition of a product's ranking score into named commerce signals (price fit, durability match, occasion match, review influence, brand preference, nudge exposure, new-merchant caution), including a real nonlinear synergy term so the decomposition is a genuine game-theoretic computation, not a relabeling of raw weights. | Claim 1 |
| `app/bias_detection.py` | Standalone Bias Detection Engine: position-bias risk (is the #1 spot a decisive win or a near-tie decided by display order?), nudge-exposure audit (does a sponsored placement actually earn ranking weight?), choice-homogeneity (how much real variety is behind an apparently long shortlist?). | supports Claim 1 |
| `app/irreversibility.py` | Standalone Irreversibility Classifier, pulled out from being implicit inside ASIDE's amount thresholds -- an auditable, rule-based LOW/MEDIUM/HIGH/CRITICAL score with named factors. | supports Claims 1 & 2 |
| `app/behavioral_profile.py` | Dual-layer output: one attribution pass over the chosen product feeds *both* a consumer-facing plain-language explanation and an enterprise-facing behavioral profile (full signal decomposition + bias audit + drift). | Claim 2 |
| `app/drift.py` | A narrow, honestly-scoped behavioral drift tracker: per-user Shapley-value history, flags when a signal's average weight shifts sharply across a session. | a starting slice of Claim 3 |
| `app/dashboard_service.py` + `static/dashboard.html` | Enterprise Dashboard: decision breakdown, average risk, irreversibility distribution, bias/drift flag counts, recent behavioral profiles. | the "Govern" stage |
| `static/widget.js` + `static/widget-demo.html` | Embeddable Consumer Widget -- a two-line integration a third-party checkout page could drop in, separate from the full chat UI. | consumer-facing half of Claim 2 |

See `PATENT_CLAIMS.md` for exactly what is and isn't demonstrated, and why
the Counterfactual-Divergence-Comparator (CDC) internals were deliberately
**not** built here. See `SECURITY_NOTES.md` for what was fixed in this pass
and what's still open.

## Why two repos, not one

Agentra's core patent claim is a *model-agnostic* middleware that sits
between an agent and the payment gateway without modifying the agent. This
service proves that claim rather than asserting it: it never imports ASIDE's
Python code, it only calls ASIDE's public REST contract
(`POST /security/evaluate`, `GET /audit/decision-policy`) over HTTP. Either
side can be swapped, redeployed, or owned by a different person without
touching the other.

## Architecture

```
User message
   |
   v
Intent extraction (app/intent.py)        -- rule-based slot filling +
   |                                        goal decomposition, capped at
   |                                        2 clarifying questions
   v
Catalogue retrieval + ranking (app/retrieval.py)
   |                                      -- every justification is grounded
   |                                         in an actual Product field,
   |                                         never invented
   v
Canonical Decision Object (app/decision_object.py)
   |                                      -- stated intent, clarifying
   |                                         exchange, products retrieved,
   |                                         ranking rationale, final
   |                                         selection -- the artefact both
   |                                         halves of the pipeline share
   v
ASIDE client (app/aside_client.py)  ---->  ASIDE /security/evaluate
   |                                       (separate process/repo)
   v
ALLOW / REQUIRE_HUMAN_APPROVAL / REQUIRE_VERIFIABLE_APPROVAL / DENY
```

If ASIDE is unreachable, evaluation **fails closed** to
`REQUIRE_HUMAN_APPROVAL` rather than silently allowing the purchase or
crashing the demo (see `app/aside_client.py`).

## Running it

### Quickest path (no Docker, two terminals)

Terminal 1 — ASIDE (from its own repo):
```bash
cd aside_refactor
pip install -r requirements.txt
python -m app.ml.train          # bootstrap the ML risk model once
uvicorn app.main:app --port 8000
```

Terminal 2 — this service:
```bash
cd agentra_shopping_agent
pip install -r requirements.txt
uvicorn app.main:app --port 8100
```

Then open `http://127.0.0.1:8100/`.

### Docker

```bash
docker compose -f aside_refactor/docker-compose.yml up -d   # ASIDE + Postgres
ASIDE_BASE_URL=http://host.docker.internal:8000 docker compose up   # this service
```

## What's verified (not just claimed)

This build was tested against a **live ASIDE instance** (real Postgres +
pgvector, real trained GradientBoosting risk model, real guardrails), not
just mocks:

- A ₹6,250 anniversary-ring purchase (over the ₹5,000 autonomous limit): the
  ML model recommended ALLOW at 72% confidence, but the deterministic
  `AUTONOMOUS_PURCHASE_LIMIT` guardrail overrode it to
  `REQUIRE_HUMAN_APPROVAL` — exactly the "guardrails always win" design ASIDE
  documents, demonstrated end-to-end from this agent's UI.
- A ₹2,200 pendant purchase: `ALLOW`, risk score 7.
- Real RAG evidence came back with the flagged transaction (spending-limit
  bypass, high-value-transaction policy, historical incident precedent) —
  the explanation panel isn't a placeholder.

(One sandbox-specific note: HuggingFace was network-blocked in the verification
environment, so a deterministic stand-in embedder was used *only* to prove
the pipeline wiring — it does not touch ASIDE's shipped code. On a normal
machine with internet access, ASIDE downloads and caches
`all-MiniLM-L6-v2` itself, as its own README describes.)

## Test suite

```bash
pip install -r requirements.txt
pytest
```

43 tests: intent parsing, retrieval/ranking, the ASIDE client (success +
fail-closed, mocked with `respx`), the conversation orchestrator, and the
new layer -- Shapley-value correctness (the efficiency property: contributions
sum exactly to the total score), bias detection (position-bias, nudge
exposure, choice homogeneity), the irreversibility classifier, drift
detection, the dual-layer output (consumer/enterprise views sharing one
attribution pass), and the dashboard rollup.

## Demo script for Review 2

1. Open the UI, state an anniversary-gift situation with a budget and a
   durability preference in one sentence — show zero or one clarifying
   question (the brief's "hard cap the number of questions" principle).
2. Point at a justification string and show it's grounded in a real
   catalogue field, not generated prose.
3. Buy the item that's over the ₹5,000 autonomous limit — show the ML
   model's own recommendation in the risk-factors panel next to the
   guardrail that overrode it. This is the single strongest moment: it
   proves the "model is a recommendation, not the authority" trust boundary
   live, not on a slide.
4. Expand the "Why this decision" panel that appears with the gate result —
   point at the plain-language `top_reason`, then expand "Enterprise
   behavioral profile" underneath it and show the exact same product's full
   Shapley signal decomposition, irreversibility score, and bias audit. Say
   explicitly: *this is one computation, rendered twice* — that's the claim.
5. Open `/dashboard` in a second tab before the demo starts; after a few
   purchases it shows the rollup across sessions (decision breakdown,
   irreversibility distribution, bias/drift flags) live.
6. Open `/widget-demo`, paste in a completed session's ID, and show the
   embeddable Consumer Widget rendering the same decision a third-party
   merchant's own checkout page could show with two lines of integration
   code.
7. Expand "Decision trajectory" and show the Canonical Decision Object —
   this is literally the artefact the conversational-commerce brief and
   Agentra's own ADML both describe as the shared interface between the two
   halves of the platform.
8. If time allows, run a prompt-injection scenario against ASIDE directly
   (`docs/DEMO_CURL.md` in the ASIDE repo) to show DENY, without needing it
   routed through the shopping agent at all — proving ASIDE works as a
   standalone gate for *any* agent, not just this one.


## Autonomous mode and verification (added Oct 2026)
* `POST /agent/run {"goal": "..."}` -- goal in, verified order (or an honest stop) out. `GET /agent/orders`,
  `GET /agent/orders/{id}/verify`.
* `python scripts/razorpay_smoke_test.py` -- one Rs 10 TEST-mode order, retrieved and verified (needs rzp_test_ keys in `.env`).
* `GET /enterprise/behaviour-profile?policy=agentra|llm|first|cheapest` -- 2AFC behaviour experiment vs the human reference.
  `policy=llm` needs GROQ_API_KEY (or ANTHROPIC_API_KEY) in `.env` (capped at 200 pairs).
* `python scripts/human_baseline_from_opera.py --explore | --derive` -- optional OPeRA-derived human tilts.


## Web assistant (`/assistant`)
Conversational shopping assistant: Groq (needs `GROQ_API_KEY` in `.env`) asks at most one question at a time, searches the web (Tavily if `TAVILY_API_KEY` is set; else a Groq compound model; else DuckDuckGo; force one with `SEARCH_BACKEND=tavily|compound|ddg`), and proposes pages. Agentra then evaluates the list in code: grounded-source check, transparent score (rating shrunk towards 3.5 for few ratings, price fit, grounded; weights shown), and a comparison of the model's order with the evaluated order (Spearman, rank shifts, flags). Review authenticity: for the top 5 grounded results Agentra fetches the public page and a Groq model extracts the visible reviews; the existing rule-based check (`app/review_authenticity.py`) then flags bursts, low reviewer diversity and duplicate text (score penalty 0.15 if flagged; "not assessable" under 3 readable reviews). Limits: a visible sample only, pages may block reading, extraction is model-read and can be wrong; nothing is trained. Endpoint: `POST /assistant/message`.

Pre-payment risk: every result carries a backend `risk_score` (0-100, additive points: unverified source 40, flagged reviews 30, thin ratings 15, no rating 10, no price 10, low rating 10, over budget 5) and `risk_level` ("High risk" at 40 or more, else "Low risk") plus plain-language `risk_reasons`. The page shows only the tag and reasons; "High risk" results need an explicit acknowledgement before the seller link opens. Agentra takes no payment here. Rule-based signal, not a fraud verdict.

## Security mechanism and evaluation
See `MECHANISM.md` (CDC v2, anomaly detection, hardened injection detection, signal trust model, measured results and limits).
Run the attack harness: `python scripts/attack_harness.py --offline` (rules only) or `--aside-url http://localhost:8000` (full ASIDE).


## Behaviour study
See `PRICE_TILT_STUDY.md`: human price-tilt baseline (OPeRA), agent protocol, and the hidden price-proxy defect found and fixed in the ranker.
