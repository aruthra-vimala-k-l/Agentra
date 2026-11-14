# Vulnerability notes — what was fixed, what remains open

Written honestly rather than to look complete: a panel or a future
maintainer should be able to tell exactly what changed and what didn't.

## Fixed in this pass

**Unbounded in-memory session growth (DoS-by-memory-exhaustion).**
`app/session_store.py` previously had no cap. Any client could call
`POST /session/start` in a loop and grow the process's memory without bound.
Fixed: the store now caps at `MAX_SESSIONS` (500) and evicts the oldest
session once full. This bounds memory but does not add real rate-limiting,
per-IP throttling, or authentication on session creation — those still don't
exist.

**No authentication between this service and ASIDE.**
`app/aside_client.py` now sends an `X-API-Key` header (from `ASIDE_API_KEY`)
on every call to ASIDE, when configured. **This is only half the fix**:
ASIDE's own repo (a teammate's, deliberately not modified here — see the
README's "why two repos" section) has no matching check yet, so today this
header is inert. It needs a corresponding middleware check added to ASIDE
before this closes the gap. Flagged as a joint action item, not claimed as
solved.

## Still open (named, not fixed)

**Prompt-injection detection is ASIDE's regex-based classifier, unchanged.**
This shopping agent doesn't add its own injection defenses — it relies
entirely on ASIDE's `tool_output` scanning, which (per the original review)
is regex-based and can likely be evaded by rephrasing, translation, or
splitting a payload across turns. Out of scope for this repo to fix, since
it lives in ASIDE's codebase, but still true and still worth raising with
your teammate before the final review.

**The intent parser is keyword-based and brittle to off-script phrasing.**
`app/intent.py` fills slots by literal keyword match. A user who phrases
their situation in a way that doesn't hit any keyword gets stuck asking for
clarification rather than being understood. This was a deliberate initial
tradeoff (deterministic and explainable in front of a panel, no dependency on
an LLM API key being configured on demo day) but it is a real robustness gap,
not a design feature to defend indefinitely.

**Session store is process-local and not concurrency-audited beyond a single
lock.** Fine for a single-process demo; would not survive a horizontally
scaled deployment (two processes would have two disjoint session stores) or
a crash (no persistence). `app/drift.py`'s per-user history and
`app/dashboard_service.py`'s rollup share this same limitation — they all
read from the same in-memory, single-process state.

**RAG evidence retrieval depends on a live embedding call per request, not
just at ingestion.** This is an ASIDE-side architectural fact (every
`/security/evaluate` call re-embeds the query to search evidence), so an
embedding-service outage degrades ASIDE's *evidence* specifically —
guardrails and the ML risk model keep functioning. Not something this repo
can fix from the outside; worth naming to the panel as a known dependency,
not a hidden one.

**The Bias Detection Engine and Irreversibility Classifier are self-audits
of this agent's own shortlist, not of ASIDE or of the catalogue's own
sourcing.** They will not, for example, catch a biased catalogue (e.g. if
every "premium" brand happened to be the same merchant) — they audit the
ranking and selection logic in this repo, not the input data feeding it.


## Autonomous mode, payments and keys (added Oct 2026)
* `POST /agent/run` drives the normal conversation, so the ASIDE gate, audit log and payment rules all apply. It stops
  (never self-approves) on missing information, any REQUIRE_*_APPROVAL, or an unreachable gate; DENY moves to the next
  candidate. Before buying it re-verifies the product record; after buying it retrieves the order and checks amount,
  currency and status (`payments.verify_order`).
* Payments: `PAYMENT_MODE=dry_run` by default. `razorpay_test` accepts only `rzp_test_` keys, enforces a hard cap
  (PURCHASE_HARD_CAP_INR, default 50,000) and per session/product idempotency. Orders are "created", never "paid".
* Keys (RAZORPAY_*, GROQ_API_KEY, ANTHROPIC_API_KEY) live only in `.env` / environment; `app/envload.py` returns names only and never prints
  values; `.env` is git-ignored. The LLM behaviour experiment sends product titles, prices and ratings to the chosen LLM provider (Groq or Anthropic).
* The audit log is tamper-evident (hash chain), not tamper-proof.
* Human reference values are from Cherep et al. (ICLR 2026, N=30 humans) and are a small-sample reference; the Agentra
  rule-based policy is not an LLM agent. OPeRA-derived values are a different metric and are kept in a separate file.
