# Patent claim status — honest accounting

Agentra's proposal names three claims. This document says plainly which of
them this build actually demonstrates, which it partially demonstrates, and
which it deliberately does not touch — and why.

## Claim 1 — game-theoretic decomposition of commerce signals

**Demonstrated, with a real caveat about scale.**

`app/attribution.py` computes exact Shapley values over a set function that
scores a product against a shopper's stated intent (price fit, durability
match, occasion match, review influence, brand preference, nudge exposure,
new-merchant caution), including a genuine nonlinear interaction term
(durability + occasion synergy), so the decomposition is provably not just a
relabeling of the raw per-signal weights — see
`tests/test_attribution.py::test_shapley_values_satisfy_the_efficiency_property`
for the check that makes this a verifiable claim rather than an assertion.

**The caveat:** this is exact Shapley computation over a small, hand-named
set of interpretable commerce signals (<=7), which is only tractable because
the signal count is small (2^7 = 128 coalitions). A production system
attributing a decision across hundreds of raw features (every catalogue
field, every session event) would need sampling-based approximation
(KernelSHAP) instead of exact enumeration. That's a disclosed scaling
limitation, not a hidden one — and it's also, incidentally, the honest reason
this is closer to "SHAP-style" than "SHAP the library": it's a from-scratch
exact Shapley calculator sized to this problem, not a wrapper around the
`shap` package.

## Claim 2 — dual-layer architecture (one inference pass, two outputs)

**Demonstrated.**

`app/behavioral_profile.build_dual_layer()` calls `attribute()` exactly
once per chosen product per turn. Both the consumer-facing `consumer_view`
and the enterprise-facing `enterprise_view` are built from that single
`AttributionResult`, plus one irreversibility classification and one bias
audit — nothing is computed twice, and nothing shown to the consumer
contradicts the enterprise numbers because they're the same numbers.
`tests/test_behavioral_profile.py` checks this by construction (both views
are built from local variables computed once, in one function call) rather
than by re-deriving anything.

## Claim 3 — behavioral drift detection over time

**Partially demonstrated — a real but narrow slice, not the full claim.**

`app/drift.py` tracks each user's Shapley-value history and flags a
meaningful shift in which signals are driving their purchases turn-over-turn.
This genuinely detects drift in the technical sense (a moving-average
baseline compared against the latest sample). What it is *not*: persistent
across restarts, statistically rigorous (no significance testing, just a
fixed threshold), or informed by a learned baseline the way ASIDE's own
online model is. Treat this as proof the mechanism is buildable, not as the
finished claim.

## What was deliberately NOT built: the CDC internals

Agentra's own architecture note (kept in the founder's private project
notes, not this repo) describes the Counterfactual-Divergence-Comparator's
internal mechanism — the specific "semantic-divergence-weighted-by-
irreversibility" re-derivation logic that differentiates it from prior art
like MELON/ARGUS — as **not to be disclosed in any public repo or demo
artifact until patent-attorney clearance**, because a push to a repo (even a
private-looking one used for a university submission) can count as prior-art
disclosure that weakens or forecloses a later patent filing.

This build does not implement that mechanism. Where the pipeline references
"Secure" or counterfactual reasoning, it relies entirely on ASIDE's own
guardrail + ML risk model (a teammate's independent, separately-authored
system) and this repo's irreversibility classifier (a straightforward,
rule-based, publicly-defensible scoring function — not the CDC's semantic-
divergence mechanism). If a future iteration of this capstone needs to
demonstrate Claim 1's *counterfactual* half specifically (as opposed to the
signal-decomposition half demonstrated here), that requires either
attorney clearance first, or a live, undisclosed demo (e.g. running only on
a machine in front of the review panel, never committed to any repo) rather
than a submitted artifact. Flagging this explicitly rather than building
around it quietly, since the two paths have very different IP consequences.
