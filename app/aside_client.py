"""Client for ASIDE's Pre-Transaction Gate.

Deliberately a thin HTTP client against ASIDE's public contract
(POST /security/evaluate, GET /audit/decision-policy) -- this shopping agent
never imports ASIDE's Python code. That's the point: Agentra's core claim is
a *model-agnostic* middleware that sits between any agent and the payment
gateway. Calling ASIDE only over its REST API, from a separate service, is
what makes that claim demonstrable rather than asserted.

If ASIDE is unreachable (not running, wrong URL, still booting), evaluation
fails closed to REQUIRE_HUMAN_APPROVAL rather than either crashing the demo
or silently allowing every purchase -- a security gate that fails open on a
network hiccup would be a worse bug than one that occasionally over-escalates.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

ASIDE_BASE_URL = os.environ.get("ASIDE_BASE_URL", "http://localhost:8000")
TIMEOUT_SECONDS = float(os.environ.get("ASIDE_TIMEOUT_SECONDS", "8"))

# Vulnerability fix (client side only -- see SECURITY_NOTES.md): the original
# integration called ASIDE's API with no authentication at all, so anything
# that could reach the port could submit fabricated security evaluations.
# This sends a shared-secret header on every call when configured. It is
# NOT a complete fix by itself -- ASIDE's own repo (owned by a teammate,
# deliberately not modified here per the two-repo architecture) needs a
# matching check on its side before this closes the gap; until then this
# header is inert extra data. Documented as a joint action item, not claimed
# as solved unilaterally.
ASIDE_API_KEY = os.environ.get("ASIDE_API_KEY")


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": ASIDE_API_KEY} if ASIDE_API_KEY else {}


@dataclass
class GateResult:
    decision: str
    risk_score: int
    confidence: float
    threats: list[str] = field(default_factory=list)
    violated_policies: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    reasoning: str = ""
    evidence: list[dict] = field(default_factory=list)
    request_id: str | None = None
    unreachable: bool = False
    agentra_signals: dict = field(default_factory=dict)   # buyer-side evidence sent to / computed for the gate


def evaluate(payload: dict) -> GateResult:
    """POST to ASIDE's /security/evaluate. `payload` must match ASIDE's
    SecurityEvaluationRequest schema (see app/orchestrator.py for how the
    Canonical Decision Object gets translated into it).
    """
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(f"{ASIDE_BASE_URL}/security/evaluate", json=payload, headers=_auth_headers())
            response.raise_for_status()
            data = response.json()
        return GateResult(
            decision=data["decision"],
            risk_score=data["risk_score"],
            confidence=data.get("confidence", 0.0),
            threats=data.get("threats", []),
            violated_policies=data.get("violated_policies", []),
            risk_factors=data.get("risk_factors", []),
            reasoning=data.get("reasoning", ""),
            evidence=data.get("evidence", []),
            request_id=data.get("request_id"),
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return GateResult(
            decision="REQUIRE_HUMAN_APPROVAL",
            risk_score=100,
            confidence=0.0,
            threats=["aside_unreachable"],
            reasoning=(
                f"ASIDE security gate could not be reached at {ASIDE_BASE_URL} ({exc.__class__.__name__}). "
                "Failing closed to REQUIRE_HUMAN_APPROVAL rather than allowing the purchase."
            ),
            unreachable=True,
        )


def get_decision_policy() -> dict | None:
    """Fetch ASIDE's published thresholds/tool allowlist for display in the UI
    (the "based on what parameters" explanation). Returns None if unreachable
    -- purely cosmetic, so callers should degrade gracefully rather than fail.
    """
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.get(f"{ASIDE_BASE_URL}/audit/decision-policy", headers=_auth_headers())
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return None
