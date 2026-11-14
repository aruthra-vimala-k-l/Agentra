import respx
from httpx import ConnectError, Response

from app.aside_client import ASIDE_BASE_URL, evaluate, get_decision_policy


@respx.mock
def test_evaluate_parses_a_successful_aside_response():
    respx.post(f"{ASIDE_BASE_URL}/security/evaluate").mock(
        return_value=Response(200, json={
            "request_id": "abc-123",
            "risk_score": 12,
            "decision": "ALLOW",
            "threats": [],
            "violated_policies": [],
            "risk_factors": ["low amount"],
            "evidence_ids": [],
            "confidence": 0.9,
            "reasoning": "Routine purchase.",
            "evidence": [],
            "learning_applied": False,
            "learning_factors": [],
        })
    )
    result = evaluate({"agent_id": "a", "user_id": "u", "action": "purchase", "tool": "commerce.checkout"})
    assert result.decision == "ALLOW"
    assert result.risk_score == 12
    assert result.unreachable is False


@respx.mock
def test_evaluate_fails_closed_when_aside_is_unreachable():
    respx.post(f"{ASIDE_BASE_URL}/security/evaluate").mock(side_effect=ConnectError("connection refused"))
    result = evaluate({"agent_id": "a", "user_id": "u", "action": "purchase", "tool": "commerce.checkout"})
    assert result.decision == "REQUIRE_HUMAN_APPROVAL"
    assert result.unreachable is True


@respx.mock
def test_evaluate_fails_closed_on_non_200_response():
    respx.post(f"{ASIDE_BASE_URL}/security/evaluate").mock(return_value=Response(500))
    result = evaluate({"agent_id": "a", "user_id": "u", "action": "purchase", "tool": "commerce.checkout"})
    assert result.decision == "REQUIRE_HUMAN_APPROVAL"
    assert result.unreachable is True


@respx.mock
def test_decision_policy_returns_none_when_unreachable():
    respx.get(f"{ASIDE_BASE_URL}/audit/decision-policy").mock(side_effect=ConnectError("down"))
    assert get_decision_policy() is None


@respx.mock
def test_decision_policy_returns_payload_when_reachable():
    respx.get(f"{ASIDE_BASE_URL}/audit/decision-policy").mock(
        return_value=Response(200, json={"thresholds": {"risk_deny_threshold": 90}})
    )
    policy = get_decision_policy()
    assert policy["thresholds"]["risk_deny_threshold"] == 90
