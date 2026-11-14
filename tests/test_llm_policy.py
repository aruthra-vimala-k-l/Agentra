import os

import httpx
import respx
from fastapi.testclient import TestClient

from app import envload, llm_policy
from app.main import app
from app.mock_catalog import CATALOGUE


def test_llm_policy_parses_answer(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with respx.mock:
        respx.post(llm_policy.API_URL).mock(return_value=httpx.Response(200, json={"content": [{"type": "text", "text": "B"}]}))
        assert llm_policy.llm_policy(CATALOGUE[0], CATALOGUE[1]) == 1


def test_groq_policy_parses_answer(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert llm_policy.provider() == "groq"
    with respx.mock:
        route = respx.post(llm_policy.GROQ_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]}))
        assert llm_policy.llm_policy(CATALOGUE[0], CATALOGUE[1]) == 0
        assert route.calls.last.request.headers["authorization"] == "Bearer test-key"
        assert "key=" not in str(route.calls.last.request.url)


def test_llm_endpoint_refuses_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert TestClient(app).get("/enterprise/behaviour-profile?policy=llm").status_code == 400


def test_envload_does_not_override_and_returns_names_only(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("FOO_TEST_X=1\nBAR_TEST_Y='2'\n")
    monkeypatch.setenv("BAR_TEST_Y", "keep")
    assert envload.load(f) == ["FOO_TEST_X"]
    assert os.environ["BAR_TEST_Y"] == "keep"
    monkeypatch.delenv("FOO_TEST_X", raising=False)


def test_groq_policy_falls_back_when_model_missing_and_allows_thinking_tokens(monkeypatch):
    import json
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_POLICY_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = []
    def side(request):
        d = json.loads(request.content); seen.append(d)
        if d["model"] == "openai/gpt-oss-20b":
            return httpx.Response(404, json={"error": {"message": "no access"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think>B"}}]})
    with respx.mock:
        respx.post(llm_policy.GROQ_URL).mock(side_effect=side)
        assert llm_policy.llm_policy(CATALOGUE[0], CATALOGUE[1]) == 1
    assert seen[0]["max_tokens"] >= 200 and seen[1]["model"] == "openai/gpt-oss-120b"
