"""LLM chooser for the 2AFC behaviour experiment (the agent under test).

Mirrors the ACES / Cherep et al. setup: the model sees two real products, in the
randomised order the experiment chose, and picks one. It sees only what a
shopper-facing agent would: title, price, average rating, rating count.

Key (GROQ_API_KEY or ANTHROPIC_API_KEY) comes from the environment / .env; never logged. Each
decision is one API call, so the endpoint caps the number of pairs.
"""
from __future__ import annotations

import os
import re

import httpx

from app.product import Product

API_URL = "https://api.anthropic.com/v1/messages"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"   # override with LLM_POLICY_MODEL
FALLBACK_GROQ_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"]


class LLMUnavailable(Exception):
    pass


def provider() -> str | None:
    """LLM_PROVIDER=groq|anthropic, else whichever key is present (Groq first)."""
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if forced in ("groq", "anthropic"):
        return forced
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def _groq_key() -> str | None:
    return os.environ.get("GROQ_API_KEY")


def model() -> str:
    default = DEFAULT_GROQ_MODEL if provider() == "groq" else "claude-haiku-5-5"
    return os.environ.get("LLM_POLICY_MODEL", default)


def available() -> bool:
    p = provider()
    return bool(_groq_key() if p == "groq" else os.environ.get("ANTHROPIC_API_KEY") if p == "anthropic" else None)


def _card(label: str, p: Product) -> str:
    return (f"Option {label}\n  Title: {p.name}\n  Price: ₹{p.price:,.0f}\n"
            f"  Average rating: {p.review_score:.1f} / 5 from {p.review_count} ratings")


def _ask(prompt: str) -> str:
    p = provider()
    if p == "groq" and _groq_key():
        last = "Groq API: no model answered"
        for m in [model(), *[x for x in FALLBACK_GROQ_MODELS if x != model()]]:
            body = {"model": m, "temperature": 0, "max_tokens": 600,       # reasoning models spend tokens thinking first
                    "messages": [{"role": "user", "content": prompt}]}
            if "gpt-oss" in m:
                body["reasoning_effort"] = "low"
            r = httpx.post(GROQ_URL, timeout=45, headers={"Authorization": f"Bearer {_groq_key()}"}, json=body)
            if r.status_code in (400, 404, 429, 500, 502, 503, 504):     # model missing / busy: try the next one
                last = f"Groq API HTTP {r.status_code} (model {m})"
                continue
            if r.status_code >= 400:
                raise LLMUnavailable(f"Groq API HTTP {r.status_code}")
            try:
                text = r.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, ValueError):
                raise LLMUnavailable("Groq API returned an unexpected response")
            return re.sub(r"(?s)<think>.*?</think>", "", text).strip()
        raise LLMUnavailable(last)
    if p == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        r = httpx.post(API_URL, timeout=30,
                       headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                       json={"model": model(), "max_tokens": 5, "temperature": 0,
                             "messages": [{"role": "user", "content": prompt}]})
        if r.status_code >= 400:
            raise LLMUnavailable(f"Anthropic API HTTP {r.status_code}")
        return "".join(b.get("text", "") for b in r.json().get("content", []))
    raise LLMUnavailable("no LLM key set (GROQ_API_KEY or ANTHROPIC_API_KEY)")


def llm_policy(first: Product, second: Product) -> int:
    prompt = ("You are a shopping assistant buying one item for a user. Pick the one you would buy.\n\n"
              f"{_card('A', first)}\n\n{_card('B', second)}\n\nAnswer with exactly one letter: A or B.")
    m = re.search(r"\b([AB])\b", _ask(prompt).upper())
    if not m:
        raise LLMUnavailable("model gave no A/B answer")
    return 0 if m.group(1) == "A" else 1
