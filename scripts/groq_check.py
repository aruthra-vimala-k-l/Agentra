"""Shows exactly what Groq says to your key, model by model. Never prints the key.
    py scripts/groq_check.py
"""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx
from app import envload, llm_policy, web_assistant as wa

envload.load()
key = llm_policy._groq_key()
if not key:
    sys.exit("No GROQ_API_KEY found in .env")
print(f"key loaded: {key[:4]}...{key[-3:]} ({len(key)} chars)")
chain = wa._model_chain(key)
print("chat models this key can call (chain):", chain)
print("web-search models (tried in order):", wa._search_chain())
for m in chain + wa._search_chain():
    try:
        r = httpx.post(wa.GROQ_URL, timeout=40, headers={"Authorization": f"Bearer {key}"},
                       json={"model": m, "max_tokens": 5, "messages": [{"role": "user", "content": "Say OK"}]})
    except httpx.HTTPError as e:
        print(f"{m:34s} NETWORK {type(e).__name__}"); continue
    msg = ""
    try: msg = (r.json().get("error") or {}).get("message", "")
    except ValueError: pass
    print(f"{m:34s} HTTP {r.status_code}  {msg[:110]}  {('retry-after '+r.headers['retry-after']+'s') if 'retry-after' in r.headers else ''}")

print("search backend:", wa._search_backend(), "(set TAVILY_API_KEY for the most reliable search)")
if wa._search_backend() != "compound":
    try:
        res = wa._external_search(["OnePlus Nord Buds 3R buy online India"])
        print(f"search OK: {len(res)} results, first: {res[0]['uri'][:80]}")
    except Exception as e:
        print("search FAILED:", e)
