"""Conversational web shopping assistant (Groq: compound web search + chat models) with an Agentra evaluation layer.

Flow per turn:
  1. The Groq model decides: ask ONE clarifying question | search | chat about the current results.
  2. On "search": a Groq compound model searches the web (built-in search) and proposes candidate pages.
  3. Agentra evaluates the candidates IN CODE (not by the LLM):
       * grounded    -- the page's site appears among the search sources actually returned; ungrounded
                        candidates are demoted and flagged (guards against invented links);
       * transparent score -- 0.5 * shrunk rating + 0.3 * price fit + 0.2 * grounded, weights shown to the user;
       * bias audit  -- the model's own order is compared with the transparent order (rank shifts, Spearman);
                        large shifts and thin-evidence ratings are flagged.
Review check: for the top candidates Agentra fetches the public page and a Groq model extracts the visible reviews;
Agentra's existing rule-based check (app/review_authenticity.py) runs on them. It is a sample, not all reviews, and
pages may block reading. It does not train or fine-tune any model. Ratings/prices are as stated by sources and may be stale.
"""
from __future__ import annotations

import concurrent.futures as cf
from html import unescape
import time
import json
import os
import re
import uuid
from datetime import datetime
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from app import injection_guard, llm_policy, risk_weights

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
WEIGHTS = {"rating": 0.5, "price_fit": 0.3, "grounded": 0.2}
AUTH_PENALTY = 0.15   # subtracted from the score only when the review-authenticity check flags the page
HIGH_RISK_AT = risk_weights.threshold()     # risk_score >= this -> "High risk" tag (40 points = odds x4)
# Transparent risk points (0-100, capped). Higher = riskier to pay.
# Text in a listing that tries to give orders to an AI agent. Shared with the payment gate (web_checkout).
INJECTION = injection_guard.PATTERN   # rule families in app/injection_guard.py (measured: RISK_EVALUATION.md)
# Weights come from app/risk_weights.py (data/references/risk_weights.json): each is "fitted" (from labelled data), "ahp" (expert
# pairwise judgments, Analytic Hierarchy Process) or "provisional" (original hand-set value until ratings are collected).
# Method and limits: RISK_METHOD.md. 20 points = odds of an unsafe listing doubled; High risk at 40 points = odds x4.
RISK_POINTS = risk_weights.points()
# A trusted MARKETPLACE does not make the seller or the product trusted: fakes, hijacked reviews and look-alike sellers live on
# all of these. Trust has to come from evidence about THIS seller and THIS product (ratings count, seller record, price sanity).
MARKETPLACES = {"amazon.in", "amazon.com", "flipkart.com", "meesho.com", "myntra.com", "ajio.com", "nykaa.com", "snapdeal.com",
                "tatacliq.com", "paytmmall.com", "jiomart.com", "shopsy.in", "indiamart.com"}
PLATFORM_SELLERS = ("amazon", "flipkart", "myntra", "ajio", "nykaa", "tata cliq", "jiomart")
POPULARITY_BADGES = ("bestseller", "best seller", "amazon's choice", "amazons choice", "choice", "sponsored", "ad", "trending", "top rated", "assured")
AUTH_TOP_N = 5        # reviews are read for the top-N candidates only (cost/latency)
PRIOR_RATING, PRIOR_COUNT = 3.5, 20   # Bayesian shrinkage: few ratings -> pulled towards 3.5
_SESSIONS: dict[str, dict] = {}


class AssistantUnavailable(Exception):
    pass


_DISCOVERED: dict = {}


PREFERRED_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b",      # plain chat/JSON models, best first
                    "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]               # (override: ASSISTANT_MODEL; dead ids just 404 and are skipped)
SEARCH_MODELS = ["groq/compound", "groq/compound-mini"]                    # Groq models with built-in web search
                                                                           # (override: ASSISTANT_SEARCH_MODEL)


def model() -> str:
    return (os.environ.get("ASSISTANT_MODEL") or os.environ.get("LLM_POLICY_MODEL")
            or _DISCOVERED.get("model") or PREFERRED_MODELS[0])


def _search_chain() -> list[str]:
    own = os.environ.get("ASSISTANT_SEARCH_MODEL")
    return ([own] if own else []) + [m for m in SEARCH_MODELS if m != own]


def _candidates(key: str) -> list[str]:
    """Chat models this Groq key can call, best first: the known-good ones, then other text models. Speech, guard,
    embedding, TTS and the search-agent (compound) models are skipped."""
    try:
        r = httpx.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        names = [m["id"] for m in r.json().get("data", []) if m.get("id")]
    except (httpx.HTTPError, ValueError, KeyError):
        return []
    good = ("llama", "gpt-oss", "qwen", "kimi")          # families that follow chat + JSON-mode instructions
    bad = ("whisper", "guard", "tts", "playai", "orpheus", "embed", "compound", "safeguard", "vision", "audio", "allam")
    ok = [n for n in names if any(g in n for g in good) and not any(b in n for b in bad)]
    return ([n for n in PREFERRED_MODELS if n in ok] + sorted(n for n in ok if n not in PREFERRED_MODELS))[:6]


STATIC_FALLBACKS = list(PREFERRED_MODELS)
RETRYABLE = (404, 429, 500, 502, 503, 504)   # model missing / rate-limited / overloaded -> try the next model
CHAIN_SIZE, ROUNDS = 6, 2


def _model_chain(key: str, discover: bool = True) -> list[str]:
    """Up to 6 models in the order they are tried: the one that last worked (or the env override), then the models this
    key can call, then fixed well-known fallbacks. Duplicates removed."""
    if discover and "queue" not in _DISCOVERED:
        _DISCOVERED["queue"] = _candidates(key) or []
    chain: list[str] = []
    for m in [model(), *_DISCOVERED.get("queue", []), *STATIC_FALLBACKS]:
        if m and m not in chain:
            chain.append(m)
    return chain[:CHAIN_SIZE]


def _discover_model(key: str) -> bool:
    """Kept for callers/tests: switch the remembered model to the next one in the chain. True if there is one."""
    chain = _model_chain(key)
    nxt = [m for m in chain if m != model()]
    if not nxt:
        return False
    _DISCOVERED["model"] = nxt[0]
    return True


_SNIPPETS: dict[str, str] = {}      # url -> title + snippet from the search that found it (fallback evidence)
_BLOCKED = re.compile(r"robot check|captcha|enter the characters|access denied|are you a human|unusual traffic", re.I)
_UA = {"User-Agent": "Mozilla/5.0 (compatible; AgentraBot/1.0; +evaluation)"}


def _direct_page(url: str, limit: int) -> str:
    try:
        r = httpx.get(url, headers=_UA, follow_redirects=True, timeout=20)
    except httpx.HTTPError:
        return ""
    if r.status_code >= 400:
        return ""
    html = r.text
    ld = re.findall(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", html, re.S | re.I)
    body = re.sub(r"(?is)<(script|style|noscript|svg)\b.*?</\1>", " ", html)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", unescape(body)).strip()
    if _BLOCKED.search(body[:1500]) and len(body) < 1500 and not ld:
        return ""                                  # a bot-check page, not the product
    return (("STRUCTURED DATA: " + " ".join(ld)[:5000] + "\n") if ld else "") + "VISIBLE TEXT: " + body


def _tavily_extract(url: str, limit: int, advanced: bool = False) -> str:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return ""
    try:
        body = {"urls": [url]}
        if advanced:                       # renders the page more fully: marketplaces fill ratings in with scripts
            body["extract_depth"] = "advanced"
        r = httpx.post("https://api.tavily.com/extract", timeout=40, headers={"Authorization": "Bearer " + key},
                       json=body)
        if r.status_code >= 400:
            return ""
        res = (r.json().get("results") or [{}])[0]
        return ("EXTRACTED TEXT: " + (res.get("raw_content") or "")) if res.get("raw_content") else ""
    except (httpx.HTTPError, ValueError, IndexError):
        return ""


def _page_text(url: str, limit: int = 12000) -> str:
    """Evidence about a product page for the model, best first: (1) the page fetched directly (structured data +
    visible text), (2) Tavily's page extractor (works on many sites that block simple fetches), (3) the search snippet
    that surfaced the page. Empty string when there is nothing. The model must use null for anything not shown."""
    text = _direct_page(url, limit)
    mkt = any(_host(url).endswith(m) for m in MARKETPLACES)
    src = "direct"
    # A marketplace page often comes back as a script skeleton: long enough, but with no rating or review count in it.
    # That used to be accepted, and the card then showed "No rating". Ask the extractor for the rendered page instead.
    if len(text) < 400 or (mkt and not _has_rating_signal(text)):
        ext = _tavily_extract(url, limit, advanced=mkt)
        if ext:
            text, src = (ext + "\n" + text).strip(), "extractor"
    if len(text) < 400 and _SNIPPETS.get(url):
        text = (text + "\nSEARCH SNIPPET ONLY (the page itself could not be opened): " + _SNIPPETS[url]).strip()
    text = text[:limit]
    print(f"[agentra] page text: {_host(url)} source={src} chars={len(text)} rating_signal={_has_rating_signal(text)}", flush=True)
    return text


def _has_rating_signal(text: str) -> bool:
    """True when the page text actually contains a rating or review count (structured data or visible text)."""
    return bool(re.search(r"aggregateRating|ratingValue|out of 5|\d[\d,]*\s+(?:global\s+)?(?:ratings?|reviews?)|\d(?:\.\d)?\s*★", text, re.I))


def _search_backend() -> str:
    """Where web search comes from. SEARCH_BACKEND=compound|tavily|ddg forces one; auto = Tavily if TAVILY_API_KEY is set,
    else Groq's compound model, else (when compound is not available to this key) DuckDuckGo."""
    b = os.environ.get("SEARCH_BACKEND", "auto").strip().lower()
    if b in ("compound", "tavily", "ddg"):
        return b
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    if _DISCOVERED.get("compound_ok") is False:
        return "ddg"
    return "compound"


def _tavily(query: str) -> list[dict]:
    r = httpx.post("https://api.tavily.com/search", timeout=30,
                   headers={"Authorization": "Bearer " + os.environ["TAVILY_API_KEY"]},
                   json={"query": query, "max_results": 8, "search_depth": "basic"})
    if r.status_code >= 400:
        raise AssistantUnavailable(f"Tavily search HTTP {r.status_code}")
    return [{"title": x.get("title", ""), "uri": x["url"], "snippet": (x.get("content") or "")[:300]}
            for x in r.json().get("results", []) if x.get("url")]


def _ddg(query: str) -> list[dict]:
    """DuckDuckGo HTML results (no key needed, best effort: may be rate limited)."""
    try:
        r = httpx.get("https://html.duckduckgo.com/html/?q=" + quote_plus(query), headers=_UA, timeout=20, follow_redirects=True)
    except httpx.HTTPError:
        return []
    if r.status_code >= 400:
        return []
    out = []
    for m in re.finditer(r'<a\b[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', r.text, re.S):
        tag = m.group(0)
        h = re.search(r'href="([^"]+)"', tag)
        if not h:
            continue
        href = unescape(h.group(1))
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href if href.startswith("http") else "https:" + href).query).get("uddg", [""])[0])
        if href.startswith("http") and "duckduckgo.com" not in href:
            title = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", m.group(1)))).strip()
            out.append({"title": title, "uri": href, "snippet": ""})
    return out[:8]


def _external_search(queries: list[str]) -> list[dict]:
    backend = _search_backend()
    if backend == "tavily" and not os.environ.get("TAVILY_API_KEY"):
        raise AssistantUnavailable("TAVILY_API_KEY is not set (put it in .env, never in chat).")
    fn = _tavily if backend == "tavily" else _ddg
    found, seen = [], set()
    qs = [x for x in queries if x][:6]
    if len(qs) > 1:                                # the searches are independent: run them together, keep the original order
        with cf.ThreadPoolExecutor(max_workers=len(qs)) as ex:
            batches = list(ex.map(fn, qs))
    else:
        batches = [fn(q) for q in qs]
    for batch in batches:
        for item in batch:
            if item["uri"] not in seen:
                seen.add(item["uri"])
                found.append(item)
                _SNIPPETS[item["uri"]] = (item.get("title", "") + " " + item.get("snippet", "")).strip()
    if not found:
        raise AssistantUnavailable("Web search returned no results (" + backend + "). Set TAVILY_API_KEY in .env for reliable "
                                   "search (free tier at tavily.com), or try again in a minute.")
    return found[:14]


def _fmt_results(results: list[dict]) -> str:
    return "\n".join(f"- {r['title']} | {r['uri']}" + (f" | {r['snippet']}" if r.get("snippet") else "") for r in results)


def _generate(prompt: str, *, search: bool = False, as_json: bool = False, url_context: bool = False,
              queries: list[str] | None = None, prefer: str | None = None) -> tuple[str, list[dict]]:
    """One Groq call. search=True gets real web results first (Tavily, a Groq compound model, or DuckDuckGo: see
    _search_backend) and returns the pages found as chunks [{title, uri}]; url_context=True fetches the URL in the
    prompt and gives the model its text."""
    key = llm_policy._groq_key()
    if not key:
        raise AssistantUnavailable("GROQ_API_KEY is not set (put it in .env, never in chat).")
    content = prompt
    if url_context:
        m_url = re.search(r"https?://[^\s\"']+", prompt)
        page = _page_text(m_url.group(0)) if m_url else ""
        if not page:
            return "", []        # unreadable page: callers treat an empty answer as "nothing readable"
        content = prompt + "\n\nPAGE CONTENT (fetched by Agentra; may be partial, use only this):\n" + page
    backend = _search_backend() if search else None
    compound = search and backend == "compound"
    results: list[dict] = []
    if search and not compound:
        results = _external_search(queries or [prompt.splitlines()[0][:200]])
        content = (prompt + "\n\nSEARCH RESULTS (real pages Agentra just fetched; use ONLY these, never invent or edit a URL):\n"
                   + _fmt_results(results))
    body: dict = {"messages": [{"role": "user", "content": content}], "temperature": 0.2}
    if as_json and not compound:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}"}

    def call(m: str):
        extra = {"reasoning_effort": "low"} if "gpt-oss" in m else {}      # reasoning models: keep answers fast
        return httpx.post(GROQ_URL, timeout=90 if compound else 45, headers=headers, json={**body, "model": m, **extra})

    chain = _search_chain() if compound else _model_chain(key, discover=False)   # no extra network call while things work
    if prefer and not compound:                  # a small fast model first for simple calls; the normal chain is the fallback
        chain = [prefer] + [m for m in chain if m != prefer]
    failures: list[str] = []
    r = None
    for rnd in range(ROUNDS):                    # go through the whole chain, then once more, before giving up
        wait = 0
        i = 0
        while i < len(chain):
            m = chain[i]; i += 1
            try:
                r = call(m)
            except httpx.HTTPError as exc:       # timeout / dropped connection
                failures.append(f"{m}: {type(exc).__name__}")
                if not compound and "queue" not in _DISCOVERED:
                    chain = _model_chain(key); i = chain.index(m) + 1 if m in chain else i
                continue
            retry = r.status_code in RETRYABLE or (r.status_code == 400 and any(
                k in r.text.lower() for k in ("decommission", "tool choice", "does not support", "not support", "json_validate", "reasoning")))
            if retry:
                failures.append(f"{m}: HTTP {r.status_code}")
                if not compound and "queue" not in _DISCOVERED:   # first trouble: ask which models this key can really call
                    chain = _model_chain(key); i = chain.index(m) + 1 if m in chain else i
                try:
                    wait = max(wait, int(float(r.headers.get("retry-after", 0))) + 1 if r.headers.get("retry-after") else 0)
                except ValueError:
                    pass
                continue
            if r.status_code < 400 and not compound:
                _DISCOVERED["model"] = m         # start from the model that worked next time
            break
        else:
            r = None
            if rnd < ROUNDS - 1:
                time.sleep(min(max(wait, 3), 20))   # brief pause before the second pass
            continue
        break
    if r is None and compound and failures and all("HTTP 404" in f for f in failures):
        _DISCOVERED["compound_ok"] = False       # this key cannot call the compound models: search another way
        return _generate(prompt, search=search, as_json=as_json, url_context=url_context, queries=queries)
    if r is None:
        raise AssistantUnavailable(
            "Groq did not answer in time on any of " + str(len(chain)) + " models, tried twice (" + "; ".join(failures[-6:]) +
            "). Spikes in demand or the free rate limit are usually temporary - try again in a minute.")
    if r.status_code >= 400:
        try:
            detail = (r.json().get("error") or {}).get("message", "")[:160]
        except ValueError:
            detail = ""
        raise AssistantUnavailable(f"Groq API HTTP {r.status_code} (model {model()}): {detail}")
    msg = ((r.json().get("choices") or [{}])[0]).get("message", {})
    text = re.sub(r"(?s)<think>.*?</think>", "", msg.get("content") or "").strip()    # some models inline their reasoning
    chunks, seen = [], set()
    for t in msg.get("executed_tools") or []:
        for item in (t.get("search_results") or {}).get("results") or []:
            u = item.get("url")
            if u and u not in seen:
                seen.add(u)
                chunks.append({"title": item.get("title", ""), "uri": u})
    if search and not compound:
        chunks = [{"title": x["title"], "uri": x["uri"]} for x in results]
    return text, chunks


def _names_from(text: str) -> list[str]:
    """Product names from a model reply: a bare list, an object wrapping a list ({"products": [...]}), or prose around either."""
    try:
        data = _json_from(text or "")
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    out = []
    for x in (data if isinstance(data, list) else []):
        n = x.get("name") if isinstance(x, dict) else (x if isinstance(x, str) else None)
        if n and str(n).strip():
            out.append(str(n).strip())
    return out[:N_PRODUCTS]


def _json_from(text: str):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", t, re.S)
        return json.loads(m.group(1)) if m else None


def _host(url: str) -> str:
    h = (urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "en-IN,en;q=0.9"}
KNOWN_SITES = ["amazon.in", "amazon.com", "flipkart.com", "myntra.com", "ajio.com", "nykaa.com", "croma.com",
               "reliancedigital.in", "tatacliq.com", "meesho.com", "snapdeal.com", "bigbasket.com", "paytmmall.com"]
RISKY_TLDS = {"xyz", "top", "shop", "click", "club", "buzz", "online", "site", "store", "icu", "live", "vip", "cyou", "rest"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "rebrand.ly", "is.gd", "ow.ly", "tiny.cc"}
URL_RE = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}/[^\s<>\"']*)", re.I)


def check_link(url: str, timeout: float = 8) -> dict:
    """Really open the link from the server. state: ok (page loads) | blocked (site refuses bots, page may exist)
    | dead (404/410/unknown host: the page does not exist) | unreachable (timeout/other)."""
    try:
        r = httpx.get(url, headers=UA, follow_redirects=True, timeout=timeout)
    except httpx.ConnectError:
        return {"status": None, "final_url": url, "state": "dead"}
    except httpx.HTTPError:
        return {"status": None, "final_url": url, "state": "unreachable"}
    st = r.status_code
    state = "ok" if st < 400 else "dead" if st in (404, 410) else "blocked"
    return {"status": st, "final_url": str(r.url), "state": state}


def _registrable(host: str) -> str:
    p = host.split(".")
    return ".".join(p[-3:]) if len(p) >= 3 and len(p[-1]) == 2 and p[-2] in ("co", "com", "org", "net", "gov") else ".".join(p[-2:])


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def url_risk(url: str) -> list[tuple[str, str]]:
    """Rule-based signals read from the URL itself (no network)."""
    u = urlparse(url if "://" in url else "https://" + url)
    host = (u.hostname or "").lower()
    out: list[tuple[str, str]] = []
    if u.scheme != "https":
        out.append(("insecure_url", "The link is not secure (no HTTPS), so anything you type could be intercepted."))
    if re.fullmatch(r"[\d.]+", host) or ":" in host or "xn--" in host:
        out.append(("odd_url", "The address is a raw number or uses disguised characters instead of a normal shop name."))
    reg = _registrable(host)
    if reg in SHORTENERS:
        out.append(("shortener", "A shortened link hides where it really goes."))
    if reg not in KNOWN_SITES:
        first, _, rest = reg.partition(".")
        regn = first.translate(str.maketrans("0134578@$", "oleastbas")) + ("." + rest if rest else "")   # amaz0n -> amazon
        near = [k for k in KNOWN_SITES if any(0 < _edit(x, k) <= 2 or (k.split(".")[0] in x.split(".")[0] and x != k and len(x.split(".")[0]) > len(k.split(".")[0]))
                                              for x in {reg, regn})]
        if near:
            out.append(("lookalike_domain", f"The site name looks like {near[0]} but is not it. Fake shops often copy real names."))
    if host.rsplit(".", 1)[-1] in RISKY_TLDS:
        out.append(("risky_tld", "The web address ending (." + host.rsplit(".", 1)[-1] + ") is common on throwaway shops."))
    return out


FOREIGN_MARKERS = ("/uk/", "/us/", "/en-gb/", "/en-us/", "/eu/", "/au/", "/ca/", "/de/", "/fr/", "/ae/", "/sg/")
FOREIGN_TLDS = (".co.uk", ".com.au", ".ca", ".de", ".fr", ".ae", ".com.sg")


STRONG_PRODUCT = re.compile(r"/dp/[A-Za-z0-9]{10}|/gp/product/|/p/[^/?]+|[?&]pid=|/itm[0-9a-z]|/buy(/|$|\?)|/product/|/products/", re.I)
LISTING_TOKENS = {"best", "top", "under", "below", "deals", "deal", "vs", "versus", "review", "reviews", "guide", "alp", "category",
                  "categories", "collection", "collections", "search", "blog", "blogs", "news", "compare", "comparison", "list",
                  "offers", "sale", "buying"}


def looks_like_listing(url: str) -> bool:
    """Category / search / roundup / blog URLs (seen in testing: /earbuds-under-2000, /alp/..., /best-tws-earbuds-deals-...)."""
    if STRONG_PRODUCT.search(url):
        return False
    path = urlparse(url if "://" in url else "https://" + url).path.lower()
    return bool(set(re.split(r"[/\-_.]+", path)) & LISTING_TOKENS)


def page_quality(url: str) -> list[tuple[str, str]]:
    """Cheap checks that catch the faults seen in testing: home/category pages and other-country stores."""
    u = urlparse(url if "://" in url else "https://" + url)
    path = (u.path or "/").lower().rstrip("/") + "/"
    segs = [x for x in path.split("/") if x]
    out = []
    if not segs or segs in (["in"], ["en"], ["en-in"], ["india"]):
        out.append(("not_product_page", "This is a home page, not a specific product page."))
    elif segs[0] in ("collections", "category", "categories", "search", "c", "s", "blog", "blogs") and len(segs) <= 2:
        out.append(("not_product_page", "This looks like a category or search page, not a specific product."))
    if looks_like_listing(url) and not out:
        out.append(("not_product_page", "This looks like a category, search or roundup page, not one specific product."))
    host = (u.hostname or "").lower()
    if any(m in path for m in FOREIGN_MARKERS) or host.endswith(FOREIGN_TLDS):
        out.append(("wrong_region", "This looks like another country's store, so price, stock and delivery may not apply in India."))
    return out


def resolve_sources(chunks: list[dict]) -> list[dict]:
    """Search sources: open each one to learn the real page URL (and drop dead ones)."""
    chunks = [c for c in chunks if c.get("uri")][:12]
    with cf.ThreadPoolExecutor(max_workers=max(1, min(len(chunks), 6))) as ex:
        links = list(ex.map(lambda c: check_link(c["uri"]), chunks))
    out = []
    for c, lk in zip(chunks, links):
        if lk["state"] != "dead":
            out.append({"title": c.get("title", ""), "url": lk["final_url"], "host": _host(lk["final_url"]), "state": lk["state"]})
    return out


def _tokens(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2}


def verify_candidates(cands: list[dict], resolved: list[dict]) -> None:
    """Open every proposed link. Dead link -> try to repair it with a real source page on the same site whose title
    matches the product; otherwise leave it marked dead (the UI then offers a search instead of the broken link)."""
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        links = list(ex.map(lambda c: check_link(c["url"]), cands))
    for c, lk in zip(cands, links):
        c["url_risk"] = url_risk(c["url"]) + page_quality(c["url"])
        if lk["state"] == "dead":
            host = _host(c["url"])
            best = max((r for r in resolved if r["host"] == host),
                       key=lambda r: len(_tokens(r["title"]) & _tokens(c.get("name", ""))), default=None)
            if best and len(_tokens(best["title"]) & _tokens(c.get("name", ""))) >= 2:
                c["url"], c["link_repaired"] = best["url"], True
                lk = {"status": 200, "final_url": best["url"], "state": "ok"}
        c["link"] = lk


def risk_assessment(c: dict, grounded: bool, adj_rating: float, budget_inr: float | None) -> dict:
    """Pre-payment risk score (0-100, backend) with plain-language reasons; the UI shows only the tag + reasons.
    Rule-based and additive so every point can be explained. Not a fraud verdict."""
    pts, why = 0, []
    def add(key, msg):
        nonlocal pts
        pts += RISK_POINTS[key]
        why.append(msg)
    auth = c.get("authenticity") or {}
    rating, n, price = c.get("rating"), c.get("rating_count"), c.get("price_inr")
    state = (c.get("link") or {}).get("state")
    if state == "dead":
        add("dead_link", "This link does not open a working page (the site says it does not exist), so do not rely on it.")
    elif state == "unreachable":
        add("dead_link", "This site did not respond when we tried to open it, so we could not confirm it is real or working.")
    elif not grounded:
        add("unverified_source", "We could not confirm this page in the search results, so the link may be wrong or fake.")
    for key, msg in c.get("url_risk", []):
        add(key, msg)
    if INJECTION.search(" ".join(str(c.get(k) or "") for k in ("name", "why"))):
        add("prompt_injection", "The listing text contains instructions aimed at an AI shopping agent (for example 'ignore previous "
                                "instructions'). Real sellers do not write this: treat the listing as hostile and do not pay.")
    if auth.get("suspicious"):
        add("review_flagged", "The reviews look unusual (" + "; ".join(auth.get("reasons", []))[:200] + ").")
    host = _registrable(_host(c.get("url", "")))
    market = host in MARKETPLACES
    if (rating is None and market) or (rating is not None and (n or 0) < 5):
        add("no_rating_evidence", ("A star rating is shown but we could not confirm that anyone actually rated it (fewer than 5 ratings), so it proves nothing."
                                   if rating is not None else
                                   "No customer ratings or reviews could be confirmed for this listing. New and fake listings look exactly like this, "
                                   "even on a trusted marketplace."))
    elif rating is None:
        add("no_rating", "No customer rating is shown.")
    elif (n or 0) < 30:
        add("thin_ratings", "The rating rests on very few customers (under 30), so it can be swung by a handful of people.")
    if rating is not None and rating >= 4.7 and (n or 0) < 100:
        add("too_good_few", "A near-perfect rating from so few customers is a common sign of seeded or fake reviews.")
    seller = (c.get("seller") or "").strip()
    if market:
        first_party = seller.lower().startswith(PLATFORM_SELLERS)
        if not seller:
            add("seller_unknown", f"The seller could not be identified. {host} is a trusted marketplace, but the seller on it may not be.")
        elif not first_party:
            sr, sn = c.get("seller_rating"), c.get("seller_rating_count")
            if sr is None or sr < 4.0 or (sn is not None and sn < 50):
                add("seller_unproven", f"Sold by a third party ({seller[:40]}) with no visible good track record"
                    + (f" (seller rating {sr})" if sr is not None else "") + f". {host} being trusted does not make this seller trusted.")
    mrp = c.get("mrp_inr")
    if mrp and price and mrp > 0 and price / mrp <= 0.3:
        add("extreme_discount", f"Listed {round((1 - price / mrp) * 100)}% below its list price. Inflated 'original prices' are a common trick.")
    pm = c.get("_peer_median")
    if pm and price and price < 0.3 * pm:
        add("price_too_low", "The price is far below similar listings in this search, a common counterfeit sign.")
    if rating is not None and adj_rating < 3.5:
        add("low_rating", "Customers rate this below average.")
    if price is None:
        add("no_price", "The price was not shown, so the final cost is unknown.")
    if budget_inr and price and price > budget_inr:
        add("over_budget", "It costs more than your budget.")
    pts = min(100, pts)
    # Missing core evidence is not the same as safe: two or more of price / seller / rating unknown => "Unverified".
    gaps = int(price is None) + int(market and not seller) + int(rating is None)
    level = "High risk" if pts >= HIGH_RISK_AT else ("Unverified" if gaps >= 2 else "Low risk")
    if level == "Unverified":
        why.insert(0, "We could not read enough of this page (price, seller or ratings are missing), so we cannot call it low risk. Open the seller page and check these yourself.")
    if not why:
        why = ["The page opens or was confirmed in search, the address looks normal and nothing unusual was found in what we could read."]
    return {"risk_score": pts, "risk_level": level, "risk_reasons": why,
            "risk_note": "A rule-based signal to help you decide, not proof of fraud. Check the seller before paying."}


def evaluate(candidates: list[dict], source_titles: list[str], budget_inr: float | None) -> dict:
    """Deterministic evaluation of model-proposed candidates. Pure function (unit-tested)."""
    sources = {s.lower().removeprefix("www.") for s in source_titles if s}
    rows = []
    priced = sorted(c["price_inr"] for c in candidates if isinstance(c.get("price_inr"), (int, float)) and c["price_inr"] > 0)
    for i, c in enumerate(candidates, start=1):
        others = [p for p in priced if p != c.get("price_inr")] if len(priced) > 2 else []
        c["_peer_median"] = others[len(others) // 2] if len(others) >= 2 else None
        host = _host(c.get("url", ""))
        state = (c.get("link") or {}).get("state")
        grounded = state not in ("dead", "unreachable") and (state == "ok" or (bool(host) and any(
            host == s or host.endswith("." + s) or (s and s.endswith(host)) for s in sources)))
        rating, n, price = c.get("rating"), c.get("rating_count"), c.get("price_inr")
        r_adj = ((rating * (n or 0)) + PRIOR_RATING * PRIOR_COUNT) / ((n or 0) + PRIOR_COUNT) if rating else PRIOR_RATING
        if price is None or not budget_inr:
            fit = 0.5
        else:
            fit = 1.0 if price <= budget_inr else max(0.0, 1 - (price - budget_inr) / budget_inr)
        score = WEIGHTS["rating"] * (r_adj / 5) + WEIGHTS["price_fit"] * fit + WEIGHTS["grounded"] * (1.0 if grounded else 0.0)
        flags = []
        if state in ("dead", "unreachable"):
            flags.append("link does not open a page" if state == "dead" else "site did not respond")
        elif not grounded:
            flags.append("page not found among the search sources returned -- treat link as unverified")
        if rating and (n or 0) < 10:
            flags.append("rating rests on fewer than 10 ratings (or count unknown)")
        auth = c.get("authenticity")
        if auth and auth.get("suspicious"):
            score = max(0.0, score - AUTH_PENALTY)
            flags.extend("review check: " + x for x in auth.get("reasons", []))
        elif auth is not None and auth.get("review_count", 0) < 3:
            flags.append("review authenticity not assessable (fewer than 3 reviews readable on the page)")
        if price is None:
            flags.append("price not stated")
        if budget_inr and price and price > budget_inr:
            flags.append("over budget")
        risk = risk_assessment(c, grounded, r_adj, budget_inr)
        if risk["risk_level"] == "High risk":
            score = max(0.0, score - 0.2)             # evidence problems must not be outvoted by a nice rating
        notes = []
        for b in c.get("badges") or []:
            if any(k == b.lower().strip() or k in b.lower() for k in POPULARITY_BADGES):
                notes.append(f"Badge '{b}' ignored: it reflects sales or paid placement, not quality or authenticity.")
        if _registrable(host) in MARKETPLACES:
            notes.append("Marketplace listing: the site being trusted is not counted as evidence for this seller or product.")
        rows.append({**c, **risk, "bias_notes": notes, "model_rank": i, "host": host, "grounded": grounded, "link_state": state or "unchecked", "score": round(score, 3),
                     "adjusted_rating": round(r_adj, 2), "flags": flags})
    order = sorted(rows, key=lambda r: (-r["score"], r["model_rank"]))
    for j, r in enumerate(order, start=1):
        r["evaluated_rank"] = j
        r["rank_shift"] = r["model_rank"] - j      # positive: evaluation moved it UP
    n = len(rows)
    rho = None
    if n >= 3:
        d2 = sum((r["model_rank"] - r["evaluated_rank"]) ** 2 for r in rows)
        rho = round(1 - 6 * d2 / (n * (n * n - 1)), 2)
    top_moved = bool(order and order[0]["model_rank"] > 2)
    note = ("The model's top result is not in the evaluated top 2 -- its ordering may reflect position or popularity "
            "rather than your criteria." if top_moved else "The model's ordering broadly agrees with the transparent score.")
    hi = sum(1 for r in order if r["risk_level"] == "High risk")
    return {"high_risk_count": hi, "results": order, "weights": WEIGHTS, "spearman_model_vs_evaluated": rho, "bias_note": note,
            "limits": "Reviews are read from the public page by the model (a visible sample, often only the top reviews), "
                      "then checked with Agentra's rule-based review check. Where fewer than 3 reviews are readable the check "
                      "says so instead of guessing. A flag is a signal, not proof; the page may also block reading."}


REVIEWS = """Read this product page: {url}
List the individual customer reviews that are VISIBLE on the page, copied exactly (do not invent or paraphrase).
Return ONLY a JSON array (max 15) of {{"reviewer": str, "rating": number|null, "date": "YYYY-MM-DD"|null, "text": str}}.
If the page shows no individual reviews or cannot be read, return []."""


def extract_reviews(url: str) -> list[dict]:
    """Fetch the page and ask the model to return visible reviews. Never raises for page problems."""
    try:
        text, _ = _generate(REVIEWS.format(url=url), url_context=True)
        arr = _json_from(text)
    except (AssistantUnavailable, httpx.HTTPError, json.JSONDecodeError):
        return []
    out = []
    for r in arr if isinstance(arr, list) else []:
        if isinstance(r, dict) and str(r.get("text", "")).strip():
            out.append({"reviewer": str(r.get("reviewer") or "").strip(), "rating": r.get("rating"),
                        "date": r.get("date"), "text": str(r["text"])})
    return out


def authenticity(reviews: list[dict]) -> dict:
    """Run Agentra's existing rule-based review check (app/review_authenticity.py) on extracted reviews."""
    from app.catalog import Product
    from app.review_authenticity import check
    usable = []
    for r in reviews:
        try:
            ts = datetime.fromisoformat(r["date"]).isoformat() if r.get("date") else None
        except (ValueError, TypeError):
            ts = None
        if ts:
            usable.append({"reviewer": r["reviewer"], "rating": r.get("rating"), "timestamp": ts, "text": r["text"]})
    # burst check needs dates; if none are readable, assess only diversity/duplicates by giving all the same date-free path
    if len(usable) < 3 and len(reviews) >= 3:
        usable = [{"reviewer": r["reviewer"], "rating": r.get("rating"), "timestamp": "2000-01-01T00:00:00", "text": r["text"]}
                  for r in reviews]
        res = check(Product(id="web", name="", category="", price=0, merchant="", is_new_merchant=False, reviews=usable)).as_dict()
        res["reasons"] = [x for x in res["reasons"] if "within" not in x]
        res["burst_detected"] = False
        res["suspicious"] = bool(res["reasons"])
        res["note"] = "no review dates readable: burst check skipped"
        return res
    return check(Product(id="web", name="", category="", price=0, merchant="", is_new_merchant=False, reviews=usable)).as_dict()


def _history_text(h: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['text']}" for m in h[-12:])


DECIDE = """You are a careful shopping assistant that talks like a helpful person. Read the conversation and decide ONE action.
Return JSON only:
{{"action": "ask" | "search" | "chat",
  "question": "<ONE short clarifying question, only if essential info (what, budget or who it is for) is missing>",
  "query": "<a precise web search query for product pages, when action is search>",
  "budget_inr": <number or null>,
  "exclude_brand_origin": "<country the user does NOT want brands from, e.g. india, china, or null>",
  "require_brand_origin": "<country the user ONLY wants brands from, or null>",
  "reply": "<plain answer, when action is chat, e.g. explaining current results>"}}
Rules: ask at most one question at a time; do not ask if you can search sensibly; never invent products.
Constraints already active in this chat: {constraints}
KEEP every active constraint (budget, brand origin) in your JSON unless the user changes it or moves to a different product. Put the constraints into the search query too.
Current results (if any): {results}
Conversation:
{history}"""

RULES_PATH = os.path.join(os.path.dirname(__file__), "prompts", "web_sourcing_rules.md")
_FALLBACK_RULES = ("Only return URLs you actually saw in search results; never construct or guess a URL. Product pages only, "
                   "India store only, prefer the brand's India site or large trusted marketplaces. Use null for anything "
                   "not shown on the page.")


def sourcing_rules() -> str:
    """Instructions file for the model (editable, re-read each search). Comment lines (#) are dropped."""
    try:
        with open(RULES_PATH, encoding="utf-8") as f:
            txt = "\n".join(l for l in f.read().splitlines() if not l.lstrip().startswith("#")).strip()
        return txt or _FALLBACK_RULES
    except OSError:
        return _FALLBACK_RULES


SEARCH = """Use web search to find real product pages for this shopping need: {query}
Budget (INR): {budget}
Return ONLY a JSON array (max 8) ordered by how well they fit, each item:
{{"name": str, "url": str (the actual product page), "price_inr": number|null, "rating": number|null,
  "rating_count": integer|null, "why": str (<=20 words, factual)}}
Follow these sourcing rules exactly:
{rules}
{retry}"""


PAGE = """Read this product page: {url}
Return ONLY JSON: {{"is_product_page": bool (true only if this is ONE specific product you can buy; false for home, category,
search, blog or roundup pages), "name": str|null, "seller": str|null, "price_inr": number|null (only if shown in rupees),
"rating": number|null, "rating_count": integer|null (how many people rated it; 0 if the page says no ratings/reviews yet),
"seller_rating": number|null (the SELLER's own rating, not the product's), "seller_rating_count": integer|null,
"fulfilled_by_platform": bool|null, "mrp_inr": number|null (the struck-through list price, if shown),
"badges": [str] (labels such as Bestseller, Amazon's Choice, Sponsored, Assured, exactly as shown; [] if none),
"reviews": [{{"reviewer": str, "rating": number|null, "date": "YYYY-MM-DD"|null, "text": str}}] (visible reviews copied exactly, max 15)}}
Use null/[] when not shown or the page cannot be read. Never invent anything."""


def _read_page(url: str) -> dict:
    """Fetch one page and ask the model to read it. Returns {} when unreadable. Never raises for page problems."""
    try:
        text, _ = _generate(PAGE.format(url=url), url_context=True)
        facts = _json_from(text)
    except (AssistantUnavailable, httpx.HTTPError, json.JSONDecodeError) as exc:
        print(f"[agentra] page read failed for {_host(url)}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return {}
    return facts if isinstance(facts, dict) else {}


def _market_fields(f: dict) -> dict:
    """Seller and popularity facts read from the page, carried into the risk check."""
    num = lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    return {"seller": str(f["seller"]) if f.get("seller") else None, "seller_rating": num(f.get("seller_rating")),
            "seller_rating_count": num(f.get("seller_rating_count")), "fulfilled_by_platform": f.get("fulfilled_by_platform"),
            "mrp_inr": num(f.get("mrp_inr")), "badges": [str(b)[:40] for b in (f.get("badges") or []) if b][:6]}


def _clean_reviews(facts: dict) -> list[dict]:
    return [{"reviewer": str(r.get("reviewer") or ""), "rating": r.get("rating"), "date": r.get("date"), "text": str(r.get("text", ""))}
            for r in facts.get("reviews", []) if isinstance(r, dict) and str(r.get("text", "")).strip()]


def check_url(url: str) -> dict:
    """Review ANY product link: open it, read what the page shows, run the same authenticity + risk rules."""
    if "://" not in url:
        url = "https://" + url
    link = check_link(url)
    facts = _read_page(url) if link["state"] in ("ok", "blocked") else {}
    host = _host(link["final_url"] or url)
    revs = _clean_reviews(facts)
    cand = {"name": facts.get("name") or host or url, "url": link["final_url"] if link["state"] != "dead" else url,
            "price_inr": facts.get("price_inr"), "rating": facts.get("rating"), "rating_count": facts.get("rating_count"),
            "why": ("Seller: " + str(facts["seller"])) if facts.get("seller") else "", **_market_fields(facts),
            "link": link, "url_risk": url_risk(url) + page_quality(url), "authenticity": authenticity(revs)}
    ev = evaluate([cand], [host] if link["state"] in ("ok", "blocked") else [], None)
    row = ev["results"][0]
    if link["state"] == "blocked":
        row["flags"].append("this site blocks automated reading, so ratings and reviews could not be checked")
    if not facts:
        row["flags"].append("could not read the product details from the page")
    return ev


def _retry_failed(query: str, budget, cands: list[dict], resolved: list[dict]) -> list[dict]:
    """One corrective pass: tell the model which links failed and why, ask for working replacements, verify them."""
    bad = [c for c in cands if (c.get("link") or {}).get("state") in ("dead", "unreachable")
           or any(k in ("not_product_page", "wrong_region") for k, _ in c.get("url_risk", []))]
    if not bad or os.environ.get("ASSISTANT_RETRY", "1") == "0":
        return cands
    note = ("These links FAILED when opened or were not suitable (dead page, no response, home/category page or wrong "
            "country): " + "; ".join(f"{c.get('name','?')} -> {c['url']}" for c in bad[:6]) +
            ". Find real, working India product pages for these or similar products. Do not reuse these URLs and do not "
            "alter them to guess a fix.")
    try:
        found, chunks = _generate(SEARCH.format(query=query, budget=budget or "not stated", rules=sourcing_rules(), retry=note),
                                  search=True, queries=[f"{query} buy online India"])
        new = _json_from(found)
    except (AssistantUnavailable, httpx.HTTPError, json.JSONDecodeError):
        return cands
    new = [c for c in (new if isinstance(new, list) else []) if isinstance(c, dict) and c.get("url")][:len(bad)]
    if not new:
        return cands
    verify_candidates(new, resolve_sources(chunks) + resolved)
    seen = {c["url"] for c in cands}
    good_new = [c for c in new if c["url"] not in seen and (c.get("link") or {}).get("state") in ("ok", "blocked")
                and not c.get("url_risk")]
    keep = [c for c in cands if c not in bad]
    # failed items that could not be replaced stay visible (flagged) so the user sees what was dropped
    return keep + good_new + (bad if not good_new else [])


NAMES = """Use web search to find the best matching SPECIFIC products for this shopper need in India: {query}
Return ONLY a JSON array (max {n}) of {{"name": str (brand + exact model), "brand": str}}.
Real, currently sold products only. No categories, no articles, no "best of" lists.
{budget_line}Follow these sourcing rules:
{rules}"""
FIND = """Use web search to find where to buy each of these products in India (search amazon.in, flipkart.com and the
brand's official India site for each one): {names}
Reply with one short line per product."""
MIN_RESULTS = 3
N_PRODUCTS = 5
READ_TOP = int(os.environ.get("ASSISTANT_READ_TOP", "4"))   # pages the model reads per search (quota: 1 call each)


def _pick_page(name: str, resolved: list[dict]) -> dict | None:
    """Best real product page for a product name among resolved search sources (title/URL overlap, trusted sites first)."""
    want = _tokens(name)
    best, best_score = None, 0
    for r in resolved:
        if page_quality(r["url"]) or url_risk(r["url"]):
            continue
        overlap = len(want & _tokens(r["title"] + " " + urlparse(r["url"]).path))
        score = overlap + (0.5 if _registrable(r["host"]) in KNOWN_SITES else 0)
        if overlap >= 2 and score > best_score:
            best, best_score = r, score
    return best


_LAST_TRACE: dict = {}      # where the last search lost its candidates (shown when nothing survives)


def source_from_search(query: str, budget: float | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """ROUTE A (preferred). The model never writes a URL.
    1) The model names specific products. 2) A web-search call returns real pages; we open them to get the real
    final URL and keep only a product page (no category/roundup/other-country page). 3) The model READS each page for
    price, rating, reviews and must confirm it is one specific product. Returns (candidates, resolved, chunks)."""
    cap = f" under ₹{budget:.0f}" if budget else ""
    bline = (f"The shopper's budget is ₹{budget:.0f}: every product you name must normally sell for LESS than ₹{budget:.0f} in India "
             f"(prefer ones between 50% and 100% of it). Do not name anything dearer.\n") if budget else ""
    t0 = time.perf_counter(); tm: dict[str, int] = {}
    text, _ = _generate(NAMES.format(query=query, n=N_PRODUCTS, rules=sourcing_rules(), budget_line=bline), search=True,
                        queries=[f"{query}{cap} buy online India", f"best {query}{cap} India amazon.in flipkart"])
    tm["names_ms"] = int((time.perf_counter() - t0) * 1000)
    names = _names_from(text)
    raw = (text or "")[:160].replace("\n", " ")
    if not names:      # the model answered in prose or an unexpected shape: ask once more in strict JSON mode
        text2, _ = _generate(NAMES.format(query=query, n=N_PRODUCTS, rules=sourcing_rules(), budget_line=bline) +
                             '\nReply with ONLY this JSON object: {"products": [{"name": "...", "brand": "..."}]}',
                             search=True, as_json=True, queries=[f"{query}{cap} buy online India"])
        names, raw = _names_from(text2), (text2 or "")[:160].replace("\n", " ")
    _LAST_TRACE.clear(); _LAST_TRACE.update(names=len(names), pages=0, matched=0, readable=0)
    if not names:
        print("[agentra] model gave no product names. Raw reply:", repr(raw), flush=True)

    if not names:
        return [], [], []
    find_queries = [f"{n} buy online India amazon.in flipkart" for n in names]
    if _search_backend() == "compound":     # the search happens inside the model call, so the call is needed
        _, chunks_all = _generate(FIND.format(names="; ".join(names)), search=True, queries=find_queries)
    else:                                   # Tavily / DuckDuckGo: only the search results were ever used, so skip the model call
        chunks_all = [{"title": x["title"], "uri": x["uri"]} for x in _external_search(find_queries)]
    tm["search_ms"] = int((time.perf_counter() - t0) * 1000) - tm["names_ms"]
    resolved_all = resolve_sources(chunks_all)
    tm["links_ms"] = int((time.perf_counter() - t0) * 1000) - tm["names_ms"] - tm["search_ms"]
    _LAST_TRACE["pages"] = len(resolved_all)
    pool, seen = [], set()
    for name in names:
        pick = _pick_page(name, resolved_all)
        if pick:
            key = (pick["host"], urlparse(pick["url"]).path.rstrip("/"))
            if key not in seen:
                seen.add(key)
                pool.append((name, pick))
    pool = pool[:READ_TOP]
    _LAST_TRACE["matched"] = len(pool)
    with cf.ThreadPoolExecutor(max_workers=max(1, min(len(pool), int(os.environ.get("ASSISTANT_READ_WORKERS", "2"))))) as ex:
        facts = list(ex.map(lambda np: _read_page(np[1]["url"]), pool))
    _LAST_TRACE["readable"] = sum(1 for f in facts if f.get("is_product_page") is True)
    tm["read_ms"] = int((time.perf_counter() - t0) * 1000) - tm["names_ms"] - tm["search_ms"] - tm["links_ms"]
    print("[agentra] sourcing trace:", dict(_LAST_TRACE), "timing:", tm, flush=True)
    cands = []
    for (name, r), f in zip(pool, facts):
        if f.get("is_product_page") is not True:      # unreadable or not one specific product: do not show it
            continue
        cands.append({"name": f.get("name") or name, "url": r["url"], "price_inr": f.get("price_inr"),
                      "rating": f.get("rating"), "rating_count": f.get("rating_count"),
                      "why": ("Seller: " + str(f["seller"])) if f.get("seller") else "", **_market_fields(f),
                      "link": {"state": r["state"], "final_url": r["url"]}, "url_risk": [],
                      "authenticity": authenticity(_clean_reviews(f)), "from_search": True})
    return cands, resolved_all, chunks_all


def _llm_link_route(query: str, budget) -> tuple[list[dict], list[dict], list[dict]]:
    """ROUTE B (fallback): the model proposes URLs; every one is opened, repaired or flagged."""
    found, chunks = _generate(SEARCH.format(query=query, budget=budget or "not stated", rules=sourcing_rules(), retry=""), search=True,
                              queries=[f"{query} buy online India"])
    cands = _json_from(found)
    cands = [c for c in (cands if isinstance(cands, list) else []) if isinstance(c, dict) and c.get("url")][:8]
    resolved = resolve_sources(chunks)
    verify_candidates(cands, resolved)
    return _retry_failed(query, budget, cands, resolved), resolved, chunks


# ---------------------------------------------------------------- user constraints (budget, brand origin)
# Brand origin is NOT in product pages reliably, so it comes from this reviewed lookup (first words of the product title).
# Unknown brands are kept but labelled "origin not verified" -- never guessed.
_ORIGINS = {
    "India": "boat noise boult mivi fire-boltt fireboltt zebronics ptron ubon portronics lava gizmore crossbeats pebble micromax "
             "campus sparx liberty wildcraft titan fastrack fabindia raymond woodland mamaearth himalaya lakme prestige bajaj havells "
             "usha crompton orient pigeon butterfly ifb godrej voltas wipro syska boldfit urbanista-india",
    "USA": "jbl apple bose skullcandy google hp dell whirlpool nike levi's levis fitbit harman jbl-harman ge",
    "Japan": "sony panasonic hitachi uniqlo audio-technica yamaha casio",
    "South Korea": "samsung lg",
    "China": "oneplus realme xiaomi redmi mi poco oppo vivo anker soundcore lenovo tcl haylou qcy huawei honor",
    "Taiwan": "asus acer htc",
    "Germany": "sennheiser adidas puma bosch",
    "Denmark": "jabra",
    "UK": "nothing",
    "Netherlands": "philips",
    "Spain": "zara",
    "Sweden": "h&m",
}
BRAND_ORIGIN = {b: c for c, bs in _ORIGINS.items() for b in bs.split()}
_ADJ = {"indian": "india", "chinese": "china", "american": "usa", "us": "usa", "japanese": "japan", "korean": "south korea",
        "german": "germany", "taiwanese": "taiwan", "danish": "denmark", "british": "uk", "uk": "uk", "dutch": "netherlands"}


# The model sometimes writes the *word* "null"/"none" instead of JSON null. That used to become a real country filter
# ("oneplus is from China, not Null") and wiped out every result, so these all mean "no constraint".
_NULLISH = {"", "null", "none", "nil", "n/a", "na", "any", "anything", "no", "false", "unknown", "unspecified",
            "not stated", "not specified", "no preference", "nan", "-"}
_COUNTRY_WORDS = {"india": ("india", "indian"), "china": ("china", "chinese"), "usa": ("usa", "u.s.", "america", "american"),
                  "japan": ("japan", "japanese"), "south korea": ("korea", "korean"), "germany": ("germany", "german"),
                  "taiwan": ("taiwan", "taiwanese"), "denmark": ("denmark", "danish"), "uk": ("uk", "britain", "british"),
                  "netherlands": ("netherlands", "dutch", "holland")}


def _country(x) -> str | None:
    x = (str(x).strip().lower() if x else "")
    if x in _NULLISH:
        return None
    return _ADJ.get(x, x) or None


def _user_named(country: str | None, history: list[dict]) -> bool:
    """True only if the USER's own messages mention this country. A model-invented origin rule is never enforced."""
    if not country:
        return False
    mine = " ".join(h["text"] for h in history if h.get("role") == "user")
    low = mine.lower()
    for w in _COUNTRY_WORDS.get(country, (country,)):
        if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low):
            return True
    return country == "usa" and bool(re.search(r"\bUS\b", mine))


_BUD_RE = re.compile(r"(?:under|below|within|upto|up to|max(?:imum)?|less than|budget(?: of| is)?|around|₹|rs\.?|inr)\s*₹?\s*(\d[\d,]*(?:\.\d+)?)", re.I)
_PRICE_WORDS = re.compile(r"\b(?:under|below|within|upto|up to|max(?:imum)?|less than|budget(?: of| is)?)\s*(?:₹|rs\.?|inr)?\s*\d[\d,]*\b|(?:₹|rs\.?|inr)\s*\d[\d,]*", re.I)


def parse_budget(text: str, allow_bare: bool = False) -> float | None:
    """Budget the user typed in plain words ('under 50', 'Max 25', 'rs.20'); a bare number only counts for a short follow-up reply."""
    m = _BUD_RE.search(text or "")
    if m:
        return float(m.group(1).replace(",", ""))
    if allow_bare:
        m = re.fullmatch(r"\s*(?:₹|rs\.?)?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:inr|rs\.?|rupees|\?)?\s*", text or "", re.I)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def brand_of(name: str) -> tuple[str | None, str | None]:
    """(brand, origin country) from the first words of the product title; origin None when the brand is not in the lookup."""
    toks = re.findall(r"[\w&'-]+", (name or "").lower())[:2]
    for t in toks:
        if t in BRAND_ORIGIN:
            return t, BRAND_ORIGIN[t]
    return (toks[0] if toks else None), None


def apply_constraints(cands: list[dict], budget: float | None, exclude_origin=None, require_origin=None) -> tuple[list[dict], list[str]]:
    """Hard filters the user asked for. Over-budget and wrong-origin items are removed (not just down-ranked)."""
    ex, rq = _country(exclude_origin), _country(require_origin)
    kept, dropped = [], []
    for c in cands:
        brand, origin = brand_of(c.get("name", ""))
        c["brand"], c["brand_origin"] = brand, origin
        price = c.get("price_inr")
        if budget and isinstance(price, (int, float)) and price > budget:
            dropped.append(f"{c.get('name', '?')[:50]} (₹{price:.0f} is over your ₹{budget:.0f} budget)")
        elif ex and origin and origin.lower() == ex:
            dropped.append(f"{c.get('name', '?')[:50]} ({brand} is a {origin.title() if origin != 'USA' else origin} brand)")
        elif rq and origin and origin.lower() != rq:
            dropped.append(f"{c.get('name', '?')[:50]} ({brand} is from {origin}, not {rq.title()})")
        else:
            if (ex or rq) and not origin:
                c["brand_note"] = f"Brand origin not verified ({brand or 'unknown brand'}): check it yourself."
            kept.append(c)
    return kept, dropped


def closest_misses(pool: list[dict], budget: float | None, exclude_origin=None, require_origin=None, n: int = 3) -> list[dict]:
    """When nothing meets every limit, show the nearest misses instead of an empty answer. Each one is labelled with exactly
    how it breaks the user's rule, so a limit is never silently ignored."""
    ex, rq = _country(exclude_origin), _country(require_origin)
    scored = []
    for c in pool:
        brand, origin = brand_of(c.get("name", ""))
        c["brand"], c["brand_origin"] = brand, origin
        price, notes, miss = c.get("price_inr"), [], 0.0
        if budget and isinstance(price, (int, float)) and price > budget:
            notes.append(f"₹{price - budget:.0f} over your ₹{budget:.0f} budget")
            miss += (price - budget) / budget
        if ex and origin and origin.lower() == ex:
            notes.append(f"{brand} is a {origin.title() if origin != 'USA' else origin} brand, which you asked to avoid")
            miss += 1.0
        if rq and origin and origin.lower() != rq:
            notes.append(f"{brand} is from {origin}, not {rq.title()}")
            miss += 1.0
        c["brand_note"] = ("Closest match, but outside your limits: " + "; ".join(notes) + ".") if notes else c.get("brand_note")
        scored.append((miss, c))
    scored.sort(key=lambda t: t[0])
    return [c for _, c in scored[:n]]


def _converse(session_id: str | None, message: str) -> dict:
    sid = session_id if session_id in _SESSIONS else uuid.uuid4().hex[:12]
    st = _SESSIONS.setdefault(sid, {"history": [], "results": None})
    st.setdefault("constraints", {})
    st["history"].append({"role": "user", "text": message})
    m = URL_RE.search(message)
    if m:   # a pasted link: review that page directly
        ev = check_url(m.group(0).rstrip(".,);]"))
        r0 = ev["results"][0]
        st.setdefault("seen", {})[r0.get("url")] = r0
        reply = (f"I checked {r0['host'] or 'that link'}. Result: {r0['risk_level']}. "
                 "Open 'Why' to see exactly what I found and what I could not check.")
        st["history"].append({"role": "assistant", "text": reply})
        return {"session_id": sid, "stage": "url_check", "reply": reply, "results": ev}
    summary = json.dumps([{k: r[k] for k in ("evaluated_rank", "name", "price_inr", "rating", "host")}
                          for r in (st["results"] or {}).get("results", [])][:8])
    cons = st["constraints"]
    text, _ = _generate(DECIDE.format(results=summary, history=_history_text(st["history"]),
                                      constraints=json.dumps({k: v for k, v in cons.items() if v}) if any(cons.values()) else "none"), as_json=True,
                         prefer=os.environ.get("ASSISTANT_FAST_MODEL") or None)
    d = _json_from(text) or {}
    action = d.get("action")
    out: dict = {"session_id": sid, "stage": "chat", "results": None}
    prior = (st["constraints"] or {}).get("budget_inr")
    typed = parse_budget(message, allow_bare=bool(prior) or action == "ask" or len(message.split()) <= 2)
    asked_budget = action == "ask" and re.search(r"budget|price|spend|afford|how much|maximum", d.get("question") or "", re.I)
    if asked_budget and (typed or prior or d.get("budget_inr")) :
        # the user already gave a budget (or just answered it): do not ask again, search
        action = "search"
        d["query"] = d.get("query") or st.get("last_query") or next((h["text"] for h in st["history"] if h["role"] == "user" and not parse_budget(h["text"], True)), message)
    if action == "search" and d.get("query"):
        budget = d.get("budget_inr")
        bud = budget if isinstance(budget, (int, float)) else (typed or (prior if len(message.split()) <= 4 else None))
        st["last_query"] = _PRICE_WORDS.sub("", d["query"]).strip() or d["query"]
        ex_o, rq_o = _country(d.get("exclude_brand_origin")), _country(d.get("require_brand_origin"))
        if ex_o and not _user_named(ex_o, st["history"]):
            print(f"[agentra] ignored a brand-origin rule the user never stated: exclude {ex_o!r}")
            ex_o = None
        if rq_o and not _user_named(rq_o, st["history"]):
            print(f"[agentra] ignored a brand-origin rule the user never stated: require {rq_o!r}")
            rq_o = None
        st["constraints"] = {"budget_inr": bud, "exclude_brand_origin": ex_o, "require_brand_origin": rq_o}
        cands, resolved, chunks = source_from_search(d["query"], bud)
        if not cands and _PRICE_WORDS.search(d["query"]):   # "under 25" in the query can drown the search: retry without the price words
            cands, resolved, chunks = source_from_search(st["last_query"], bud)
        if len(cands) < MIN_RESULTS:   # not enough real, readable product pages: add checked model-proposed links
            extra, res2, ch2 = _llm_link_route(d["query"], budget)
            have = {c["url"] for c in cands}
            cands += [c for c in extra if c["url"] not in have and (c.get("link") or {}).get("state") == "ok" and not c.get("url_risk")]
            resolved, chunks = resolved + res2, chunks + ch2
        if not cands and bud:   # nothing readable at this price: look again without the cap so there is always something to show
            cands, resolved, chunks = source_from_search(st["last_query"], None)
        c0 = st["constraints"]
        over = sorted(c["price_inr"] for c in cands if bud and isinstance(c.get("price_inr"), (int, float)) and c["price_inr"] > bud)
        pool = [dict(c) for c in cands]
        cands, dropped = apply_constraints(cands, bud, c0.get("exclude_brand_origin"), c0.get("require_brand_origin"))
        relaxed = False
        if not cands and dropped and pool:   # never an empty answer: show the nearest misses, each labelled with how it misses
            cands, relaxed = closest_misses(pool, bud, c0.get("exclude_brand_origin"), c0.get("require_brand_origin")), True
        if not cands:
            t = _LAST_TRACE
            why = (f" (searched: {t.get('names', 0)} product names, {t.get('pages', 0)} live pages, {t.get('matched', 0)} matched, "
                   f"{t.get('readable', 0)} readable as a product page)") if t else ""
            reply = ("I searched but could not find working product pages I can stand behind" + why +
                     (f" at ₹{bud:.0f} or less. That price may simply be too low for this item: want me to look without the limit?" if bud
                      else ". Could you narrow it down (item type, budget)?"))
        else:
            titles = [c.get("title", "") for c in chunks] + [r["host"] for r in resolved]
            need = []
            for r in evaluate(cands, titles, bud)["results"][:AUTH_TOP_N]:
                src = next(c for c in cands if c.get("url") == r["url"])
                if r["grounded"] and "authenticity" not in src:
                    need.append((src, r["url"]))
            if need:                              # each one is a page fetch plus a model call: do them together
                with cf.ThreadPoolExecutor(max_workers=len(need)) as ex:
                    for (src, _u), revs in zip(need, ex.map(lambda t: extract_reviews(t[1]), need)):
                        src["authenticity"] = authenticity(revs)
            ev = evaluate(cands, titles, bud)
            st["results"] = ev
            for r in ev["results"]:
                st.setdefault("seen", {})[r.get("url")] = r
            out.update(stage="results", results=ev)
            reply = ("Here is my evaluated shortlist. The pages come straight from search results (not from links the "
                     "model wrote), and each page was opened and read. " + ev["bias_note"])
            if relaxed:
                reply = ("Nothing I found met every limit you set, so these are the closest matches. Each one is marked with exactly "
                         "how it misses. Raise the budget or relax a rule if you want a clean fit. " + ev["bias_note"])
            elif dropped:
                reply += " I left out " + str(len(dropped)) + " that broke your rules: " + "; ".join(dropped[:3]) + "."
    elif action == "ask" and d.get("question"):
        out["stage"] = "asking"
        reply = d["question"]
    else:
        reply = d.get("reply") or "Tell me what you're shopping for and your rough budget."
    st["history"].append({"role": "assistant", "text": reply})
    out["reply"] = reply
    return out



# ---------------------------------------------------------------- activity log
# Every assistant turn is appended to the SAME tamper-evident, hash-chained audit log the agent uses
# (data/audit/purchases.jsonl), so the dashboard's "audit chain intact" check covers it too. Logging never breaks a chat.
ASSISTANT_EVENTS = ("assistant_user_message", "assistant_shortlist", "assistant_url_check", "assistant_reply", "assistant_error",
                    "assistant_demo_scenarios", "assistant_gate_decision", "assistant_human_decision", "assistant_order_blocked", "wallet_debit", "wallet_refund")


def _audit(event: str, payload: dict) -> None:
    try:
        from app import audit_log
        audit_log.append(event, payload)
    except Exception:       # a full disk or a locked file must not take the assistant down
        pass


def _row(r: dict) -> dict:
    return {k: r.get(k) for k in ("evaluated_rank", "name", "host", "url", "price_inr", "rating", "rating_count",
                                  "risk_level", "risk_score", "grounded") if k in r} | {"flags": (r.get("flags") or [])[:6]}


def converse(session_id: str | None, message: str) -> dict:
    """Run one assistant turn and record it in the audit log (message, outcome, what was shown, errors)."""
    sid = session_id or ""
    _audit("assistant_user_message", {"session_id": sid, "text": message[:300], "has_link": bool(URL_RE.search(message))})
    try:
        out = _converse(session_id, message)
    except AssistantUnavailable as exc:
        _audit("assistant_error", {"session_id": sid, "error": str(exc)[:300]})
        raise
    except Exception as exc:                      # unexpected: record it, then let the API return its normal 500
        _audit("assistant_error", {"session_id": sid, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        raise
    sid = out.get("session_id", sid)
    res = (out.get("results") or {})
    rows = res.get("results") or []
    if out.get("stage") == "url_check" and rows:
        _audit("assistant_url_check", {"session_id": sid, **_row(rows[0])})
    elif out.get("stage") == "results":
        _audit("assistant_shortlist", {"session_id": sid, "count": len(rows), "high_risk_count": res.get("high_risk_count"),
                                       "spearman_model_vs_evaluated": res.get("spearman_model_vs_evaluated"),
                                       "items": [_row(r) for r in rows[:8]]})
    else:
        _audit("assistant_reply", {"session_id": sid, "stage": out.get("stage"), "reply": str(out.get("reply", ""))[:300]})
    return out


def activity_log(limit: int = 100, session_id: str | None = None) -> dict:
    """Newest-first assistant events from the audit log, plus whether the whole hash chain still verifies."""
    from app import audit_log
    entries = [e for e in audit_log._read() if e.get("event") in ASSISTANT_EVENTS
               and (not session_id or e.get("payload", {}).get("session_id") == session_id)]
    return {"chain": audit_log.verify(), "total": len(entries), "events": list(reversed(entries))[:max(1, min(limit, 500))]}
