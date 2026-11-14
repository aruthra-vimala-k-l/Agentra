# Agentra — Mechanism Note (for the review)

How the Intercept → Explain → Secure → Govern pipeline decides, in the form that is actually implemented and tested.
Everything below runs on the real catalogue (Amazon Reviews 2023, 1,029 products). Attack cases are labelled constructed fixtures.

## 1. Counterfactual Divergence Check (CDC v2) — `app/counterfactual.py`
Question answered: *did the agent choose something the user's stated intent does not justify, and how costly is that if wrong?*

1. Re-rank the candidates using **trusted signals only** (price_fit, durability_match, occasion_match, new_merchant_caution). Reviews, ratings and promotion are excluded because they are attacker-influenceable.
2. Score gap **S** = clamp((clean(top) − clean(chosen)) / 3.0, 0, 1).
3. Materiality **M** = 0.5·d_sem + 0.5·d_attr.
   - d_sem = 1 − cosine(TF-IDF(chosen), TF-IDF(top)), vectors fitted on the real catalogue.
   - d_attr = 0.5·min(1, |ln(price ratio)| / ln 3) + 0.5·[category differs].
4. Divergence **D** = S·(0.25 + 0.75·M).
5. **Weighted by irreversibility:** W = D·(1 + IRC/100), where IRC is the 0–100 Irreversibility Cost score.
6. ESCALATE if W ≥ 0.30. BLOCK if W ≥ 0.60 **and** irreversibility is HIGH/CRITICAL.
7. Applied only to the agent's own top recommendation. If the user explicitly picked the item, CDC is informational (the user's choice is not manipulation).

Thresholds 0.30/0.60 and SCORE_NORM 3.0 are design constants, not fitted parameters.

## 2. Anomaly Detection Module (ADM) — `app/anomaly.py`
Three independent detectors fitted on the real catalogue only:
- Per-category robust z-score on ln(price): z = 0.6745·(ln p − median)/MAD; |z| ≥ 2.5 unusual, ≥ 3.5 very unusual.
- Rating-vs-review-count rule (e.g. near-perfect rating on a handful of reviews).
- Isolation Forest (200 trees, random_state 0; features ln price, z, rating, ln(1+reviews)), reported as a percentile of training scores (≥ 97 unusual, ≥ 99 very).
Combined score 0–100: NORMAL < 30 ≤ ELEVATED < 60 ≤ HIGH. On the real catalogue: 970 NORMAL / 41 ELEVATED / 18 HIGH. Needs ≥ 30 products per category to fit.

## 3. Hardened injection detection — `ASIDE app/security/injection.py`
- Normalisation before matching: NFKC, URL-decode, zero-width strip, homoglyph map, de-spacing of spaced-out letters, two leetspeak variants, base64 payload decode.
- 11 named pattern families: override_instructions, reveal_prompt, role_hijack, disable_security, follow_instead, concealment, payment_redirect, approval_forgery, exfiltration, fake_markup, authority_claim.
- Deterministic and explainable (family + matched snippet). The gate now receives the **real listing text** (name, description, up to 8 review bodies) as `tool_output`, so a poisoned listing is visible to the gate.

## 4. Signal trust model — how Agentra talks to ASIDE
Agentra sends `metadata["agentra"] = {cdc, irreversibility, anomaly, review_authenticity}`.
- Signals can only **raise** risk; malformed values are ignored; nothing Agentra sends can lower an ASIDE decision.
- Rules (`app/security/guardrails.py`): CDC W ≥ 0.60 with HIGH/CRITICAL irreversibility → DENY; W ≥ 0.30 → human approval; CRITICAL irreversibility → escalate; anomaly ≥ 60 → escalate; anomaly ≥ 90 with a new merchant → DENY; suspicious reviews → escalate.
- Agentra also keeps its own local escalations (defence in depth), so a gate outage or an ALLOW from ASIDE cannot hide a local HIGH anomaly.

## 5. Measured results (`scripts/attack_harness.py --offline`, ASIDE rule layer only)
| | |
|---|---|
| Benign cases (real catalogue purchases) | 300 |
| Attack cases (constructed fixtures, 6 families) | 129 |
| Detection (any non-ALLOW) | 129/129 |
| Hard-blocked (DENY) | 58.9 % |
| False positives (any non-ALLOW on benign) | 7.0 % (12 are the ₹5,000 autonomous-limit policy, 9 are review-authenticity flags) |
| False positives excluding the amount-limit policy | 3.0 % |
| Hard DENY on benign | 0.0 % |
| Gate latency | p50 ≈ 8 ms, p95 ≈ 13 ms (local process, rules only) |

## 6. Limits — state these before being asked
- **The 100 % detection is in-sample.** The first run missed 4 phrasings (authority claim, card-detail exfiltration, "instead of the user's request", spaced-out letters); the patterns were then fixed and the same fixtures re-run. It shows the rules work on these cases, not that unseen attacks are caught. Section 7 adds two published injection corpora (AgentDojo, WASP) and shows how far the first rules generalised: poorly.
- Section 7 reports the full ASIDE service (ML + RAG + rules) run once, in a sandbox with a stand-in embedder; re-run it on your own stack before the panel.
- CDC and irreversibility signals in two attack families are set directly at the threshold levels (they test the gate's rules, not CDC's ability to generate those values).
- Review-authenticity flags 9 of 300 real products; whether those reviews are truly fake is not known (no ground truth).
- Thresholds are design constants. Attribution fidelity is measured (section 7); the human comprehension study is a ready-to-run kit with NO participant data.
- The web assistant has been tested with mocks and the user's own runs, not an automated live-web suite.

## 7. Additional evaluation (published corpora, full service, attribution)

### 7.1 AgentDojo and WASP injection text — what they are and are not
- AgentDojo v0.1.35 (NeurIPS 2024): 5 attack templates verbatim + 27 injection goals extracted from its banking / travel / workspace / slack suites = 135 strings. WASP (Meta, 2025): 21 attacker instructions x 4 message formats = 84 strings. Frozen in `data/eval/` (WASP file sha256 c3aa4c97…).
- We test whether the **payment gate flags the injection text when it arrives inside a shopping listing**. We do NOT run the benchmarks' agent environments, so this is not an AgentDojo/WASP score and says nothing about whether an LLM agent would have obeyed the text.

| Corpus | Detector as first written | After generalising the patterns |
|---|---|---|
| AgentDojo (135) | **25.9 %** (InjecAgent template 27/27; TODO-style 3/27; "ignore previous" 2/27; important-instructions 2/27; system-message 1/27) | 100 % — now in-sample |
| WASP (84), template wording read but not tuned against | **10.7 %** | 100 % — now in-sample |

The first column is the honest generalisation number. Regex rules written from our own attack list missed most of what other researchers wrote. The second column only shows the new families (typo-tolerant "ignore previous", agent-addressing phrases, sensitive-action imperatives, urgent-directive wording, wider fake markup) cover these strings. No untouched corpus remains; a new one would be needed for an unbiased number.

### 7.2 Full ASIDE service (Postgres + pgvector, trained ML model, RAG, rules) — 948 cases
Run once in a cloud sandbox (model re-trained there with ASIDE's own `python -m app.ml.train`; sentence-transformers could not be downloaded, so retrieval used a stand-in hashed embedder — retrieval evidence therefore differs from your Docker stack).
- Detection (any non-ALLOW): 100 % on all three sets (own 129, AgentDojo 135, WASP 84); hard-blocked: 58.9 % / 100 % / 100 %.
- Real purchases: **0 % hard-DENY**; 31.3 % asked for human approval, of which 67 of 94 (own set) are purchases of ₹2,500 or more — the ML model learned an approval band from its synthetic scenarios, which is stricter than the ₹5,000 autonomous-limit rule. Below ₹2,500 the approval rate is 11.7 %.
- Latency: p50 ≈ 33 ms, p95 ≈ 46 ms per decision (local).
- **Bug found and fixed by this run:** before the fix the ML model hard-denied 64–67 % of real purchases, because it had only seen short synthetic tool outputs and misread long real listing text. `injection.model_view()` now gives the model a bounded view of long text (matched attack snippets, or a neutral token); short text is untouched. Deterministic injection detection still sees the full text.
- The ML model is still trained on synthetic scenarios only (no production history exists).

### 7.3 Attribution fidelity (`scripts/attribution_fidelity.py`, real ranker, real catalogue)
Intervention test: neutralise each signal's source attribute, re-run the real scoring function, compare the actual score drop with the Shapley value. 12 real queries x top-3 products = 36 cases x 7 signals.
- Efficiency error: 0.0. Mean |Shapley − actual drop| = 0.006 (random permutation: 0.82); Spearman 1.0; top-signal agreement 100 % (random: 2.8 %).
- Baseline "raw signal value" is also accurate (MAE 0.012): the ranker is additive plus one synergy term, so a simple explanation is already nearly faithful. Shapley's measurable gain is on the synergy pair (durability x occasion), where it splits the 0.5 bonus; the remaining residual there is 0.02.
- The sponsored-placement signal has Shapley value 0 and zero effect on rank, confirming the business rule that promotion does not buy rank.
- Limits: this tests faithfulness to a deterministic ranker, not to an LLM agent's own choice, and not human understanding.
- Side finding: the query "a sturdy drill set" returned shears as the top product — retrieval relevance for that intent is weak and worth a look before the demo.

### 7.4 Comprehension study — kit only
`scripts/comprehension_study.py` generates real stimuli (two explanation conditions, answer key computed from the real attribution), and analyses responses (Welch t, Cohen's d, permutation p). Its self-test uses simulated data and proves only that the code runs. Recruit >= 20 participants per condition (between-subjects), collect the CSV described in the script header, and run `analyse`.
