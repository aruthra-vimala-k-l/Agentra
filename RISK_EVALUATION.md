# Measured accuracy of Agentra's rule-based risk detectors

Reproduce: `py scripts/evaluate_risk_detectors.py all` (results: `data/references/risk_detector_eval.json`). Every figure below was
measured by that script; 95% intervals are Wilson intervals. These are triage rules, not classifiers; the gate and human approval are
the safety net. This page reports where the rules are weak.

## 1. URL signals vs real phishing (4,000 verified PhishTank URLs vs 4,000 benign-list URLs)
| | Flagged High (>= 40 points) | Any URL signal fires | AUC |
|---|---|---|---|
| All URL signals | TPR 2.9% (2.4-3.4), FPR 0.3% (0.2-0.6) | TPR 58%, FPR 95% | 0.32 |
| Without "insecure link" | TPR 1.7% (1.3-2.1), FPR 0.3% (0.2-0.6) | TPR 5.3%, FPR 0.3% | 0.52 |
Reading: as a standalone phishing-URL detector the rule set is close to chance. It is designed for shopping links as one input among
page, seller, rating and price checks, not for generic phishing. Precision at an assumed 1% prevalence is about 8%.
"Insecure link" fires on 95% of the benign list because that list is old (5.5% https); phishing today is often https (45.7% here), so
http-versus-https is not a usable discriminator in this data.
Per signal (phishing / benign): raw-IP host 1.6% / 0%; risky TLD 2.4% / 0%; shortener 1.2% / 0%; lookalike domain 0.08% / 0.32%.
Limit: benign URLs are not shopping URLs; only URL-structure signals can be tested this way.

## 2. Review-authenticity check
Real catalogue products (1,029, Amazon reviews): **27 flagged (2.6%; 1.8-3.8%)**, all by the exact-duplicate-text rule (short generic
reviews repeat). Burst and low-diversity rules: 0 flagged. False-positive proxy only: reviews are assumed mostly organic, not labelled.
Injected fake reviews (synthetic by construction, 200 real products, k = 5, 10, 20 fakes):
| Attack | Fakes alone | Fakes added to the product's real reviews |
|---|---|---|
| Burst (2 days) | 100% | 4% (= background rate: not detected) |
| Sockpuppets (2 accounts, 90 days) | 100% | 4% at k=5,10; 41% at k=20 |
| Identical template text | 100% | 100% |
| Stealth (distinct accounts and text, 90 days) | 0% | 4% (not detected) |
Reading: the rules catch crude, standalone fraud and exact copies. They miss fakes mixed into real review histories and anything
non-identical, because the burst rule needs ALL reviews inside 5 days and duplicate detection is exact-match only.

## 3. Prompt-injection detector (the payment gate and the risk score both use `app/injection_guard.py`)
Three versions measured on the same data (Wilson 95% CIs in the JSON):
| Detector | AgentDojo templates x 6 goals (30) | InjecAgent plain instructions (62) | InjecAgent with override wrapper (62) | False positives on 24,035 real texts |
|---|---|---|---|---|
| Old: 4 literal phrases | 33% (10/30) | 0% | 100% | 1 |
| v2: override, role, sequence, action families | 100% (30/30) | 0% | 100% | 1 |
| v3: v2 + exfiltration/account-action rules | 100% (30/30) | 55% (34/62; 43-67%) | 100% | 1 |
Honest reading:
- The AgentDojo number is a **development-set** result: those templates were seen while designing v2, so 100% is optimistic.
- InjecAgent is independent for v2. v2 catches attacks that use override or role wording (wrapper set: 100%) and **none** of the plain instructions
  ("email my medical records to ..."), which read like ordinary requests.
- v3's exfiltration family was written after seeing 2 of the 62 rows, so its 55% is optimistic too.
- False alarms stayed at 1 in 24,035 real descriptions and reviews.
- No regex reliably separates a plain instruction from a normal request. That needs a model or a provenance rule (never act on text from
  a listing); the deterministic gate and human approval remain the real protection.

## What this means
- Precision is high (very few false alarms on real text), recall is low. The system errs towards silence on subtle attacks.
- Defensible statement: "rule-based, transparent triage with measured, limited recall; the safety guarantee comes from the
  deterministic gate and human approval, not from detection."
- Fixes with the clearest payoff: (a) learn lexical URL features (@, hyphens, subdomain count, keywords) as a scorecard and validate on
  a held-out split; (b) windowed burst and rating-distribution tests plus near-duplicate matching for reviews; (c) a broader injection
  detector validated on templates it was not tuned on.
