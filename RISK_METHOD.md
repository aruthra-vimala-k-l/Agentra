# How Agentra's pre-payment risk score is built (methods note)

The score is an additive points system. Every number has a recorded source, in `data/references/risk_weights.json`
(`status`: `provisional` until the expert ratings are collected, then `ahp`). Four separate questions, four separate answers.

## 1. Which signals? (literature)
| Signal group | Basis |
|---|---|
| Fabricated / thin reviews | Ott, Choi, Cardie & Hancock (2011), *Finding Deceptive Opinion Spam by Any Stretch of the Imagination* (ACL). arXiv:1107.4557 |
| URL structure: raw-IP host, shortener, insecure link, risky TLD | Mohammad, Thabtah & McCluskey, phishing-website feature assessment (Univ. of Huddersfield) |
| Lookalike domains | Moore & Edelman (2010), *Measuring the Perpetrators and Funders of Typosquatting* |
| Seller and price signals for counterfeits | *Counterfeit Products on the Internet: The Role of Seller-Level and Product-Level Information* (IJEC) |
| Instructions aimed at an AI agent | Greshake et al. (2023), indirect prompt injection, arXiv:2302.12173; AgentDojo, arXiv:2406.13352 (NeurIPS 2024) |
| Points-based scores as an interpretable model class | Ustun & Rudin (2019), *Learning Optimized Risk Scores*, JMLR 20(150) |
(Verify each citation against the paper before quoting a detail from it.)

## 2. What is one point? (data)
Scorecard convention: points = (PDO / ln 2) * ln(LR), PDO = 20. So **20 points double the odds that a listing is unsafe**.
LR = P(feature | phishing) / P(feature | legitimate), Laplace-smoothed, from 10,000 labelled URLs
(`scripts/calibrate_url_weights.py`, results in `data/references/url_calibration.json`):
raw-IP host = 136 points (95% CI 56 to 216), shortened link = 14 points (CI 10 to 17).
Limit: legitimate URLs in that set are popular domains, which probably overstates the lifts.

## 3. How much does each signal matter relative to the others? (expert judgment, AHP)
Saaty's Analytic Hierarchy Process. Six families (page, address, seller, ratings/reviews, price, agent manipulation) and the
signals inside each are compared pairwise on the 1-9 scale by several raters (`scripts/ahp_rater_form.html`, 53 comparisons,
about 10 minutes). Raters are combined by the geometric mean; priorities are the principal eigenvector; each matrix must
have a consistency ratio below 0.10 (`scripts/ahp_aggregate.py`). The relative weights are converted to points with the
**data-fitted signals as the unit anchor**, and the script reports whether the experts' ratio of raw-IP to shortener
importance agrees with the data's ratio (136 / 14 = 9.7), a direct check of expert judgment against data.

## 4. Where is the High-risk line? (odds)
40 points = odds of an unsafe listing multiplied by 4 (2^(40/20)). "High risk" means the checks have made the listing at least
four times more likely to be unsafe than a listing with no warning signs. The threshold is stated in odds, not picked by feel.
`scripts/risk_weight_sensitivity.py`: +/-20% on every weight flips the High/Low call for about 4% of simulated listings
(firing rates assumed, so illustrative).

## What is and is not evidence-backed (state this plainly)
- Fitted from data: raw-IP host, shortened link.
- Expert-derived (AHP, N raters, consistency ratio reported): all other weights, once the ratings are in.
- Not independently validated: no public dataset has joint labels (real vs fake listing) with seller, rating and price signals.
  Validating the whole score against labelled marketplace listings is the stated next step.
- Detection thresholds inside a few rules (for example fewer than 30 ratings, price below 30% of the peer median) are still
  chosen values and should be replaced with statistical rules (Bayesian credible bound, MAD outlier test) in future work.
