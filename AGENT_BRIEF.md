# Agent brief - finishing Agentra for the final review

State: catalogue layer, human-baseline layer, purchase execution and hash-chained audit log are built and
tested (`python -m pytest -q`). The CDC module is IP-restricted: **do not build it**. Do not modify ASIDE
(`ASIDE_Refactor`), it is teammate-owned and only called over REST.

## Remaining tasks (in order)
1. Run `scripts/build_catalog.py` (see DATA.md) with real Amazon Reviews 2023 files; check `/system/data-status`.
2. OPeRA (huggingface.co/datasets/NEU-HAI/OPeRA) is a set of human browsing TRACES (clicks, page HTML, purchases
   with price/ASIN), not two-option choices, so the 2AFC "pp" metric cannot be computed from it directly. Derive
   comparable human statistics instead - e.g. position of the clicked/purchased result in the result list
   (position bias) and price percentile of the purchased item among results viewed - compute the same two
   numbers for the agent in `app/behaviour_experiment.py`, and store the human values in
   `data/references/human_literature.json` with source = "OPeRA". Until then the literature figures stay as the
   (unverified) reference.
3. Verify the Cherep et al. numbers in that JSON against the paper.
4. Add a dashboard panel for `/enterprise/behaviour-profile` (static/dashboard).
5. Optional LLM policy for `behaviour_experiment` (env API key; never hard-code keys).
6. Payments: set `PAYMENT_MODE=razorpay_test` with `rzp_test_` keys in env only. Never use live keys.
7. Keep `pytest` green (tests run with `CATALOG_SOURCE=mock`).
