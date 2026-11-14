# Price-tilt study: Agentra vs. human shoppers

Reproduce (PowerShell, from `agentra_shopping_agent`):
`py scripts/human_baseline_from_opera.py --derive --sensitivity --export-sets` (human side, needs Hugging Face access),
`py -m app.price_tilt_study 200 3` (agent side), `py -m app.price_tilt_validation` (simulated validation).
Outputs: `data/references/{opera_human_sensitivity,price_tilt_study,price_tilt_validation}.json`.

## Question
Does Agentra systematically favour cheaper or dearer items than a human would, and if so, why?

## Metric
Price percentile of the chosen item within the set it was chosen from: 0 = cheapest, 0.5 = no tilt, 1 = dearest. 95%
bootstrap CIs. This is a browsing-trace tilt, **not** the Cherep et al. 2AFC percentage-point metric. Ties count as "not
cheaper", which biases a price-blind chooser about 0.02 below 0.5 (uniform random measures 0.47-0.50).

## Human reference (OPeRA, 123 purchases, one per session)
| "Products seen" defined as | Price percentile | Below-median share |
|---|---|---|
| all priced lists before the click | 0.359 (0.316-0.404) | 0.675 (0.593-0.756) |
| search results only | 0.375 (0.323-0.427) | 0.635 (0.539-0.722) |
| last search page only | 0.392 (0.339-0.447) | 0.574 (0.481-0.667) |

Robust: humans buy near the 36th-39th price percentile. The below-median share is not robust (its strictest CI includes 0.5).

## Agent results (k = 5 same-category catalogue products, 3 seeds x 200 sets)
| Chooser | No budget | Loose budget |
|---|---|---|
| Original ranker | 0.82 | 0.88 |
| after fix 1 only (brand-tier guard) | 0.54 | 0.88 |
| **final ranker (both fixes)** | **0.54 (0.51-0.57)** | **0.54 (0.51-0.57)** |
| live assistant ranking stage | 0.51 (0.48-0.54) | 0.51 (0.48-0.54) |
| controls: always-first / uniform random / cheapest / dearest | 0.50 / 0.47 / 0.00 / 1.00 | |

Median and tight budgets are omitted from the headline: they filter the set to cheaper items, so the percentile is capped
mechanically (0.24 and 0.11) and says nothing about preference.

### Defect 1: brand tier was a price rank
In the real catalogue `brand_tier` = "top price quartile within category" (`app/catalog_build.py`): premium share is 25% in
every category, Spearman with price 0.67-0.75. The ranker's brand bonus therefore rewarded price. Ablation: removing it moved
the no-budget percentile from 0.82 to 0.54; no other signal mattered. Fix: `retrieval.GUARD_PRICE_DERIVED_TIER`.

### Defect 2: budget-fit rewarded spending the budget
`price_fit` paid +1.5 to items using at least half the budget and +0.3 otherwise. Under a loose budget this alone moved the
percentile from 0.54 to 0.88. Fix: `retrieval.PRICE_FIT_MODE = "flat"` - every affordable item fits equally; over-budget is
still penalised. The old shape remains behind the switch so the before/after is reproducible.

### Residual
The final ranker sits 0.04 above 0.5 (about 0.02 after the tie bias). Removing `review_influence` brings it to 0.51: ratings
carry a small price correlation in this catalogue (Spearman -0.08 to 0.07), so it is not a price rule.

### Live assistant
Its ranking stage (`web_assistant.evaluate`: rating + budget fit + grounding - risk penalties) measures 0.51, price-neutral
with and without a budget, on catalogue products presented as one-seller listings. The model and web steps are not included.

## Simulated validation (synthetic; method check only, never evidence about shoppers)
* **Planted-tilt recovery:** synthetic choosers with a known price preference are measured at 0.81, 0.66, 0.49, 0.31, 0.19,
  0.12 for planted strengths -3, -1.5, 0, 1.5, 3, 4.5: monotone, with the neutral case at 0.49. The metric recovers what is planted.
* **Set-size invariance:** a uniform-random chooser measures 0.47-0.505 at k = 3, 5, 8, 15.
* **Power:** for a chooser tuned to the human-like 0.38, a 95% CI excludes 0.5 with probability 0.36 (n = 30), 0.59 (60),
  **0.89 (123, the human sample size)**, 0.99 (250), 1.0 (500). The human result is adequately powered but not overwhelming.
* **Real human sets, synthetic ratings:** the rankers were run on the actual OPeRA price lists (108 usable sets) with synthetic
  ratings correlated with price. Price percentile of the choice: final ranker **0.478**, live assistant ranking **0.471**,
  uniform random 0.461, humans on the same sets **0.392**. Sensitivity to the invented rating-price correlation: rho = -0.4
  gives 0.23 and rho = +0.4 gives 0.69, so the agent-side figure depends strongly on that assumption and is NOT evidence
  about real shoppers. Ties count as "not cheaper", so a price-blind chooser sits about 0.02-0.04 below 0.5 (random = 0.461).
  Reading: on identical sets the agents are close to price-neutral (0.47-0.48) while humans tilt cheaper (0.39).

## Limits (not fixable from this codebase)
* The real-sets run above now puts agents and humans on identical price lists; ratings on those sets are still synthetic.
* OPeRA is Amazon US browsing in USD; the catalogue is Indian-market data. Percentiles are scale-free, behaviour may differ.
* One catalogue, four categories. The live assistant was measured on its ranking stage only, not end to end with the model.
