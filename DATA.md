# Data provenance (read this before the review)

| Component | Data | Real? |
|---|---|---|
| Product catalogue | Amazon Reviews 2023 item metadata (McAuley Lab), built by `scripts/build_catalog.py` into `data/catalog/products.jsonl` | Real records. `MANIFEST.json` lists derived fields |
| Human baseline | `data/references/human_literature.json` (Cherep et al. figures) | Published numbers - verify against the paper; swap for OPeRA-derived values |
| Agent behaviour profile | 2AFC experiment on the real catalogue (`/enterprise/behaviour-profile`) | Agent choices are measured live; products are real |
| Unit tests | `app/mock_catalog.py` (`CATALOG_SOURCE=mock`) | Synthetic - tests only, never used when `CATALOG_SOURCE=real` |

`GET /system/data-status` reports which source is live and flags `synthetic: true` for mock.
There is no silent fallback: an empty real catalogue returns a warning, not mock products.

## Build the real catalogue (run on your laptop, from this folder)
`AmazonReviews2023-main` on GitHub is only code. The data comes from mcauleylab.ucsd.edu (links on
amazon-reviews-2023.github.io). Two kinds of file per category:
* REVIEW files (`<Category>.jsonl.gz`) - downloaded by `../download_amazon_7.py`
* METADATA files (`meta_<Category>.jsonl.gz`: title, price, rating, store) - downloaded by `../download_amazon_meta_7.py`
```
META_SCAN_LIMIT=200000 REVIEW_SCAN_LIMIT=1000000 python scripts/build_catalog.py --dir ../Amazon_Reviews_7 --per-category 150
# Windows cmd: set META_SCAN_LIMIT=200000 & set REVIEW_SCAN_LIMIT=1000000 first. Slow passes are cached in data/catalog/.cache,
# so an interrupted run resumes; delete that folder to force a fresh build.
```
Per source file it keeps `--per-category` products (seeded random sample, products with >= 10 real
reviews first), attaches the real reviews, writes `data/catalog/products.jsonl` + `MANIFEST.json`.
The 2023 dataset repeats a review once per colour/size variant; those repeats are collapsed so the
Review Authenticity check does not mistake them for fraud.

MartVerse (`bias_testbed_storefront.html`) is a synthetic bias-testbed fixture. It is excluded in real
mode; set `CATALOG_INCLUDE_TESTBED=1` to inject it as a labelled attack/bias test (`source=martverse_testbed`,
and `/system/data-status` then reports `synthetic: true`).

## Known limits (say these before an evaluator does)
- Prices are USD converted at an **assumed** rate (`--usd-inr`, recorded in the manifest).
- `durability`, `occasion_tags`, `brand_tier` are keyword/quartile rules over real data, not source fields.
- `delivery_days`, sponsored flag, seller age are not in the source: left unknown/False.
- The sensitivity metric is an approximation of Cherep et al.'s LPM metric.
- Razorpay test mode creates an Order (status `created`); it does not move money.


## Sampling note (Oct 2026 build)
The five large category files (Electronics, Clothing, Grocery, Arts & Crafts, Tools) are read only up to the first
200,000 metadata lines and 1,000,000 review lines each, then up to 600 candidate products per file are drawn at random
(seed 7) from products with >= 10 reviews present in both files. Handmade and All_Beauty are read in full. The catalogue is
therefore a sample, not the full Amazon database.
