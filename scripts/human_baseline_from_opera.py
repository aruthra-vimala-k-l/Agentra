"""Derive HUMAN reference statistics from OPeRA shopping traces (NEU-HAI/OPeRA, CC-BY-4.0, research use).

OPeRA is browsing traces, not 2AFC choices, so results are NOT the same metric as the Cherep et al.
percentage points. This script reports clearly-named, comparable *tilts* and writes them beside
(never over) the verified literature values.

  python scripts/human_baseline_from_opera.py --explore          # look at the data first
  python scripts/human_baseline_from_opera.py --derive           # compute tilts, write data/references/opera_human.json

Data: pip install datasets pandas pyarrow ; the script downloads  NEU-HAI/OPeRA  subset filtered_action
(or pass --parquet path/to/files*.parquet).

Derived stats (per session with >= 2 distinct priced products seen and a purchase/cart action):
  cheaper_than_median_rate : P(chosen product price < median price of products seen in that session)
  price_percentile_mean    : mean percentile (0 cheapest .. 1 priciest) of the chosen product among those seen
Interpretation: 0.5 / 0.5 would mean no price tilt. Heuristics (what counts as a choice, what counts as seen)
are printed so they can be defended or changed.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHOICE_RE = re.compile(r"buy now|place (your )?order|add to cart|proceed to checkout", re.I)


def load_frame(parquet: str | None):
    import pandas as pd
    if parquet:
        import glob
        files = sorted(glob.glob(parquet))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    from datasets import load_dataset
    ds = load_dataset("NEU-HAI/OPeRA", "filtered_action")
    return pd.concat([ds[k].to_pandas() for k in ds], ignore_index=True)


def parse_products(raw) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    out = []
    for p in raw if isinstance(raw, (list, tuple)) or hasattr(raw, "__iter__") else []:
        if isinstance(p, dict) and p.get("asin"):
            try:
                price = float(str(p.get("price", "")).replace("$", "").replace(",", ""))
            except ValueError:
                continue
            out.append({"asin": p["asin"], "price": price})
    return out


def element_name(raw) -> str:
    try:
        d = json.loads(raw) if isinstance(raw, str) else (raw or {})
        return str(d.get("name", "")) if isinstance(d, dict) else ""
    except json.JSONDecodeError:
        return ""


ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})")
CHOICE_IDS = ("add_to_cart", "buy_now", "place_your_order")
NOT_ALTERNATIVES = {"cart_items", "active_items", "saved_items", "order_items", "comparison_items"}   # what they already hold, not options they weighed


def _price(v):
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def seen_from_page_meta(raw, counts=None) -> dict[str, float]:
    """asin -> price for every product listed on the page the person was looking at (search results, details, promos...)."""
    try:
        d = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return {}
    out = {}
    if not isinstance(d, dict):
        return out
    for key, val in d.items():
        if key in NOT_ALTERNATIVES or not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, dict) and item.get("asin") and _price(item.get("price")) is not None:
                out[item["asin"]] = _price(item["price"])
                if counts is not None:
                    counts[key] = counts.get(key, 0) + 1
    return out


def derive(df) -> dict:
    """Per purchase choice (add-to-cart / buy-now click): price percentile of the chosen item among the products the person had
    SEEN BEFORE that click in the same session. 0 = cheapest seen, 1 = priciest. Needs >= 2 priced products seen."""
    cheaper, pct, sessions, skipped = [], [], set(), {"no_asin": 0, "price_unknown": 0, "too_few_seen": 0}
    counts: dict[str, int] = {}
    for sid, g in df.sort_values(["session_id", "timestamp"]).groupby("session_id"):
        seen: dict[str, float] = {}
        done = set()
        for _, row in g.iterrows():
            seen.update(seen_from_page_meta(row.get("page_meta"), counts))
            for p in parse_products(row.get("products")):
                seen.setdefault(p["asin"], p["price"])
            sem = str(row.get("semantic_id") or "").lower()
            if row.get("action_type") != "click" or not any(k in sem for k in CHOICE_IDS):
                continue
            m = ASIN_RE.search(str(row.get("url") or ""))
            if not m:
                skipped["no_asin"] += 1
                continue
            asin = m.group(1)
            if asin in done:
                continue
            if asin not in seen:
                skipped["price_unknown"] += 1
                continue
            prices = sorted(seen.values())
            if len(prices) < 2:
                skipped["too_few_seen"] += 1
                continue
            done.add(asin)
            price = seen[asin]
            cheaper.append(price < statistics.median(prices))
            pct.append(sum(x < price for x in prices) / (len(prices) - 1))
            sessions.add(sid)
    return {"sessions_used": len(sessions), "choices_used": len(cheaper), "skipped": skipped, "page_meta_lists_used": counts,
            "cheaper_than_median_rate": round(sum(cheaper) / len(cheaper), 3) if cheaper else None,
            "price_percentile_mean": round(sum(pct) / len(pct), 3) if pct else None}


def _lists(raw, only=None) -> dict[str, float]:
    try:
        d = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return {}
    out = {}
    for key, val in (d.items() if isinstance(d, dict) else []):
        if key in NOT_ALTERNATIVES or not isinstance(val, list) or (only and key != only):
            continue
        for it in val:
            if isinstance(it, dict) and it.get("asin") and _price(it.get("price")) is not None:
                out[it["asin"]] = _price(it["price"])
    return out


SET_SIZES: dict[str, list[int]] = {}


def choices_under(df, mode: str) -> list[tuple[str, bool, float]]:
    """(session, cheaper_than_median, percentile) per purchase click; mode = all | search_only | last_search_page."""
    res = []
    for sid, g in df.sort_values(["session_id", "timestamp"]).groupby("session_id"):
        seen, last, done = {}, {}, set()
        for _, row in g.iterrows():
            if mode == "all":
                seen.update(_lists(row.get("page_meta")))
            elif mode == "search_only":
                seen.update(_lists(row.get("page_meta"), "search_results"))
            else:
                cur = _lists(row.get("page_meta"), "search_results")
                if cur:
                    last = cur
            sem = str(row.get("semantic_id") or "").lower()
            if row.get("action_type") != "click" or not any(k in sem for k in CHOICE_IDS):
                continue
            m = ASIN_RE.search(str(row.get("url") or ""))
            pool = last if mode == "last_search_page" else seen
            if not m or m.group(1) in done or m.group(1) not in pool or len(pool) < 2:
                continue
            done.add(m.group(1))
            prices = sorted(pool.values())
            price = pool[m.group(1)]
            SET_SIZES.setdefault(mode, []).append(len(prices))
            res.append((sid, price < statistics.median(prices), sum(x < price for x in prices) / (len(prices) - 1)))
    return res


def export_sets(df) -> int:
    """Prices of every product a shopper had seen (last search page) + index of the one bought. No text, no ids."""
    sets = []
    for sid, g in df.sort_values(["session_id", "timestamp"]).groupby("session_id"):
        last, done = {}, set()
        for _, row in g.iterrows():
            cur = _lists(row.get("page_meta"), "search_results")
            if cur:
                last = cur
            sem = str(row.get("semantic_id") or "").lower()
            if row.get("action_type") != "click" or not any(k in sem for k in CHOICE_IDS):
                continue
            m = ASIN_RE.search(str(row.get("url") or ""))
            if not m or m.group(1) in done or m.group(1) not in last or len(last) < 2:
                continue
            done.add(m.group(1))
            asins = list(last)
            sets.append({"prices_usd": [last[a] for a in asins], "chosen": asins.index(m.group(1))})
    path = ROOT / "data" / "references" / "opera_choice_sets.json"
    path.write_text(json.dumps({"source": "NEU-HAI/OPeRA (CC-BY-4.0), prices only", "mode": "last_search_page", "sets": sets}), encoding="utf-8")
    print(len(sets), "choice sets ->", path)
    return 0


def sensitivity(df, boot: int = 2000) -> int:
    import random
    random.seed(0)
    out = {}
    for mode in ("all", "search_only", "last_search_page"):
        ch = choices_under(df, mode)
        if not ch:
            out[mode] = {"choices": 0}
            continue
        by_sess: dict[str, list] = {}
        for sid, c, pc in ch:
            by_sess.setdefault(sid, []).append((c, pc))
        sess = list(by_sess)
        reps_c, reps_p = [], []
        for _ in range(boot):                       # resample SESSIONS, not choices, because choices inside a session are not independent
            pick = [x for sid in random.choices(sess, k=len(sess)) for x in by_sess[sid]]
            reps_c.append(sum(c for c, _ in pick) / len(pick))
            reps_p.append(sum(p for _, p in pick) / len(pick))
        q = lambda v, f: sorted(v)[int(f * (len(v) - 1))]
        sz = sorted(SET_SIZES.get(mode, []))
        out[mode] = {"choices": len(ch), "sessions": len(sess),
                     "choice_set_size_median_q1_q3": [sz[len(sz) // 2], sz[len(sz) // 4], sz[3 * len(sz) // 4]] if sz else None,
                     "cheaper_than_median_rate": [round(sum(c for _, c, _ in ch) / len(ch), 3), round(q(reps_c, .025), 3), round(q(reps_c, .975), 3)],
                     "price_percentile_mean": [round(sum(p for _, _, p in ch) / len(ch), 3), round(q(reps_p, .025), 3), round(q(reps_p, .975), 3)]}
    print(json.dumps({"format": "[estimate, ci95_low, ci95_high]; 0.5 = no price tilt", **out}, indent=2))
    path = ROOT / "data" / "references" / "opera_human_sensitivity.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("->", path)
    return 0


def probe(df) -> int:
    import collections
    cut = lambda v, n=260: str(v)[:n].replace("\n", " ")
    nonempty = df["products"].map(lambda v: len(parse_products(v)) > 0)
    print("rows with parsable products:", int(nonempty.sum()), "of", len(df))
    print("click_type:", df["click_type"].value_counts(dropna=False).head(8).to_dict())
    print("semantic_id top:", df["semantic_id"].astype(str).value_counts().head(15).to_dict())
    print("\n--- 3 raw element_meta samples (clicks on product_options / cart_items / active_items) ---")
    for nm in ("product_options", "cart_items", "active_items"):
        sub = df[(df["action_type"] == "click") & (df["element_meta"].map(element_name) == nm)]
        if len(sub):
            print(nm, "->", cut(sub["element_meta"].iloc[0]))
    print("\n--- url kinds ---")
    kinds = collections.Counter()
    for u in df["url"].astype(str):
        kinds["cart" if "/cart" in u else "checkout" if ("checkout" in u or "/buy" in u or "/gp/buy" in u) else "product" if ("/dp/" in u or "/gp/product" in u) else "search" if ("/s?" in u or "/s/" in u) else "other"] += 1
    print(dict(kinds))
    print("\n--- first non-empty products sample ---")
    if nonempty.any():
        print(cut(df.loc[nonempty, "products"].iloc[0], 500))
    else:
        print("none; raw products dtype/example:", df["products"].dtype, "|", cut(df["products"].iloc[0]))
    print("\n--- page_meta sample (product page) ---")
    pm = df[df["url"].astype(str).str.contains("/dp/", na=False)]
    print(cut(pm["page_meta"].iloc[0], 600) if len(pm) else "no /dp/ rows")
    print("\n--- one full 'terminate' row and one cart_items click (selected fields) ---")
    for q in (df[df["action_type"] == "terminate"], df[(df["action_type"] == "click") & (df["element_meta"].map(element_name) == "cart_items")]):
        if len(q):
            r = q.iloc[0]
            print({k: cut(r[k], 200) for k in ("action_type", "semantic_id", "element_meta", "url", "rationale", "input_text")})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--sensitivity", action="store_true", help="recompute under 3 definitions of 'products seen' with bootstrap 95%% CIs")
    ap.add_argument("--export-sets", action="store_true", help="write the real choice sets (prices only) for the agent-side study")
    ap.add_argument("--probe", action="store_true", help="print raw samples to work out how purchases and products are stored")
    ap.add_argument("--parquet", help="glob of local parquet files (filtered_action)")
    a = ap.parse_args()
    df = load_frame(a.parquet)
    if a.probe:
        return probe(df)
    if a.sensitivity:
        return sensitivity(df)
    if a.export_sets:
        return export_sets(df)
    if a.explore or not a.derive:
        print("rows:", len(df), "| sessions:", df["session_id"].nunique())
        print("columns:", list(df.columns))
        print(df["action_type"].value_counts().to_string())
        names = df.loc[df["action_type"] == "click", "element_meta"].map(element_name).value_counts().head(25)
        print("most common clicked element names:\n", names.to_string())
        print("example products field:", df["products"].dropna().astype(str).iloc[0][:300] if df["products"].notna().any() else None)
        return 0
    res = derive(df)
    res.update({"dataset": "NEU-HAI/OPeRA filtered_action (CC-BY-4.0)",
                "metric_note": "Browsing-trace tilt, NOT the Cherep 2AFC pp metric. 0.5 = no price tilt.",
                "choice_heuristic": "click whose semantic_id contains add_to_cart / buy_now / place_your_order; chosen ASIN from the page URL; products seen = priced lists in page_meta (not cart/saved items) before the click"})
    out = ROOT / "data" / "references" / "opera_human.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2), "\n->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
