#!/usr/bin/env python3
"""Build data/catalog/products.jsonl from REAL Amazon Reviews 2023 data.

Easiest: point --dir at the folder holding the downloaded files
(meta_<Category>[_sample].jsonl.gz and <Category>[_sample].jsonl.gz):

    python scripts/build_catalog.py --dir ../Amazon_Reviews_7 --per-category 150

Explicit files also work:  --file All_Beauty=meta_All_Beauty.jsonl.gz --reviews All_Beauty=All_Beauty.jsonl.gz

Output: data/catalog/products.jsonl + MANIFEST.json (provenance, rate, seed, counts).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.catalog_build import (CATEGORY_MAP, build, collect_reviews,  # noqa: E402
                               count_reviews_per_asin)

META_SCAN_LIMIT = int(os.environ.get("META_SCAN_LIMIT", 400_000))   # raw meta lines read per category
REVIEW_SCAN_LIMIT = int(os.environ.get("REVIEW_SCAN_LIMIT", 4_000_000))
MIN_REVIEWS = 10


def iter_file(path, limit: int):
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        try:
            for i, line in enumerate(fh):
                if i >= limit:
                    return
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # e.g. truncated last line of a partial download
        except (EOFError, OSError):
            return  # truncated/corrupt gzip: keep what was read


def discover(folder: Path) -> tuple[dict, dict]:
    metas, revs = {}, {}
    for cat in CATEGORY_MAP:
        m = sorted(x for x in folder.glob(f"meta_{cat}*.jsonl*") if not x.name.endswith(".part"))
        r = sorted(p for p in folder.glob(f"{cat}*.jsonl*") if not p.name.startswith("meta_") and not p.name.endswith(".part"))
        if m:
            metas[cat] = m[0]
        if r:
            revs[cat] = r[0]
    return metas, revs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="folder with downloaded meta_*/review files")
    ap.add_argument("--file", action="append", default=[], help="Category=path to a META file")
    ap.add_argument("--reviews", action="append", default=[], help="Category=path to a REVIEW file")
    ap.add_argument("--per-category", type=int, default=150, help="products kept per source file")
    ap.add_argument("--usd-inr", type=float, default=83.0, help="ASSUMED conversion rate, recorded in the manifest")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "catalog" / "products.jsonl"))
    args = ap.parse_args()

    metas = dict(f.split("=", 1) for f in args.file)
    revs = dict(f.split("=", 1) for f in args.reviews)
    if args.dir:
        dm, dr = discover(Path(args.dir))
        metas = {**dm, **metas}
        revs = {**dr, **revs}
    if not metas:
        ap.error("no metadata files found: give --dir or --file Category=path (run download_amazon_meta_7.py first)")

    import pickle
    import random as _random
    cache_dir = Path(os.environ.get("BUILD_CACHE_DIR", str(ROOT / "data" / "catalog" / ".cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(name, fn):
        """Each slow pass is cached, so an interrupted build resumes where it stopped."""
        f = cache_dir / f"{name}.pkl"
        if f.exists():
            try:
                return pickle.loads(f.read_bytes())
            except Exception:  # truncated by an interrupted run: recompute
                pass
        v = fn()
        f.write_bytes(pickle.dumps(v))  # a truncated file is detected and recomputed on load
        return v

    reviews_by_asin: dict[str, list] = {}
    meta_rows: dict[str, list] = {}
    pool = args.per_category * 4
    for cat in metas:
        if cat not in revs:
            continue
        counts = cached(f"{cat}.counts", lambda: count_reviews_per_asin(iter_file(revs[cat], REVIEW_SCAN_LIMIT)))
        ok = {a for a, c in counts.items() if c >= MIN_REVIEWS}
        meta_ids = cached(f"{cat}.metaids", lambda: {r.get("parent_asin") for r in iter_file(metas[cat], META_SCAN_LIMIT)})
        both = ok & meta_ids
        wanted = set(_random.Random(args.seed).sample(sorted(both), pool)) if len(both) > pool else both
        reviews_by_asin.update(cached(f"{cat}.reviews", lambda: collect_reviews(iter_file(revs[cat], REVIEW_SCAN_LIMIT), wanted)))
        meta_rows[cat] = cached(f"{cat}.metarows", lambda: [r for r in iter_file(metas[cat], META_SCAN_LIMIT) if r.get("parent_asin") in wanted])
        print(f"{cat}: {len(ok)} products have >= {MIN_REVIEWS} reviews in the review file; {len(both)} also in metadata; {len(wanted)} kept", flush=True)

    sources = {cat: iter(meta_rows[cat]) for cat in meta_rows}
    for cat, path in metas.items():
        if cat not in sources:
            sources[cat] = iter_file(path, META_SCAN_LIMIT)
    rows = build(sources, args.usd_inr, args.per_category, reviews_by_asin or None, seed=args.seed,
                 min_reviews=MIN_REVIEWS)
    if not rows:
        print("No usable records -- check the files.", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    manifest = {
        "dataset": "Amazon Reviews 2023 (McAuley Lab) -- item metadata + reviews",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_files": {c: str(Path(p).name) for c, p in metas.items()},
        "review_files": {c: str(Path(p).name) for c, p in revs.items()},
        "usd_inr_assumed": args.usd_inr,
        "selection": f"seeded random sample (seed={args.seed}); products with >= {MIN_REVIEWS} reviews first",
        "record_count": len(rows),
        "by_category": by_cat,
        "reviews_attached": sum(1 for r in rows if r.get("reviews")),
        "derived_fields": {
            "durability": "keyword rule over title/features/description; None when unknown",
            "occasion_tags": "keyword rule (app.intent.OCCASION_KEYWORDS) over text",
            "brand_tier": "premium = top price quartile within source file",
        },
        "not_available_in_source": ["delivery_days", "sponsored flag", "seller age"],
    }
    out.with_name("MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} real products ({manifest['reviews_attached']} with reviews) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
