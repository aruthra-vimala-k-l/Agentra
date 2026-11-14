"""Real-data catalogue loader.

Reads ``data/catalog/products.jsonl`` (built by ``scripts/build_catalog.py``
from real public datasets -- Amazon Reviews 2023 metadata, Open Food Facts /
Open Prices). One JSON object per line, already normalised to the ``Product``
field names. A sibling ``MANIFEST.json`` records provenance (source, licence,
fetch time, row counts).

There is deliberately NO silent fallback to the mock catalogue: if the file is
missing the real catalogue is simply empty and ``status()`` says why. That is
what lets us tell an evaluator "every product the agent can buy is a real
record" without an asterisk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.product import Product

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = _ROOT / "data" / "catalog" / "products.jsonl"

_FIELDS = set(Product.__dataclass_fields__)


def catalog_path() -> Path:
    return Path(os.environ.get("CATALOG_PATH", str(DEFAULT_PATH)))


def load() -> list[Product]:
    path = catalog_path()
    if not path.exists():
        return []
    out: list[Product] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(Product(**{k: v for k, v in row.items() if k in _FIELDS}))
    return out


def manifest() -> dict | None:
    mpath = catalog_path().with_name("MANIFEST.json")
    if not mpath.exists():
        return None
    return json.loads(mpath.read_text(encoding="utf-8"))
