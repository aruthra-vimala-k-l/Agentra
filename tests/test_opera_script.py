import importlib.util
import json
from pathlib import Path

import pytest

pd = pytest.importorskip('pandas')

spec = importlib.util.spec_from_file_location("opera", Path(__file__).resolve().parent.parent / "scripts" / "human_baseline_from_opera.py")
opera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opera)


def test_derive_logic_on_tiny_frame():
    A, B, C = "A" * 10, "B" * 10, "C" * 10
    pm = json.dumps({"search_results": [{"asin": A, "price": "$10"}, {"asin": B, "price": "$30"}, {"asin": C, "price": "$20"}]})
    rows = [
        {"session_id": "s1", "timestamp": 1, "action_type": "click", "semantic_id": "x", "url": "u", "page_meta": pm, "products": "[]"},
        {"session_id": "s1", "timestamp": 2, "action_type": "click", "semantic_id": "buybox.purchase_form.add_to_cart",
         "url": "https://www.amazon.com/dp/" + A, "page_meta": "{}", "products": "[]"},
    ]
    r = opera.derive(pd.DataFrame(rows))
    assert r["choices_used"] == 1 and r["cheaper_than_median_rate"] == 1.0 and r["price_percentile_mean"] == 0.0
