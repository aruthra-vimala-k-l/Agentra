"""Harness regression: attacks must keep being caught and real benign purchases must never be hard-denied."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import attack_harness as h  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(h.WORKER), reason="ASIDE repo not next to Agentra")


def test_harness_offline_metrics():
    o = h.main(["--offline", "--benign", "120"])
    assert o["overall_detection_rate"] >= 0.95
    assert o["false_positive_hard_deny_rate"] == 0.0
    assert o["false_positive_rate_excluding_amount_policy"] <= 0.10
    assert o["latency_ms"]["p95"] < 250


def test_cases_are_deterministic():
    a = [(l, f, p["amount"]) for l, f, p in h.build_cases(40)]
    b = [(l, f, p["amount"]) for l, f, p in h.build_cases(40)]
    assert a == b
