"""Behavioral drift tracking -- a narrow, honest slice of Patent Claim 3
("behavioral drift detection over time").

This is intentionally simple: an in-memory, process-local history of each
user's Shapley signal-decomposition snapshots (see app/attribution.py),
compared turn-over-turn. It genuinely detects a shift in which signals are
driving a user's (or a compromised agent acting as that user's) purchase
decisions -- e.g. a sudden jump in weight on `nudge_exposure` or
`new_merchant_caution` across a session is exactly the kind of drift the
proposal describes wanting to catch.

What this is *not*: a production drift detector. It has no persistence
across restarts, no statistical significance testing, and a fixed threshold
rather than a learned baseline. Documented as a starting point, not
oversold as the full mechanism -- see SECURITY_NOTES.md.
"""
from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_history: dict[str, list[dict[str, float]]] = defaultdict(list)

DRIFT_THRESHOLD = 1.0  # absolute shapley-value delta considered meaningful


def record_and_check(user_id: str, shapley_values: dict[str, float]) -> dict:
    with _lock:
        history = _history[user_id]
        history.append(dict(shapley_values))
        if len(history) < 3:
            return {"drift_detected": False, "samples": len(history),
                    "note": "Not enough history yet to assess drift for this user."}

        baseline = history[:-1]
        keys = set()
        for h in history:
            keys.update(h.keys())

        deltas: dict[str, float] = {}
        for k in keys:
            baseline_vals = [h.get(k, 0.0) for h in baseline]
            baseline_avg = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0
            latest = history[-1].get(k, 0.0)
            deltas[k] = round(latest - baseline_avg, 3)

        drifted_signals = [k for k, d in deltas.items() if abs(d) > DRIFT_THRESHOLD]
        return {
            "drift_detected": bool(drifted_signals),
            "drifted_signals": drifted_signals,
            "deltas": deltas,
            "samples": len(history),
        }


def reset() -> None:
    """Test-only: clear all recorded history."""
    with _lock:
        _history.clear()
