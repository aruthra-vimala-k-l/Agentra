"""Shared test fixtures.

app.drift keeps process-global, in-memory state keyed by user_id (by
design -- see app/drift.py's own module docstring about the limitation).
That's fine for the running service but means tests across different
modules that reuse a user_id (e.g. "user-1") could otherwise see stale
history from an earlier test. Reset it before every test so each test's
drift assertions are self-contained.
"""
import os

# Unit tests run against the hand-authored mock catalogue (they assert on planted
# items such as the h-502 dark pattern). Must be set before app.catalog is imported.
os.environ.setdefault("CATALOG_SOURCE", "mock")

import pytest

import app.drift as drift
import app.payments as payments


@pytest.fixture(autouse=True)
def _reset_drift_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("PAYMENT_MODE", "dry_run")
    payments.reset()
    drift.reset()
    yield
    drift.reset()
