"""Tamper-evident, hash-chained purchase audit log (append-only JSONL).

Each entry stores prev_hash and hash = SHA-256(prev_hash || canonical JSON of
the entry body). Editing, deleting or reordering any past line breaks every
later hash, which ``verify()`` detects. This makes the log tamper-EVIDENT, not
tamper-proof: someone with file access can rewrite the whole chain, so the head
hash should be anchored externally for real deployments.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64
_lock = threading.Lock()


def log_path() -> Path:
    return Path(os.environ.get(
        "AUDIT_LOG_PATH", str(Path(__file__).resolve().parent.parent / "data" / "audit" / "purchases.jsonl")))


def _hash(prev: str, body: dict) -> str:
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((prev + canon).encode("utf-8")).hexdigest()


def _read() -> list[dict]:
    p = log_path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(event: str, payload: dict) -> dict:
    with _lock:
        entries = _read()
        prev = entries[-1]["hash"] if entries else GENESIS
        body = {"seq": len(entries) + 1, "ts": datetime.now(timezone.utc).isoformat(),
                "event": event, "payload": payload, "prev_hash": prev}
        entry = {**body, "hash": _hash(prev, body)}
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


def verify() -> dict:
    entries = _read()
    prev = GENESIS
    for i, e in enumerate(entries, start=1):
        body = {k: e[k] for k in ("seq", "ts", "event", "payload", "prev_hash")}
        if e["prev_hash"] != prev or e["seq"] != i or e["hash"] != _hash(prev, body):
            return {"valid": False, "entries": len(entries), "first_bad_seq": i}
        prev = e["hash"]
    return {"valid": True, "entries": len(entries), "head_hash": prev}
