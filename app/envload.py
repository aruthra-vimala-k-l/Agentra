"""Minimal .env loader (no dependency). Real environment variables win; values are never printed.

Looks for ``.env`` in the project root. Keep that file out of git (it is in .gitignore).
"""
from __future__ import annotations

import os
from pathlib import Path


def load(path: Path | None = None) -> list[str]:
    p = path or Path(__file__).resolve().parent.parent / ".env"
    loaded: list[str] = []
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded  # names only
