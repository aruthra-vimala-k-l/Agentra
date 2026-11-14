"""Process-local session store -- fine for a demo, not for production
(matches the same "prototype, not production" honesty ASIDE's own
app/main.py states about its process-local startup state).

Vulnerability fix (partial -- see SECURITY_NOTES.md): the original version
had no bound on how many sessions could accumulate in memory, which is an
unauthenticated-DoS-by-memory-exhaustion vector for any endpoint that
creates a session per request. This caps the store and evicts the oldest
session once the cap is hit. It does not add persistence, sharding, or
cross-process consistency -- those are out of scope for a capstone demo and
are named explicitly, not silently left out.
"""
from __future__ import annotations

import threading
import time
import uuid

from app.orchestrator import SessionState

_lock = threading.Lock()
_sessions: dict[str, SessionState] = {}
_created_at: dict[str, float] = {}

MAX_SESSIONS = 500


def create_session(agent_id: str = "shopping-agent-01", user_id: str = "user-demo") -> SessionState:
    session_id = str(uuid.uuid4())
    state = SessionState(session_id=session_id, agent_id=agent_id, user_id=user_id)
    with _lock:
        if len(_sessions) >= MAX_SESSIONS:
            oldest_id = min(_created_at, key=_created_at.get)
            _sessions.pop(oldest_id, None)
            _created_at.pop(oldest_id, None)
        _sessions[session_id] = state
        _created_at[session_id] = time.time()
    return state


def get_session(session_id: str) -> SessionState | None:
    with _lock:
        return _sessions.get(session_id)


def all_sessions() -> list[SessionState]:
    """Read-only snapshot for the Enterprise Dashboard (app/dashboard_service.py).
    Returns a copy so callers can't mutate the live store."""
    with _lock:
        return list(_sessions.values())
