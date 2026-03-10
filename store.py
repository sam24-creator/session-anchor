"""
store.py — Session memory store for SessionAnchor.

Persists session metadata and conversation history to disk (JSON).
Designed to be swappable — replace with Redis, SQLite, or a vector DB
for production deployments.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class SessionStore:
    """
    File-backed store for session envelopes and message history.

    Directory layout:
        <store_dir>/
            <session_id>/
                envelope.json    — session metadata and temporal snapshots
                messages.jsonl   — newline-delimited message log

    Swap this class out for a RedisSessionStore or SQLiteSessionStore
    in production — the interface is intentionally minimal.
    """

    def __init__(self, store_dir: str = ".session_anchor"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ── Session lifecycle ────────────────────────────────────────────────────

    def create_session(
        self,
        user_id: Optional[str] = None,
        timezone_name: str = "UTC",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Create a new session and return its ID.
        """
        session_id = str(uuid.uuid4())
        session_dir = self.store_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(tz=timezone.utc).isoformat()
        envelope = {
            "session_id": session_id,
            "user_id": user_id,
            "timezone_name": timezone_name,
            "created_at": now,
            "last_active_at": now,
            "ended_at": None,
            "metadata": metadata or {},
        }

        (session_dir / "envelope.json").write_text(
            json.dumps(envelope, indent=2), encoding="utf-8"
        )
        return session_id

    def end_session(self, session_id: str) -> None:
        """
        Mark a session as ended, recording the exact end timestamp.
        This is the timestamp used for gap calculation on resume.
        """
        envelope = self._load_envelope(session_id)
        envelope["ended_at"] = datetime.now(tz=timezone.utc).isoformat()
        self._save_envelope(session_id, envelope)

    def get_last_ended_session(self, user_id: str) -> Optional[dict]:
        """
        Find the most recently ended session for a given user.
        Returns the envelope dict or None.
        """
        candidates = []
        for session_dir in self.store_dir.iterdir():
            env_path = session_dir / "envelope.json"
            if not env_path.exists():
                continue
            env = json.loads(env_path.read_text(encoding="utf-8"))
            if env.get("user_id") == user_id and env.get("ended_at"):
                candidates.append(env)

        if not candidates:
            return None

        return max(candidates, key=lambda e: e["ended_at"])

    # ── Message logging ──────────────────────────────────────────────────────

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Append a message to the session log with a server-side timestamp.
        """
        session_dir = self.store_dir / session_id
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **(metadata or {}),
        }
        with open(session_dir / "messages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(msg) + "\n")

        # Keep last_active_at current
        envelope = self._load_envelope(session_id)
        envelope["last_active_at"] = msg["timestamp"]
        self._save_envelope(session_id, envelope)

    def get_messages(self, session_id: str) -> list[dict]:
        """
        Return all messages for a session in chronological order.
        """
        path = self.store_dir / session_id / "messages.jsonl"
        if not path.exists():
            return []
        messages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                messages.append(json.loads(line))
        return messages

    def get_envelope(self, session_id: str) -> Optional[dict]:
        env_path = self.store_dir / session_id / "envelope.json"
        if not env_path.exists():
            return None
        return json.loads(env_path.read_text(encoding="utf-8"))

    # ── Internals ────────────────────────────────────────────────────────────

    def _load_envelope(self, session_id: str) -> dict:
        path = self.store_dir / session_id / "envelope.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_envelope(self, session_id: str, envelope: dict) -> None:
        path = self.store_dir / session_id / "envelope.json"
        path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
