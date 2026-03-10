"""
anchor.py — The primary public API for SessionAnchor.

Usage:
    from session_anchor import SessionAnchor

    anchor = SessionAnchor(user_id="user_123", timezone="America/New_York")

    # Start or resume a session
    session_id = anchor.start()

    # Build a grounded prompt to send to your LLM
    messages = anchor.build_context(user_message="Continue where we left off")

    # After the LLM responds, log both turns
    anchor.log("user", "Continue where we left off")
    anchor.log("assistant", llm_response)

    # When done, end the session
    anchor.end()
"""

from __future__ import annotations

from typing import Optional

from session_anchor.core.temporal import create_temporal_snapshot, TemporalSnapshot
from session_anchor.decay.decay import score_conversation, ScoredMessage, DecayClass
from session_anchor.memory.store import SessionStore


class SessionAnchor:
    """
    Grounds LLM sessions in real time by injecting temporal context
    and flagging stale information from prior sessions.

    Core problems solved:
    - LLM assumes old session time is still "now"
    - Time-specific context (schedules, deadlines) leaks across sessions
    - No awareness of how much time has elapsed between sessions
    """

    def __init__(
        self,
        user_id: str,
        timezone: str = "UTC",
        store_dir: str = ".session_anchor",
        include_stale_messages: bool = False,
        max_prior_messages: int = 20,
    ):
        """
        Args:
            user_id:                 Stable identifier for the user across sessions.
            timezone:                IANA timezone name (e.g. "Europe/London").
            store_dir:               Where to persist session data.
            include_stale_messages:  If True, stale messages are included with warnings.
                                     If False (default), stale messages are omitted.
            max_prior_messages:      Max messages to carry forward from prior session.
        """
        self.user_id = user_id
        self.timezone = timezone
        self.include_stale_messages = include_stale_messages
        self.max_prior_messages = max_prior_messages

        self._store = SessionStore(store_dir=store_dir)
        self._session_id: Optional[str] = None
        self._snapshot: Optional[TemporalSnapshot] = None

    # ── Session lifecycle ────────────────────────────────────────────────────

    def start(self, metadata: Optional[dict] = None) -> str:
        """
        Start a new session, automatically detecting any prior session gap.

        Returns:
            The new session_id.
        """
        # Find the last ended session for this user
        prior = self._store.get_last_ended_session(self.user_id)
        prior_ended_at = prior["ended_at"] if prior else None

        # Create session in store
        self._session_id = self._store.create_session(
            user_id=self.user_id,
            timezone_name=self.timezone,
            metadata=metadata or {},
        )

        # Capture temporal snapshot
        self._snapshot = create_temporal_snapshot(
            session_id=self._session_id,
            tz_name=self.timezone,
            previous_session_ended_at=prior_ended_at,
        )

        return self._session_id

    def end(self) -> None:
        """
        End the current session. Call this when the user closes the chat,
        logs out, or the session times out.
        """
        if self._session_id:
            self._store.end_session(self._session_id)
            self._session_id = None
            self._snapshot = None

    # ── Context building ─────────────────────────────────────────────────────

    def build_context(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        prior_session_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Build a grounded message list ready to send to any LLM API.

        This injects:
        1. A temporal grounding block into the system prompt
        2. Scored and filtered messages from the prior session
        3. The current user message

        Args:
            user_message:       The user's current input.
            system_prompt:      Your existing system prompt (will be augmented).
            prior_session_id:   Explicit session ID to pull history from.
                                If None, uses the most recent ended session.

        Returns:
            List of message dicts compatible with OpenAI/Anthropic APIs.
        """
        assert self._session_id, "Call start() before build_context()"
        assert self._snapshot, "Temporal snapshot missing — call start() first"

        messages: list[dict] = []

        # 1. Build grounded system prompt
        grounded_system = self._build_system_prompt(system_prompt)
        messages.append({"role": "system", "content": grounded_system})

        # 2. Inject prior session messages (scored for staleness)
        prior_messages = self._get_prior_messages(prior_session_id)
        if prior_messages:
            messages.extend(prior_messages)

        # 3. Append current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def log(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        """
        Log a message to the current session store.
        Always log both user and assistant turns for future session resumption.
        """
        assert self._session_id, "Call start() before log()"
        self._store.append_message(self._session_id, role, content, metadata)

    # ── Inspection ───────────────────────────────────────────────────────────

    @property
    def snapshot(self) -> Optional[TemporalSnapshot]:
        """The temporal snapshot for the current session."""
        return self._snapshot

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def get_gap_summary(self) -> Optional[str]:
        """
        Returns a human-readable summary of the time gap since the last session.
        Useful for UI display or debugging.
        """
        if not self._snapshot:
            return None
        s = self._snapshot
        if s.gap_label:
            return (
                f"Resumed {s.gap_label} after the previous session "
                f"(ended {s.previous_session_ended_at}). "
                f"Current time: {s.local_time_human} {s.timezone_name}."
            )
        return f"New session. Current time: {s.local_time_human} {s.timezone_name}."

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_system_prompt(self, base_prompt: Optional[str]) -> str:
        grounding = self._snapshot.grounding_note

        if base_prompt:
            return f"{base_prompt}\n\n{grounding}"
        return (
            "You are a helpful assistant.\n\n"
            f"{grounding}\n\n"
            "Always use the temporal context above when interpreting references "
            "to time, dates, or current state. Never assume the current time matches "
            "the time of any previous session."
        )

    def _get_prior_messages(
        self, prior_session_id: Optional[str] = None
    ) -> list[dict]:
        """
        Retrieve, score, and filter messages from the prior session.
        """
        if prior_session_id:
            raw = self._store.get_messages(prior_session_id)
        else:
            prior = self._store.get_last_ended_session(self.user_id)
            if not prior:
                return []
            raw = self._store.get_messages(prior["session_id"])

        if not raw:
            return []

        # Take the most recent N messages
        raw = raw[-self.max_prior_messages:]

        current_ts = self._snapshot.captured_at
        scored: list[ScoredMessage] = score_conversation(raw, current_ts)

        messages_out: list[dict] = []

        for sm in scored:
            if sm.is_stale:
                if self.include_stale_messages:
                    # Include with staleness warning prepended
                    content = f"{sm.staleness_warning}\n\n{sm.content}"
                    messages_out.append({"role": sm.role, "content": content})
                # else: silently drop stale messages
            else:
                messages_out.append({"role": sm.role, "content": sm.content})

        return messages_out
