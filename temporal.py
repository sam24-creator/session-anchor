"""
temporal.py — Core temporal grounding engine for SessionAnchor.

Solves the problem where LLMs carry stale temporal assumptions from prior sessions.
When a session ended at 7pm and resumes days later, the LLM should NOT assume
continuity of time, state, or context validity.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from zoneinfo import ZoneInfo


@dataclass
class TemporalSnapshot:
    """
    A precise temporal snapshot injected at session start.

    This is the core object serialized into the LLM context to anchor it in time.
    """
    session_id: str
    captured_at: str               # ISO-8601 with timezone
    timezone_name: str
    local_time_human: str          # "Tuesday, March 10 2026, 9:14 AM"
    day_of_week: str
    time_of_day_label: str         # "morning" | "afternoon" | "evening" | "night"

    # Gap awareness (only populated on resume)
    previous_session_ended_at: Optional[str] = None
    gap_seconds: Optional[int] = None
    gap_human: Optional[str] = None
    gap_label: Optional[str] = None  # "moments ago" | "hours later" | "days later" etc.

    # Grounding note injected directly into the LLM prompt
    grounding_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _time_of_day_label(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def _gap_label(seconds: int) -> str:
    if seconds < 300:
        return "moments ago"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"about {minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"about {hours} hour{'s' if hours != 1 else ''} later"
    elif seconds < 604800:
        days = seconds // 86400
        return f"{days} day{'s' if days != 1 else ''} later"
    elif seconds < 2592000:
        weeks = seconds // 604800
        return f"{weeks} week{'s' if weeks != 1 else ''} later"
    else:
        months = seconds // 2592000
        return f"about {months} month{'s' if months != 1 else ''} later"


def _build_grounding_note(snapshot: TemporalSnapshot) -> str:
    """
    Constructs the natural-language grounding note injected into the LLM prompt.
    This is the key text that re-anchors the LLM's temporal understanding.
    """
    lines = [
        f"[SESSION ANCHOR]",
        f"The current time is {snapshot.local_time_human} ({snapshot.timezone_name}).",
        f"It is {snapshot.day_of_week} {snapshot.time_of_day_label}.",
    ]

    if snapshot.gap_human and snapshot.gap_label:
        lines.append(
            f"This session is resuming after a gap — the previous session ended "
            f"at {snapshot.previous_session_ended_at}. That was {snapshot.gap_label}."
        )
        lines.append(
            "Any references to 'now', 'today', 'current time', or time-sensitive "
            "context from the previous session should be treated as STALE unless "
            "explicitly re-confirmed in this session."
        )
    else:
        lines.append("This is a new session with no prior context gap.")

    lines.append("[END SESSION ANCHOR]")
    return "\n".join(lines)


def create_temporal_snapshot(
    session_id: str,
    tz_name: str = "UTC",
    previous_session_ended_at: Optional[str] = None,
) -> TemporalSnapshot:
    """
    Create a TemporalSnapshot for the current moment.

    Args:
        session_id:                 Unique identifier for this session.
        tz_name:                    IANA timezone name (e.g. "America/New_York").
        previous_session_ended_at:  ISO-8601 string of when the prior session ended.

    Returns:
        TemporalSnapshot ready to be injected into prompt context.
    """
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz=tz)

    gap_seconds = None
    gap_human = None
    gap_lbl = None

    if previous_session_ended_at:
        prev = datetime.fromisoformat(previous_session_ended_at)
        if prev.tzinfo is None:
            prev = prev.replace(tzinfo=timezone.utc)
        delta = now - prev
        gap_seconds = int(delta.total_seconds())
        gap_human = str(delta)
        gap_lbl = _gap_label(gap_seconds)

    snapshot = TemporalSnapshot(
        session_id=session_id,
        captured_at=now.isoformat(),
        timezone_name=tz_name,
        local_time_human=now.strftime("%A, %B %d %Y, %I:%M %p"),
        day_of_week=now.strftime("%A"),
        time_of_day_label=_time_of_day_label(now.hour),
        previous_session_ended_at=previous_session_ended_at,
        gap_seconds=gap_seconds,
        gap_human=gap_human,
        gap_label=gap_lbl,
    )
    snapshot.grounding_note = _build_grounding_note(snapshot)
    return snapshot
