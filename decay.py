"""
decay.py — Context decay scoring for SessionAnchor.

Different types of information go stale at different rates.
"I'll be at the office at 3pm today" decays immediately after the session.
"I prefer concise responses" never decays.

This module scores prior messages and flags stale content before
it gets re-injected into a new session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DecayClass(str, Enum):
    """
    How quickly a piece of context becomes unreliable.
    """
    PERMANENT   = "permanent"   # User preferences, identity — never stale
    SLOW        = "slow"        # Project goals, ongoing tasks — stale after weeks
    MEDIUM      = "medium"      # Specific plans, work-in-progress — stale after days
    FAST        = "fast"        # Today's schedule, current time — stale after hours
    EPHEMERAL   = "ephemeral"   # Right-now states ("I'm in a meeting") — stale after session


# Seconds after which each class is considered stale
DECAY_THRESHOLDS: dict[DecayClass, Optional[int]] = {
    DecayClass.PERMANENT:  None,         # Never stale
    DecayClass.SLOW:       60 * 60 * 24 * 14,  # 2 weeks
    DecayClass.MEDIUM:     60 * 60 * 24 * 2,   # 2 days
    DecayClass.FAST:       60 * 60 * 6,         # 6 hours
    DecayClass.EPHEMERAL:  0,                   # Immediately after session ends
}

# Pattern → DecayClass mapping (checked in order, first match wins)
_DECAY_PATTERNS: list[tuple[re.Pattern, DecayClass]] = [
    # Ephemeral: right-now states
    (re.compile(r"\b(right now|at the moment|currently|i'm in|i am in|just now)\b", re.I), DecayClass.EPHEMERAL),

    # Fast: time-of-day specific
    (re.compile(r"\b(\d{1,2}(:\d{2})?\s*(am|pm)|this morning|this afternoon|this evening|tonight|today at)\b", re.I), DecayClass.FAST),
    (re.compile(r"\b(by eod|end of day|before lunch|after lunch|in an hour|in \d+ minutes?)\b", re.I), DecayClass.FAST),

    # Medium: day-scoped plans
    (re.compile(r"\b(today|tomorrow|this week|next week|on monday|on friday)\b", re.I), DecayClass.MEDIUM),
    (re.compile(r"\b(deadline|due date|meeting|appointment|scheduled for)\b", re.I), DecayClass.MEDIUM),

    # Slow: project/task continuity
    (re.compile(r"\b(working on|project|task|goal|plan to|trying to|building)\b", re.I), DecayClass.SLOW),

    # Permanent: preferences and identity
    (re.compile(r"\b(i prefer|i like|i want|i always|my name|call me|i'm a|i am a)\b", re.I), DecayClass.PERMANENT),
]


@dataclass
class ScoredMessage:
    """
    A message from a prior session with its decay assessment.
    """
    role: str                  # "user" | "assistant" | "system"
    content: str
    timestamp: Optional[str]   # ISO-8601 when this message was sent
    decay_class: DecayClass
    is_stale: bool
    staleness_reason: Optional[str]
    staleness_warning: Optional[str]   # Human-readable text to inject if re-used


def classify_decay(content: str) -> DecayClass:
    """
    Classify the decay class of a message based on its content.
    Pattern matching is intentionally simple and fast — override
    with your own classifier for production use.
    """
    for pattern, cls in _DECAY_PATTERNS:
        if pattern.search(content):
            return cls
    return DecayClass.SLOW   # Default: assume moderate temporal relevance


def score_message(
    role: str,
    content: str,
    message_timestamp: Optional[str],
    current_timestamp: str,
    decay_class: Optional[DecayClass] = None,
) -> ScoredMessage:
    """
    Score a single message for staleness.

    Args:
        role:                Role of the message author.
        content:             Raw message text.
        message_timestamp:   ISO-8601 when the message was originally sent.
        current_timestamp:   ISO-8601 of the current session start.
        decay_class:         Override auto-classification if known.

    Returns:
        ScoredMessage with staleness assessment.
    """
    from datetime import datetime, timezone

    cls = decay_class or classify_decay(content)
    threshold = DECAY_THRESHOLDS[cls]

    is_stale = False
    reason = None
    warning = None

    if threshold == 0:
        is_stale = True
        reason = f"Ephemeral content — only valid within the originating session."
    elif threshold is not None and message_timestamp:
        try:
            msg_dt = datetime.fromisoformat(message_timestamp)
            cur_dt = datetime.fromisoformat(current_timestamp)
            if msg_dt.tzinfo is None:
                msg_dt = msg_dt.replace(tzinfo=timezone.utc)
            if cur_dt.tzinfo is None:
                cur_dt = cur_dt.replace(tzinfo=timezone.utc)
            gap = (cur_dt - msg_dt).total_seconds()
            if gap > threshold:
                is_stale = True
                reason = (
                    f"Content classified as '{cls.value}' with {threshold//3600}h threshold; "
                    f"message is {gap/3600:.1f}h old."
                )
        except ValueError:
            pass  # Unparseable timestamps treated as non-stale

    if is_stale:
        warning = (
            f"[STALE CONTEXT — {cls.value.upper()}] "
            f"The following message may no longer be accurate: {reason or ''}"
        )

    return ScoredMessage(
        role=role,
        content=content,
        timestamp=message_timestamp,
        decay_class=cls,
        is_stale=is_stale,
        staleness_reason=reason,
        staleness_warning=warning,
    )


def score_conversation(
    messages: list[dict],
    current_timestamp: str,
) -> list[ScoredMessage]:
    """
    Score an entire conversation history for staleness.

    Args:
        messages:           List of dicts with 'role', 'content', and optional 'timestamp'.
        current_timestamp:  ISO-8601 of the current session start.

    Returns:
        List of ScoredMessage objects.
    """
    scored = []
    for msg in messages:
        scored.append(score_message(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            message_timestamp=msg.get("timestamp"),
            current_timestamp=current_timestamp,
        ))
    return scored
