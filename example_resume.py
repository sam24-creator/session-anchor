"""
example_resume.py — Demonstrates the core SessionAnchor use case.

Simulates a user who:
  Session 1 (Monday 7pm): Discusses a deadline "due tomorrow at 9am"
  Session 2 (Thursday morning): Resumes — that deadline is now stale

Without SessionAnchor: LLM would still reference the Monday deadline as relevant.
With SessionAnchor:    Stale content is flagged; LLM is re-anchored to Thursday morning.
"""

import json
import time
from datetime import datetime, timezone, timedelta

# Patch the clock so we can simulate a multi-day gap in a single script run
import session_anchor.core.temporal as _temporal_module
import session_anchor.memory.store as _store_module

_SIMULATED_NOW: datetime = None

_real_datetime_now = datetime.now

def _patched_now(tz=None):
    if _SIMULATED_NOW is not None:
        return _SIMULATED_NOW.astimezone(tz) if tz else _SIMULATED_NOW
    return _real_datetime_now(tz=tz)

# Monkey-patch for demo purposes only
from unittest.mock import patch

# ─── Setup ────────────────────────────────────────────────────────────────────

STORE_DIR = "/tmp/session_anchor_demo"

from session_anchor import SessionAnchor

# ─── Session 1: Monday at 7pm ─────────────────────────────────────────────────

MONDAY_7PM = datetime(2026, 3, 9, 19, 0, 0, tzinfo=timezone.utc)

print("=" * 60)
print("SESSION 1 — Monday 7:00 PM")
print("=" * 60)

with patch("session_anchor.core.temporal.datetime") as mock_dt:
    mock_dt.now.return_value = MONDAY_7PM
    mock_dt.fromisoformat = datetime.fromisoformat

    anchor1 = SessionAnchor(
        user_id="demo_user",
        timezone="America/New_York",
        store_dir=STORE_DIR,
    )
    session1_id = anchor1.start()
    print(f"Session started: {session1_id}")
    print(f"Snapshot: {anchor1.snapshot.local_time_human}")
    print(f"Gap note: {anchor1.get_gap_summary()}\n")

    # Simulate conversation about a time-sensitive task
    anchor1.log("user", "I have a presentation due tomorrow at 9am, help me outline it")
    anchor1.log("assistant", "Sure! Here's an outline for your 9am presentation tomorrow...")
    anchor1.log("user", "Great, I'm heading to bed now, I'll finish it in the morning")
    anchor1.log("assistant", "Good luck! You have about 14 hours before your deadline.")

    anchor1.end()
    print("Session 1 ended.\n")

# ─── Session 2: Thursday at 9am (3 days later) ───────────────────────────────

THURSDAY_9AM = datetime(2026, 3, 12, 14, 0, 0, tzinfo=timezone.utc)  # 9am ET = 2pm UTC

print("=" * 60)
print("SESSION 2 — Thursday 9:00 AM (3 days later)")
print("=" * 60)

with patch("session_anchor.core.temporal.datetime") as mock_dt:
    mock_dt.now.return_value = THURSDAY_9AM
    mock_dt.fromisoformat = datetime.fromisoformat

    anchor2 = SessionAnchor(
        user_id="demo_user",
        timezone="America/New_York",
        store_dir=STORE_DIR,
        include_stale_messages=True,   # Show stale messages with warnings
        max_prior_messages=10,
    )
    session2_id = anchor2.start()

    print(f"Session started: {session2_id}")
    print(f"Snapshot: {anchor2.snapshot.local_time_human}")
    print(f"Gap summary: {anchor2.get_gap_summary()}\n")

    # Build the context that would be sent to the LLM
    messages = anchor2.build_context(
        user_message="Can you remind me what I was working on?",
        system_prompt="You are a helpful productivity assistant.",
    )

    print("─── Messages that would be sent to LLM ───")
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content_preview = msg["content"][:300].replace("\n", " ")
        print(f"\n[{i}] {role}:\n  {content_preview}...")

    print("\n─── Grounding note injected ───")
    print(anchor2.snapshot.grounding_note)

    anchor2.end()

print("\n✓ Demo complete.")
print(f"\nKey insight: The LLM now knows it's Thursday 9am, NOT Monday 7pm.")
print("The stale 'presentation due tomorrow' context is flagged accordingly.")
