"""
Tests for SessionAnchor core modules.
Run with: pytest tests/ -v
"""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from session_anchor.core.temporal import create_temporal_snapshot, _gap_label, _time_of_day_label
from session_anchor.decay.decay import classify_decay, score_message, DecayClass
from session_anchor.memory.store import SessionStore
from session_anchor.anchor import SessionAnchor


# ─── Temporal ─────────────────────────────────────────────────────────────────

class TestTemporalSnapshot:

    def test_new_session_has_no_gap(self):
        snap = create_temporal_snapshot("test-001", tz_name="UTC")
        assert snap.gap_seconds is None
        assert snap.gap_label is None
        assert "no prior context gap" in snap.grounding_note

    def test_resumed_session_calculates_gap(self):
        prev_ended = (datetime.now(tz=timezone.utc) - timedelta(days=3)).isoformat()
        snap = create_temporal_snapshot("test-002", tz_name="UTC", previous_session_ended_at=prev_ended)
        assert snap.gap_seconds is not None
        assert snap.gap_seconds > 0
        assert "days later" in snap.gap_label
        assert "STALE" in snap.grounding_note or "stale" in snap.grounding_note.lower()

    def test_gap_label_moments(self):
        assert _gap_label(60) == "about 1 minute ago"

    def test_gap_label_hours(self):
        label = _gap_label(7200)
        assert "hour" in label

    def test_gap_label_days(self):
        label = _gap_label(86400 * 3)
        assert "day" in label

    def test_time_of_day_labels(self):
        assert _time_of_day_label(8) == "morning"
        assert _time_of_day_label(14) == "afternoon"
        assert _time_of_day_label(19) == "evening"
        assert _time_of_day_label(23) == "night"

    def test_grounding_note_contains_time(self):
        snap = create_temporal_snapshot("test-003", tz_name="UTC")
        assert "SESSION ANCHOR" in snap.grounding_note
        assert "END SESSION ANCHOR" in snap.grounding_note


# ─── Decay ────────────────────────────────────────────────────────────────────

class TestDecayClassification:

    def test_preference_is_permanent(self):
        cls = classify_decay("I prefer concise responses please")
        assert cls == DecayClass.PERMANENT

    def test_time_of_day_is_fast(self):
        cls = classify_decay("I have a meeting at 3pm today")
        assert cls == DecayClass.FAST

    def test_ephemeral_right_now(self):
        cls = classify_decay("I'm in a meeting right now")
        assert cls == DecayClass.EPHEMERAL

    def test_project_work_is_slow(self):
        cls = classify_decay("I'm working on a React project")
        assert cls == DecayClass.SLOW

    def test_stale_fast_message(self):
        now = datetime.now(tz=timezone.utc)
        old_ts = (now - timedelta(hours=10)).isoformat()
        scored = score_message(
            role="user",
            content="I need to finish this before my 3pm meeting",
            message_timestamp=old_ts,
            current_timestamp=now.isoformat(),
        )
        assert scored.is_stale is True
        assert scored.staleness_warning is not None

    def test_permanent_message_never_stale(self):
        now = datetime.now(tz=timezone.utc)
        old_ts = (now - timedelta(days=365)).isoformat()
        scored = score_message(
            role="user",
            content="I prefer bullet points in your responses",
            message_timestamp=old_ts,
            current_timestamp=now.isoformat(),
        )
        assert scored.is_stale is False

    def test_ephemeral_always_stale(self):
        now = datetime.now(tz=timezone.utc)
        scored = score_message(
            role="user",
            content="I'm in a meeting right now",
            message_timestamp=now.isoformat(),
            current_timestamp=now.isoformat(),
        )
        assert scored.is_stale is True


# ─── Store ────────────────────────────────────────────────────────────────────

class TestSessionStore:

    def test_create_and_end_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(store_dir=tmpdir)
            sid = store.create_session(user_id="u1", timezone_name="UTC")
            env = store.get_envelope(sid)
            assert env["session_id"] == sid
            assert env["ended_at"] is None

            store.end_session(sid)
            env = store.get_envelope(sid)
            assert env["ended_at"] is not None

    def test_append_and_retrieve_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(store_dir=tmpdir)
            sid = store.create_session(user_id="u2")
            store.append_message(sid, "user", "Hello")
            store.append_message(sid, "assistant", "Hi there!")
            msgs = store.get_messages(sid)
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[1]["content"] == "Hi there!"

    def test_get_last_ended_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(store_dir=tmpdir)
            sid1 = store.create_session(user_id="u3")
            store.end_session(sid1)
            sid2 = store.create_session(user_id="u3")
            store.end_session(sid2)

            last = store.get_last_ended_session("u3")
            assert last["session_id"] == sid2

    def test_no_prior_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(store_dir=tmpdir)
            result = store.get_last_ended_session("nonexistent_user")
            assert result is None


# ─── SessionAnchor integration ────────────────────────────────────────────────

class TestSessionAnchor:

    def test_full_session_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = SessionAnchor(user_id="test_user", timezone="UTC", store_dir=tmpdir)
            sid = anchor.start()
            assert sid is not None
            assert anchor.snapshot is not None

            anchor.log("user", "Hello")
            anchor.log("assistant", "Hi!")
            anchor.end()
            assert anchor.session_id is None

    def test_build_context_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = SessionAnchor(user_id="test_user", timezone="UTC", store_dir=tmpdir)
            anchor.start()
            messages = anchor.build_context("What time is it?")
            anchor.end()

            roles = [m["role"] for m in messages]
            assert "system" in roles
            assert messages[-1]["role"] == "user"
            assert messages[-1]["content"] == "What time is it?"

    def test_grounding_note_in_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = SessionAnchor(user_id="test_user", timezone="UTC", store_dir=tmpdir)
            anchor.start()
            messages = anchor.build_context("hi")
            anchor.end()

            system = next(m for m in messages if m["role"] == "system")
            assert "SESSION ANCHOR" in system["content"]

    def test_gap_detection_across_sessions(self):
        MONDAY = datetime(2026, 3, 9, 19, 0, 0, tzinfo=timezone.utc)
        THURSDAY = datetime(2026, 3, 12, 14, 0, 0, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1
            with patch("session_anchor.core.temporal.datetime") as mock_dt:
                mock_dt.now.return_value = MONDAY
                mock_dt.fromisoformat = datetime.fromisoformat
                a1 = SessionAnchor(user_id="u1", timezone="UTC", store_dir=tmpdir)
                a1.start()
                a1.log("user", "I have a meeting at 3pm today")
                a1.end()

            # Session 2
            with patch("session_anchor.core.temporal.datetime") as mock_dt:
                mock_dt.now.return_value = THURSDAY
                mock_dt.fromisoformat = datetime.fromisoformat
                a2 = SessionAnchor(user_id="u1", timezone="UTC", store_dir=tmpdir)
                a2.start()
                assert a2.snapshot.gap_seconds is not None
                assert a2.snapshot.gap_seconds > 86400  # More than a day
                a2.end()
