# SessionAnchor 🕰️

**Temporal context grounding for LLM sessions.**

Prevents language models from carrying stale time-based assumptions when conversations resume after a gap.

---

## The Problem

When you end a chat session at 7pm on Monday and resume it Thursday morning, most LLM applications inject the old conversation history without any temporal grounding. The model may then:

- Still think it's 7pm Monday
- Treat a deadline from "tomorrow" as still upcoming — even though it passed 2 days ago
- Reference "current" context that is days out of date
- Answer questions about "now" based on the old session's temporal frame

This is not just a cosmetic issue. It causes real errors in scheduling assistants, task managers, coding agents, personal AI tools, and any application where time matters.

---

## The Solution

SessionAnchor wraps every LLM call with three layers of protection:

### 1. Temporal Snapshot Injection
Every session start captures the exact current time, timezone, and day context. This is injected as a structured block into the system prompt before any history is included:

```
[SESSION ANCHOR]
The current time is Thursday, March 12 2026, 9:00 AM (America/New_York).
It is Thursday morning.
This session is resuming after a gap — the previous session ended at 2026-03-09T19:00:00.
That was 3 days later.
Any references to 'now', 'today', 'current time', or time-sensitive context
from the previous session should be treated as STALE unless explicitly
re-confirmed in this session.
[END SESSION ANCHOR]
```

### 2. Context Decay Scoring
Messages from prior sessions are classified by how quickly they go stale:

| Decay Class | Examples | Staleness Threshold |
|-------------|----------|-------------------|
| `PERMANENT` | "I prefer concise replies", "my name is Alex" | Never stale |
| `SLOW` | "I'm building a React app", project goals | 2 weeks |
| `MEDIUM` | "meeting next Tuesday", deadlines | 2 days |
| `FAST` | "see you at 3pm today", time-of-day plans | 6 hours |
| `EPHEMERAL` | "I'm in a meeting right now" | Immediately after session |

Stale messages are either dropped or re-injected with an explicit warning — your choice.

### 3. Gap-Aware History Filtering
Only relevant, non-stale messages from prior sessions are carried forward. The model is never confused by context it can no longer trust.

---

## Quick Start

```bash
pip install session-anchor
```

```python
from session_anchor import SessionAnchor

# Initialize for a user
anchor = SessionAnchor(
    user_id="user_123",
    timezone="America/New_York",
)

# Start a session (auto-detects gap from last session)
session_id = anchor.start()

# Build grounded context to send to your LLM
messages = anchor.build_context(
    user_message="What were we working on?",
    system_prompt="You are a helpful assistant.",
)

# Use messages with any LLM API
import anthropic
client = anthropic.Anthropic()

# Or use the built-in adapter:
from session_anchor.adapters import to_anthropic_messages
system, msgs = to_anthropic_messages(anchor, "What were we working on?")

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system=system,
    messages=msgs,
)

# Log the turn (important — enables gap detection in the next session)
anchor.log("user", "What were we working on?")
anchor.log("assistant", response.content[0].text)

# End the session when the user is done
anchor.end()
```

---

## Adapters

SessionAnchor ships with adapters for popular LLM frameworks:

### OpenAI
```python
from session_anchor.adapters import to_openai_messages
from openai import OpenAI

client = OpenAI()
messages = to_openai_messages(anchor, "Continue where we left off")

response = client.chat.completions.create(model="gpt-4o", messages=messages)
```

### Anthropic
```python
from session_anchor.adapters import to_anthropic_messages
import anthropic

client = anthropic.Anthropic()
system, messages = to_anthropic_messages(anchor, "Continue where we left off")

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system=system,
    messages=messages,
)
```

### LangChain
```python
from session_anchor.adapters import to_langchain_messages
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
messages = to_langchain_messages(anchor, "Continue where we left off")
response = llm.invoke(messages)
```

---

## Advanced Usage

### Custom decay classification

Override the built-in pattern matcher with your own classifier:

```python
from session_anchor.decay import DecayClass, score_message

# Manually classify a message
scored = score_message(
    role="user",
    content="I need to finish this by the sprint deadline",
    message_timestamp="2026-03-09T19:00:00+00:00",
    current_timestamp="2026-03-12T14:00:00+00:00",
    decay_class=DecayClass.MEDIUM,  # Override auto-detection
)

print(scored.is_stale)       # True (> 2 day threshold)
print(scored.staleness_warning)
```

### Include stale messages with warnings

```python
anchor = SessionAnchor(
    user_id="user_123",
    timezone="UTC",
    include_stale_messages=True,  # Include stale, but flag them
)
```

When stale messages are included, they're prepended with:
```
[STALE CONTEXT — FAST] The following message may no longer be accurate:
Content classified as 'fast' with 6h threshold; message is 74.2h old.

I need to finish this before my 3pm meeting
```

### Inspect the temporal snapshot

```python
anchor.start()
snap = anchor.snapshot

print(snap.local_time_human)    # "Thursday, March 12 2026, 9:00 AM"
print(snap.gap_label)           # "3 days later"
print(snap.gap_seconds)         # 259200
print(snap.grounding_note)      # Full text injected into the LLM prompt
print(anchor.get_gap_summary()) # Human-readable summary
```

### Custom storage backend

The default `SessionStore` writes JSON files to `.session_anchor/`. Swap it out:

```python
from session_anchor.memory.store import SessionStore

# The store interface is simple — implement these methods for any backend:
# create_session(), end_session(), get_last_ended_session()
# append_message(), get_messages(), get_envelope()

class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str):
        import redis
        self.r = redis.from_url(redis_url)
    # ... implement methods
```

---

## How It Compares

| Approach | Temporal grounding | Decay scoring | History filtering | Framework support |
|---|---|---|---|---|
| Raw message injection | ❌ | ❌ | ❌ | — |
| System prompt with `datetime.now()` | ⚠️ Partial | ❌ | ❌ | Manual |
| LangChain Memory | ❌ | ❌ | ⚠️ Token limit only | LangChain only |
| **SessionAnchor** | ✅ | ✅ | ✅ | OpenAI / Anthropic / LangChain |

---

## Architecture

```
session_anchor/
├── anchor.py              # Main public API — SessionAnchor class
├── core/
│   └── temporal.py        # Temporal snapshot creation + grounding note generation
├── decay/
│   └── decay.py           # Context decay classification + staleness scoring
├── memory/
│   └── store.py           # Session persistence (file-backed, swappable)
└── adapters/
    └── adapters.py        # OpenAI / Anthropic / LangChain adapters
```

---

## Contributing

Contributions welcome. Key areas for extension:

- **Richer decay classifiers** — NLP-based or LLM-based semantic decay scoring
- **Vector memory integration** — Connect to Chroma, Weaviate, or Pinecone for semantic retrieval of only relevant prior context
- **Storage backends** — Redis, SQLite, PostgreSQL implementations
- **More adapters** — Ollama, Mistral, Cohere, LiteLLM
- **Timezone detection** — Auto-detect user timezone from browser/IP
- **Session summarization** — Compress long sessions into structured memory objects before they're used as prior context

---

## License

MIT
