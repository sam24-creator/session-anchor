"""
adapters.py — Drop-in adapters for common LLM APIs.

Wraps SessionAnchor.build_context() output into the exact format
each provider expects.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from session_anchor.anchor import SessionAnchor


def to_openai_messages(
    anchor: "SessionAnchor",
    user_message: str,
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """
    Build a message list in OpenAI Chat Completions format.

    Usage:
        from openai import OpenAI
        from session_anchor.adapters.adapters import to_openai_messages

        client = OpenAI()
        messages = to_openai_messages(anchor, "What were we discussing?")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
    """
    return anchor.build_context(
        user_message=user_message,
        system_prompt=system_prompt,
    )


def to_anthropic_messages(
    anchor: "SessionAnchor",
    user_message: str,
    system_prompt: Optional[str] = None,
) -> tuple[str, list[dict]]:
    """
    Build a (system, messages) tuple for the Anthropic Messages API.

    The Anthropic API takes the system prompt separately from the messages list.

    Usage:
        import anthropic
        from session_anchor.adapters.adapters import to_anthropic_messages

        client = anthropic.Anthropic()
        system, messages = to_anthropic_messages(anchor, "What were we discussing?")
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
    """
    all_messages = anchor.build_context(
        user_message=user_message,
        system_prompt=system_prompt,
    )

    system_content = ""
    conversation = []

    for msg in all_messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            conversation.append({"role": msg["role"], "content": msg["content"]})

    return system_content, conversation


def to_langchain_messages(
    anchor: "SessionAnchor",
    user_message: str,
    system_prompt: Optional[str] = None,
) -> list:
    """
    Build a list of LangChain message objects.

    Usage:
        from session_anchor.adapters.adapters import to_langchain_messages
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o")
        messages = to_langchain_messages(anchor, "What were we discussing?")
        response = llm.invoke(messages)
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    except ImportError:
        raise ImportError(
            "langchain-core is required for LangChain adapter. "
            "Install it with: pip install langchain-core"
        )

    raw = anchor.build_context(user_message=user_message, system_prompt=system_prompt)
    lc_messages = []

    for msg in raw:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    return lc_messages
