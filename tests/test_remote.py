from __future__ import annotations

from dataclasses import dataclass

import pytest

from voice_separate_langgraph.remote import (
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES,
    serialize_chat_context,
    spoken_text_from_stream_part,
    validate_langgraph_url,
)


@dataclass
class Message:
    role: str
    text_content: str
    id: str
    type: str = "message"


@dataclass
class ChatContext:
    items: list[Message]


@dataclass
class StreamPart:
    event: str
    data: object


def test_url_allows_local_http_and_requires_remote_https() -> None:
    assert validate_langgraph_url("http://127.0.0.1:2024") == (
        "http://127.0.0.1:2024"
    )
    assert validate_langgraph_url("https://example.langgraph.app") == (
        "https://example.langgraph.app"
    )
    with pytest.raises(ValueError, match="HTTPS"):
        validate_langgraph_url("http://example.com")
    with pytest.raises(ValueError, match="credentials"):
        validate_langgraph_url("https://token@example.com")


def test_transcript_is_recent_bounded_and_drops_system_messages() -> None:
    messages = [Message("system", "secret instruction", "system")]
    messages.extend(
        Message("user", f"message {index}", str(index)) for index in range(50)
    )
    messages.append(Message("assistant", "x" * (MAX_MESSAGE_CHARS + 50), "last"))

    serialized = serialize_chat_context(ChatContext(messages))

    assert len(serialized) == MAX_MESSAGES
    assert serialized[-1]["id"] == "last"
    assert len(serialized[-1]["content"]) <= MAX_MESSAGE_CHARS
    assert all(message["role"] != "system" for message in serialized)


def test_only_respond_node_message_tokens_are_spoken() -> None:
    response = StreamPart(
        event="messages",
        data=[{"content": "Sunny"}, {"langgraph_node": "respond"}],
    )
    planner = StreamPart(
        event="messages",
        data=[{"content": "Calling a tool"}, {"langgraph_node": "plan"}],
    )
    assert spoken_text_from_stream_part(response) == "Sunny"
    assert spoken_text_from_stream_part(planner) is None


def test_structured_text_content_is_supported() -> None:
    part = {
        "event": "messages",
        "data": [
            {"content": [{"type": "text", "text": "Hello"}]},
            {"langgraph_node": "respond"},
        ],
    }
    assert spoken_text_from_stream_part(part) == "Hello"
