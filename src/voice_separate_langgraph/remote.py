"""Streaming adapter from LiveKit ChatContext to LangGraph Server."""

from __future__ import annotations

import asyncio
import ipaddress
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace

from .tracing import inject_distributed_trace_headers, trace_headers_only

MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 8_000
MAX_TRANSCRIPT_CHARS = 48_000
MAX_RESPONSE_CHARS = 12_000
SPOKEN_NODE = "respond"


class RemoteGraphError(RuntimeError):
    """A safe error raised when the remote graph cannot complete a turn."""


def validate_langgraph_url(url: str) -> str:
    """Allow HTTP only on loopback; require HTTPS everywhere else."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LANGGRAPH_URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "LANGGRAPH_URL must not contain credentials, query, or fragment"
        )

    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        with suppress(ValueError):
            is_loopback = ipaddress.ip_address(hostname).is_loopback
    if parsed.scheme != "https" and not is_loopback:
        raise ValueError("LANGGRAPH_URL must use HTTPS for non-loopback hosts")
    return url.rstrip("/")


def _message_text(item: Any) -> str:
    text = getattr(item, "text_content", None)
    if isinstance(text, str):
        return text
    content = getattr(item, "content", None)
    return content if isinstance(content, str) else ""


def serialize_chat_context(chat_ctx: Any) -> list[dict[str, str]]:
    """Serialize a bounded, recent LiveKit transcript for a stateless run."""
    candidates: list[dict[str, str]] = []
    for item in getattr(chat_ctx, "items", []):
        item_type = getattr(item, "type", "message")
        role = getattr(item, "role", None)
        if item_type != "message" or role not in {"user", "assistant"}:
            continue
        content = _message_text(item)
        if not content:
            continue
        if len(content) > MAX_MESSAGE_CHARS:
            content = content[: MAX_MESSAGE_CHARS - 14] + " …[truncated]"
        message_id = str(getattr(item, "id", ""))[:256]
        candidates.append({"role": role, "content": content, "id": message_id})

    bounded: list[dict[str, str]] = []
    total = 0
    for message in reversed(candidates[-MAX_MESSAGES:]):
        size = len(message["content"])
        if bounded and total + size > MAX_TRANSCRIPT_CHARS:
            break
        bounded.append(message)
        total += size
    bounded.reverse()
    return bounded


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def spoken_text_from_stream_part(part: Any) -> str | None:
    """Extract one final-response token from an SDK stream part."""
    event = getattr(part, "event", None)
    data = getattr(part, "data", None)
    if isinstance(part, Mapping):
        event = part.get("event") or part.get("type")
        data = part.get("data")
    if event not in {"messages", "messages-tuple"}:
        return None
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        return None
    message, metadata = data[0], data[1]
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("langgraph_node") != SPOKEN_NODE
    ):
        return None

    content = message.get("content") if isinstance(message, Mapping) else getattr(
        message, "content", None
    )
    text = _content_text(content)
    return text or None


@dataclass(frozen=True)
class RemoteGraphConfig:
    url: str
    assistant_id: str
    project_name: str
    api_key: str | None = None
    timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        validate_langgraph_url(self.url)
        if not self.assistant_id or len(self.assistant_id) > 200:
            raise ValueError("LANGGRAPH_ASSISTANT_ID is invalid")
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("LANGGRAPH_TIMEOUT_SECONDS must be between 1 and 300")


class RemoteLangGraph:
    """Call a deployed graph and expose only bounded spoken text deltas."""

    def __init__(self, config: RemoteGraphConfig) -> None:
        from langgraph_sdk import get_client

        self._config = config
        self._client = get_client(
            url=validate_langgraph_url(config.url), api_key=config.api_key
        )

    async def stream(
        self, transcript: list[dict[str, str]], *, session_id: str
    ) -> AsyncIterator[str]:
        if not transcript:
            raise RemoteGraphError("No transcript was available for the graph")

        tracer = trace.get_tracer("voice-separate-langgraph.remote")
        remote_stream: Any = None
        emitted = 0
        try:
            with tracer.start_as_current_span(
                "langgraph.runs.stream", kind=trace.SpanKind.CLIENT
            ) as span:
                span.set_attribute(
                    "server.address", urlparse(self._config.url).hostname or ""
                )
                span.set_attribute("langgraph.assistant_id", self._config.assistant_id)
                headers = trace_headers_only(inject_distributed_trace_headers())
                async with asyncio.timeout(self._config.timeout_seconds):
                    remote_stream = self._client.runs.stream(
                        None,
                        self._config.assistant_id,
                        input={"messages": transcript},
                        stream_mode="messages-tuple",
                        on_disconnect="cancel",
                        metadata={"livekit_session_id": session_id[:256]},
                        langsmith_tracing={
                            "project_name": self._config.project_name
                        },
                        headers=headers,
                    )
                    async for part in remote_stream:
                        text = spoken_text_from_stream_part(part)
                        if not text:
                            continue
                        emitted += len(text)
                        if emitted > MAX_RESPONSE_CHARS:
                            raise RemoteGraphError("Remote response exceeded the limit")
                        yield text
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise RemoteGraphError("The remote graph timed out") from exc
        except RemoteGraphError:
            raise
        except Exception as exc:
            raise RemoteGraphError("The remote graph request failed") from exc
        finally:
            if remote_stream is not None and hasattr(remote_stream, "aclose"):
                with suppress(Exception):
                    await remote_stream.aclose()


def total_text(parts: Iterable[str]) -> str:
    """Small test/documentation helper for streamed text."""
    return "".join(parts)
