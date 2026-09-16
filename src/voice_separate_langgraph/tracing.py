"""LangSmith/OpenTelemetry setup and narrow W3C context propagation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

_TRACE_HEADER_NAMES = frozenset(
    {"traceparent", "tracestate", "langsmith-trace"}
)
_DOTTED_SEGMENT = (
    r"[0-9]{8}T[0-9]{12}Z"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_DOTTED_ORDER_RE = re.compile(rf"{_DOTTED_SEGMENT}(?:\.{_DOTTED_SEGMENT}){{1,7}}")
_MAX_DOTTED_ORDER_CHARS = 1_024


def configure_environment(project_name: str) -> None:
    """Set tracing defaults before importing LiveKit or graph dependencies."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_TRACING_MODE", "otel")
    os.environ.setdefault("OTEL_ENABLED", "true")
    os.environ["LANGSMITH_PROJECT"] = project_name


def configure_graph_process(project_name: str) -> bool:
    """Configure the graph process to emit LangSmith SDK runs.

    LiveKit stays in OTel mode. The graph uses SDK mode because its graph factory
    installs an explicit distributed RunTree parent received in
    ``langsmith-trace``. This bridge is necessary because W3C transport context
    alone does not seed LangGraph's LangSmith callback tree in every server mode.
    """
    configure_environment(project_name)
    if not os.environ.get("LANGSMITH_API_KEY"):
        return False
    os.environ["LANGSMITH_TRACING_MODE"] = "langsmith"
    os.environ["LANGSMITH_PROJECT"] = project_name
    return True


def inject_w3c_trace_headers() -> dict[str, str]:
    """Return only W3C trace headers from the current OTel context."""
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return {
        key: value
        for key, value in carrier.items()
        if key.lower() in _TRACE_HEADER_NAMES
    }


def _dotted_segment(when: datetime, run_id: UUID) -> str:
    return f"{when.strftime('%Y%m%dT%H%M%S%fZ')}{run_id}"


def inject_langsmith_parent_header() -> dict[str, str]:
    """Represent the current OTel span as a LangSmith distributed parent.

    LangSmith's OTel ingest deterministically maps an 8-byte OTel span ID to a
    zero-padded UUID. Using the same mapping here lets SDK spans emitted by the
    remote graph name the already-exported LiveKit client span as their parent.
    """
    from langsmith.integrations.otel import langsmith_run_id_from_otel_span_id
    from opentelemetry import trace

    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return {}

    trace_id = UUID(int=context.trace_id)
    parent_run_id = langsmith_run_id_from_otel_span_id(context.span_id)
    now = datetime.now(UTC)
    dotted_order = ".".join(
        (
            _dotted_segment(now - timedelta(microseconds=1), trace_id),
            _dotted_segment(now, parent_run_id),
        )
    )
    return {"langsmith-trace": dotted_order}


def inject_distributed_trace_headers() -> dict[str, str]:
    """Return the W3C transport carrier plus the LangSmith callback parent."""
    return trace_headers_only(
        {**inject_w3c_trace_headers(), **inject_langsmith_parent_header()}
    )


def valid_langsmith_trace(value: Any) -> bool:
    """Validate the only external value allowed to influence graph ancestry."""
    return (
        isinstance(value, str)
        and len(value) <= _MAX_DOTTED_ORDER_CHARS
        and _DOTTED_ORDER_RE.fullmatch(value) is not None
    )


def install_distributed_parent_callbacks(
    config: dict[str, Any], project_name: str
) -> bool:
    """Seed a graph factory's callback manager from a validated parent header."""
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return False
    parent = configurable.get("langsmith-trace")
    if not valid_langsmith_trace(parent):
        return False

    from langchain_core.callbacks.manager import AsyncCallbackManager
    from langsmith.run_helpers import tracing_context

    with tracing_context(
        parent={"langsmith-trace": parent}, project_name=project_name
    ):
        config["callbacks"] = AsyncCallbackManager.configure()
    return True


def trace_headers_only(headers: Mapping[str, str]) -> dict[str, str]:
    """Allowlist helper used at the outbound service boundary."""
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in _TRACE_HEADER_NAMES
    }
