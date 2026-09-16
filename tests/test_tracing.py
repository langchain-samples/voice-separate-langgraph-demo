from __future__ import annotations

from contextlib import contextmanager
from uuid import UUID

import pytest
from langchain_core.callbacks.manager import AsyncCallbackManager
from opentelemetry.sdk.trace import TracerProvider

from voice_separate_langgraph.tracing import (
    inject_distributed_trace_headers,
    install_distributed_parent_callbacks,
    trace_headers_only,
    valid_langsmith_trace,
)


def test_injected_traceparent_matches_active_span() -> None:
    provider = TracerProvider()
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("livekit.llm_node") as parent:
        headers = inject_distributed_trace_headers()

    context = parent.get_span_context()
    assert headers["traceparent"].split("-")[1] == f"{context.trace_id:032x}"
    assert headers["traceparent"].split("-")[2] == f"{context.span_id:016x}"

    dotted_parts = headers["langsmith-trace"].split(".")
    assert UUID(dotted_parts[0][-36:]) == UUID(int=context.trace_id)
    assert UUID(dotted_parts[-1][-36:]).int == context.span_id
    assert valid_langsmith_trace(headers["langsmith-trace"])


def test_only_w3c_trace_headers_cross_the_boundary() -> None:
    assert trace_headers_only(
        {
            "TraceParent": "parent",
            "tracestate": "state",
            "LangSmith-Trace": "dotted",
            "authorization": "must-not-pass",
            "baggage": "must-not-pass",
        }
    ) == {
        "traceparent": "parent",
        "tracestate": "state",
        "langsmith-trace": "dotted",
    }


@pytest.mark.parametrize(
    "value",
    [None, "", "not-a-trace", "0" * 2_000, "20260101T000000000000Zbad"],
)
def test_malformed_langsmith_parent_is_rejected(value: object) -> None:
    assert not valid_langsmith_trace(value)


def test_graph_factory_installs_validated_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("langgraph.runs.stream"):
        header = inject_distributed_trace_headers()["langsmith-trace"]

    captured: dict[str, object] = {}
    callback_sentinel = object()

    @contextmanager
    def fake_tracing_context(**kwargs):
        captured.update(kwargs)
        yield

    monkeypatch.setattr(
        "langsmith.run_helpers.tracing_context", fake_tracing_context
    )
    monkeypatch.setattr(
        AsyncCallbackManager,
        "configure",
        staticmethod(lambda: callback_sentinel),
    )
    config = {"configurable": {"langsmith-trace": header}}

    assert install_distributed_parent_callbacks(config, "project")
    assert config["callbacks"] is callback_sentinel
    assert captured["parent"] == {"langsmith-trace": header}
    assert captured["project_name"] == "project"
