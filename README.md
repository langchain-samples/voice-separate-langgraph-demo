# LiveKit voice agent with a separately deployed LangGraph agent

This demo runs the voice pipeline and reasoning agent as separate processes:

```text
microphone → LiveKit STT → LiveKit llm_node
                              │ LangGraph SDK (HTTP/SSE + distributed parent)
                              ▼
                         LangGraph Server
                              │ streamed final-answer tokens
                              ▼
speaker ← LiveKit TTS ← LiveKit llm_node
```

LiveKit owns the transcript. Each graph invocation is stateless and receives
LiveKit's complete current `ChatContext`, including its corrected/truncated
assistant message after an interruption. The remote graph therefore never keeps
an unheard continuation in a separate checkpoint.

## Setup

Requires Python 3.11–3.13, `uv`, a microphone, and speakers. The checked-in
`.python-version` selects Python 3.12 for the local environment.

```bash
uv sync
cp .env.example .env
```

Fill in these names in `.env` without committing the file:

- `OPENAI_API_KEY` for the model running in LangGraph Server.
- `ASSEMBLYAI_API_KEY` for streaming STT.
- `CARTESIA_API_KEY` for TTS.
- `LANGSMITH_API_KEY` to see the nested trace.

The dummy `LIVEKIT_*` values are sufficient for console mode; no LiveKit Cloud
room is required.

## Run locally

Terminal 1 starts the separate local LangGraph deployment:

```bash
uv run langgraph dev --no-browser
```

Terminal 2 starts LiveKit's local mic/speaker console:

```bash
uv run voice-separate-langgraph
```

Ask, for example, “What is the weather in San Francisco and Tokyo?” Press
Ctrl-C in both terminals when finished.

Use an explicit audio device when needed:

```bash
LIVEKIT_OUTPUT_DEVICE="MacBook Pro Speakers" uv run voice-separate-langgraph
```

List device IDs and names with:

```bash
uv run python -m sounddevice
```

## Nested tracing

`configure_livekit` makes LiveKit's session, turn, STT, LLM-node, and TTS spans
OpenTelemetry spans. The remote adapter creates a client span inside
`llm_node` and sends two narrow carriers through `client.runs.stream(...)`:

- W3C `traceparent`/`tracestate` for transport and worker telemetry.
- `langsmith-trace`, generated from that same OTel span, for LangGraph's
  LangSmith callback tree.

The deployed export is a per-run graph factory. It validates the
`langsmith-trace` value and installs it in the callback manager before returning
the compiled graph. The graph, model, and tool runs therefore use the LiveKit
client span as their parent instead of starting a second trace.

Both processes must use:

```text
LANGSMITH_PROJECT=voice-separate-langgraph
LANGSMITH_TRACING_MODE=otel
OTEL_ENABLED=true
```

The expected LangSmith tree is:

```text
LiveKit session / turn
└── llm_node
    └── langgraph.runs.stream
        └── LangGraph Server worker.stream_run
            └── graph
                ├── plan/model
                ├── lookup_weather
                └── respond/model
```

The same project name only controls where spans land; the distributed parent
headers are what connect the two processes into one trace. LiveKit exports via
OTel, while the graph process intentionally uses LangSmith SDK mode so the
explicit remote parent IDs are preserved.

## Use a hosted LangGraph deployment

Deploy the graph described by `langgraph.json`, then change only:

```text
LANGGRAPH_URL=https://your-deployment.example
LANGGRAPH_API_KEY=<provided by the deployment>
LANGGRAPH_ASSISTANT_ID=voice_agent
```

The client rejects plain HTTP for non-loopback hosts. The API key is given to
the official LangGraph SDK and is never placed in trace attributes or graph
input.

## Tests

```bash
uv run pytest
uv run ruff check .
```
