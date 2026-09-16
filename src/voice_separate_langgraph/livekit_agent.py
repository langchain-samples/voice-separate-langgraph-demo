"""LiveKit console voice pipeline whose LLM node calls LangGraph Server."""

from __future__ import annotations

import asyncio
import os
import sys

from .prompts import GREETING, RESPONSE_PROMPT
from .remote import (
    RemoteGraphConfig,
    RemoteGraphError,
    RemoteLangGraph,
    serialize_chat_context,
)

STT_MODEL = os.getenv("LIVEKIT_STT_MODEL") or None
TTS_MODEL = os.getenv("LIVEKIT_TTS_MODEL") or None
TTS_VOICE = os.getenv("LIVEKIT_TTS_VOICE") or None


def _require_environment() -> None:
    for name, purpose in (
        ("OPENAI_API_KEY", "the remote graph model and LiveKit LLM placeholder"),
        ("ASSEMBLYAI_API_KEY", "LiveKit speech-to-text"),
        ("CARTESIA_API_KEY", "LiveKit text-to-speech"),
    ):
        if not os.environ.get(name):
            raise RuntimeError(f"{name} is required for {purpose}")


def run(config: RemoteGraphConfig) -> None:
    """Launch the blocking LiveKit local-console application."""
    try:
        _require_environment()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    from langsmith.integrations.livekit import configure_livekit
    from livekit import agents
    from livekit.agents import Agent, AgentSession, TurnHandlingOptions, llm, room_io
    from livekit.plugins import assemblyai, cartesia, silero
    from livekit.plugins import openai as lk_openai
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    from .livekit_console import run_livekit_console

    if os.environ.get("LANGSMITH_API_KEY"):
        configure_livekit(project=config.project_name)
        print(
            f"[voice] LangSmith tracing enabled in project {config.project_name!r}.",
            file=sys.stderr,
        )
    else:
        print(
            "[voice] LANGSMITH_API_KEY is unset; tracing is disabled.",
            file=sys.stderr,
        )

    def build_session() -> AgentSession:
        tts_kwargs = {}
        if TTS_MODEL:
            tts_kwargs["model"] = TTS_MODEL
        if TTS_VOICE:
            tts_kwargs["voice"] = TTS_VOICE
        return AgentSession(
            stt=assemblyai.STT(model=STT_MODEL) if STT_MODEL else assemblyai.STT(),
            # LiveKit requires an LLM object for capabilities even though the
            # overridden llm_node performs no inference through this instance.
            llm=lk_openai.LLM(model="gpt-4o-mini", temperature=0.3),
            tts=cartesia.TTS(**tts_kwargs),
            vad=silero.VAD.load(),
            turn_handling=TurnHandlingOptions(
                turn_detection=MultilingualModel(), interruption={"mode": "vad"}
            ),
        )

    class Assistant(Agent):
        def __init__(self, remote: RemoteLangGraph, session_id: str) -> None:
            super().__init__(instructions=RESPONSE_PROMPT)
            self._remote = remote
            self._session_id = session_id

        async def llm_node(
            self,
            chat_ctx: llm.ChatContext,
            tools: list[llm.Tool],
            model_settings,
        ):
            del tools, model_settings
            transcript = serialize_chat_context(chat_ctx)
            try:
                async for text in self._remote.stream(
                    transcript, session_id=self._session_id
                ):
                    yield text
            except asyncio.CancelledError:
                # Closing the SDK iterator cancels the remote run because the
                # request uses on_disconnect="cancel".
                raise
            except RemoteGraphError as exc:
                print(f"[voice] {exc}", file=sys.stderr)
                yield "I'm sorry, I couldn't reach my reasoning service just now."

    server = agents.AgentServer()

    @server.rtc_session()
    async def entrypoint(ctx: agents.JobContext) -> None:
        session_id = str(ctx.job.id)
        remote = RemoteLangGraph(config)
        session = build_session()
        await session.start(
            room=ctx.room,
            agent=Assistant(remote, session_id),
            room_options=room_io.RoomOptions(),
            record={"audio": True},
        )
        await session.say(GREETING)

    run_livekit_console(server)
