"""The separately deployed LangGraph weather agent."""

from __future__ import annotations

import os
from typing import Annotated

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from voice_separate_langgraph.prompts import PLANNER_PROMPT, RESPONSE_PROMPT
from voice_separate_langgraph.tracing import (
    configure_graph_process,
    install_distributed_parent_callbacks,
)
from voice_separate_langgraph.weather import fetch_weather

PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "voice-separate-langgraph")
MODEL_NAME = os.getenv("LANGGRAPH_MODEL", "gpt-4o-mini")

# This runs when LangGraph Server loads the graph, before the first request.
# It is a no-op when tracing is disabled or the hosted runtime already owns OTel.
configure_graph_process(PROJECT_NAME)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
async def lookup_weather(city: str) -> dict:
    """Get current weather for one city. Call once for each requested city."""
    return await fetch_weather(city)


_planner_model = ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(
    [lookup_weather]
)
_response_model = ChatOpenAI(model=MODEL_NAME, temperature=0.3)


async def plan(state: AgentState) -> dict:
    """Choose weather tool calls without producing spoken content."""
    result = await _planner_model.ainvoke(
        [SystemMessage(content=PLANNER_PROMPT), *state["messages"]]
    )
    return {"messages": [result]}


def after_plan(state: AgentState) -> str:
    """Run requested tools, otherwise proceed directly to the response node."""
    latest = state["messages"][-1]
    return "tools" if getattr(latest, "tool_calls", None) else "respond"


async def respond(state: AgentState) -> dict:
    """Generate the only user-facing, streamable answer."""
    result = await _response_model.ainvoke(
        [SystemMessage(content=RESPONSE_PROMPT), *state["messages"]]
    )
    return {"messages": [result]}


builder = StateGraph(AgentState)
builder.add_node("plan", plan)
builder.add_node("tools", ToolNode([lookup_weather]))
builder.add_node("respond", respond)
builder.add_edge(START, "plan")
builder.add_conditional_edges(
    "plan", after_plan, {"tools": "tools", "respond": "respond"}
)
builder.add_edge("tools", "respond")
builder.add_edge("respond", END)

# No checkpointer: every invocation receives LiveKit's complete, corrected
# transcript. This prevents an interrupted, unheard response from surviving in
# a separate server-side conversation history.
_compiled_graph = builder.compile()


def make_graph(config: dict):
    """Return the graph with its remote LiveKit parent installed per run."""
    install_distributed_parent_callbacks(config, PROJECT_NAME)
    return _compiled_graph
