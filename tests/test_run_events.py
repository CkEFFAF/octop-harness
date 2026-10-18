from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from langgraph.config import var_child_runnable_config

from octop_harness.config import HarnessAgentConfig
from octop_harness.observability import RunEventCallback


@pytest.mark.parametrize("transport", ["accepted", "completed", "failed", "cancelled", "unknown"])
def test_dispatch_receipt_retains_transport_and_distinct_task_outcome(transport: str) -> None:
    events: list[dict[str, Any]] = []
    callback = RunEventCallback(events.append)
    identity = uuid4()
    callback.on_tool_start({"name": "ask_agent"}, "", run_id=identity, inputs={"mode": "background"})
    callback.on_tool_end(
        json.dumps(
            {
                "dispatch_receipt": {
                    "invocation_id": "peer-invocation",
                    "transport_status": transport,
                    "task_outcome": "blocked",
                    "child_thread_id": "child",
                    "retryable": False,
                }
            }
        ),
        run_id=identity,
    )
    assert events[-1]["status"] == transport
    assert events[-1]["payload"]["task_outcome"] == "blocked"
    assert events[-1]["payload"]["invocation_id"] == "peer-invocation"


@pytest.mark.asyncio
async def test_actual_failed_fetch_and_missing_expert_cannot_be_success_evidence() -> None:
    from types import SimpleNamespace

    from octop_harness.builtin.tools.web_fetch import web_fetch
    from octop_harness.teams.tools import build_team_tools

    events: list[dict[str, Any]] = []
    callback = RunEventCallback(events.append)
    for url in ("http://127.0.0.1/source-one", "http://127.0.0.1/source-two"):
        identity = uuid4()
        callback.on_tool_start({"name": "web_fetch"}, "", run_id=identity, inputs={"url": url})
        output = await web_fetch.ainvoke({"url": url})
        assert output.startswith("Error:")
        callback.on_tool_end(output, run_id=identity)
        assert events[-1]["status"] == "failed"
        assert events[-1]["payload"]["source_urls"] == []
    token = var_child_runnable_config.set({"configurable": {"user": "1", "agent_id": "host"}})
    try:
        team = SimpleNamespace(resolve_peer=lambda *args, **kwargs: None)
        ask = next(tool for tool in build_team_tools(team) if tool.name == "ask_agent")
        identity = uuid4()
        callback.on_tool_start({"name": "ask_agent"}, "", run_id=identity, inputs={"mode": "background"})
        output = await ask.ainvoke({"expert": "nonexistent", "message": "Review"})
        assert "expert not found" in output
        callback.on_tool_end(output, run_id=identity)
        assert events[-1]["kind"] == "dispatch"
        assert events[-1]["status"] == "failed"
    finally:
        var_child_runnable_config.reset(token)


def test_error_shaped_file_content_is_not_a_tool_failure() -> None:
    events: list[dict[str, Any]] = []
    callback = RunEventCallback(events.append)
    identity = uuid4()
    callback.on_tool_start({"name": "read_file"}, "", run_id=identity)
    callback.on_tool_end('{"error":"an example from the document"}', run_id=identity)
    assert events[-1]["status"] == "completed"


def test_retrieved_sources_are_bounded_and_background_dispatch_is_only_accepted() -> None:
    events: list[dict[str, Any]] = []
    callback = RunEventCallback(events.append)
    fetch = uuid4()
    callback.on_tool_start(
        {"name": "web_fetch"}, "", run_id=fetch, inputs={"url": "https://example.org/report?token=private-value"}
    )
    callback.on_tool_end("private document body", run_id=fetch)
    assert events[-1]["payload"]["source_urls"] == ["https://example.org/report"]
    assert "private-value" not in str(events)
    assert "private document body" not in str(events)
    search = uuid4()
    callback.on_tool_start({"name": "tavily_search"}, "", run_id=search, inputs={"query": "research"})
    callback.on_tool_end(
        ToolMessage(
            '{"results":[{"url":"https://example.net/a","content":"private snippet"},{"url":"https://example.net/b"}]}',
            tool_call_id="search",
        ),
        run_id=search,
    )
    assert events[-1]["payload"]["source_urls"] == ["https://example.net/a", "https://example.net/b"]
    assert "private snippet" not in str(events)
    dispatch = uuid4()
    callback.on_tool_start(
        {"name": "ask_agent"}, "", run_id=dispatch, inputs={"expert": "Reviewer", "mode": "background"}
    )
    callback.on_tool_end('{"job_id":"peer-job","status":"pending"}', run_id=dispatch)
    assert events[-1]["kind"] == "dispatch"
    assert events[-1]["status"] == "accepted"


def test_tool_and_plan_events_retain_run_identity_without_output_bodies() -> None:
    events: list[dict[str, Any]] = []
    callback = RunEventCallback(events.append)
    token = var_child_runnable_config.set(
        {
            "configurable": {
                "thread_id": "thread",
                "user": "1",
                "octop_task_run_id": "host-run",
                "octop_turn_id": "turn",
                "octop_execution_context": {"context_id": "device-context"},
            }
        }
    )
    try:
        run_id = uuid4()
        callback.on_tool_start(
            {"name": "read_file"}, "secret input", run_id=run_id, inputs={"file_path": "/work/log.txt"}
        )
        callback.on_tool_end(ToolMessage("private file body", tool_call_id="tool-call", status="error"), run_id=run_id)
        assert events[0]["status"] == "started"
        assert events[1]["status"] == "failed"
        assert events[1]["run_id"] == "host-run"
        assert events[1]["context_id"] == "device-context"
        assert "private file body" not in str(events)
        assert "secret input" not in str(events)
        plan = uuid4()
        callback.on_tool_start(
            {"name": "write_todos"},
            "",
            run_id=plan,
            inputs={"todos": [{"content": "Check run", "status": "in_progress"}]},
        )
        callback.on_tool_end("ok", run_id=plan)
        assert events[-1]["kind"] == "plan"
        assert events[-1]["payload"]["todos"] == [{"content": "Check run", "status": "in_progress"}]
    finally:
        var_child_runnable_config.reset(token)


def test_observability_sinks_are_runtime_only() -> None:
    config = HarnessAgentConfig(run_event_sink=lambda event: None, request_snapshot_sink=lambda event: None)
    assert "run_event_sink" not in config.to_dict()
    assert "request_snapshot_sink" not in config.to_dict()


@pytest.mark.asyncio
async def test_public_callback_receives_real_graph_tool_run_context() -> None:
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    @tool
    def inspect_progress() -> str:
        """Read a progress observation."""
        return "private progress body"

    events: list[dict[str, Any]] = []
    graph = create_deep_agent(
        model=Model(
            responses=[
                AIMessage("", tool_calls=[{"id": "call-one", "name": "inspect_progress", "args": {}}]),
                AIMessage("done"),
            ]
        ),
        tools=[inspect_progress],
    )
    await graph.ainvoke(
        {"messages": [HumanMessage("Check progress")]},
        config={
            "configurable": {"thread_id": "thread", "user": "7", "octop_task_run_id": "host-run"},
            "callbacks": [RunEventCallback(events.append)],
        },
    )
    assert [event["status"] for event in events] == ["started", "completed"]
    assert events[-1]["thread_id"] == "thread"
    assert events[-1]["run_id"] == "host-run"
    assert events[-1]["payload"]["tool_call_id"] == "call-one"
    assert "private progress body" not in str(events)
