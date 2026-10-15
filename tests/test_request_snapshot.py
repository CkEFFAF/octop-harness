"""Provider-bound snapshots retain the final request and disjoint real usage."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from octop_harness.observability.request_snapshot import RequestSnapshotCallback


@pytest.mark.parametrize("secret", ["13800138000", "11010519491231002X"])
def test_snapshot_redacts_pii_crossing_content_boundary(secret: str) -> None:
    events: list[dict[str, Any]] = []
    text = " " * (16_000 - len(secret) + 1) + secret + " trailing" * 100
    RequestSnapshotCallback(events.append).on_chat_model_start({}, [[HumanMessage(text)]], run_id=uuid4())
    content = events[0]["components"][0]["content"]
    assert secret[:-1] not in content
    assert "[redacted]" in content
    assert len(content) <= 16_000
    assert events[0]["components"][0]["content_truncated"]


def test_snapshot_bounds_and_redacts_all_identifiers_and_metadata() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.observability.compression import request_transformation

    secret = "sk-" + "z" * 40
    long_reference = "/" + secret + "/" + "p" * 20_000
    events: list[dict[str, Any]] = []
    callback = RequestSnapshotCallback(events.append)
    token = request_transformation.set({"removed_source_ids": [long_reference], "kind": "message_rewrite"})
    request_id = uuid4()
    try:
        callback.on_chat_model_start(
            {},
            [
                [
                    AIMessage(
                        "",
                        id=long_reference,
                        tool_calls=[{"name": "read_file", "id": long_reference, "args": {"file_path": long_reference}}],
                    ),
                    ToolMessage("file", id=long_reference, tool_call_id=long_reference),
                ]
            ],
            run_id=request_id,
            metadata={
                key: long_reference
                for key in ("thread_id", "agent_id", "user", "octop_turn_id", "octop_task_run_id", "model")
            }
            | {"octop_execution_context": {"context_id": long_reference}},
        )
    finally:
        request_transformation.reset(token)
    callback.on_llm_error(type(secret, (Exception,), {})(), run_id=request_id)
    for event in events:
        assert secret not in str(event)
        assert event["metadata_truncated"]
        for component in event["components"]:
            assert len(component.get("source_id") or "") <= 512
            assert len(component.get("tool_call_id") or "") <= 512
            assert len(component.get("source_path") or "") <= 4096


def test_snapshot_component_cap_preserves_full_composition_and_estimates() -> None:
    events: list[dict[str, Any]] = []
    messages = [HumanMessage("") for _ in range(5000)] + [HumanMessage("later text" * 100)]
    RequestSnapshotCallback(events.append).on_chat_model_start(
        {}, [messages], run_id=uuid4(), invocation_params={"tools": [{"name": "later schema"}]}
    )
    event = events[0]
    assert len(event["components"]) <= 256
    assert event["component_count"] == 5002
    assert event["components_truncated"]
    assert event["omitted_component_count"] == 5002 - len(event["components"])
    assert event["component_counts"] == {"user": 5001, "tool_schema": 1}
    assert event["estimated_input_tokens"] > sum(item["estimated_tokens"] for item in event["components"])
    assert sum(event["estimated_tokens_by_kind"].values()) == event["estimated_input_tokens"]


def test_compression_receipt_identifies_removed_duplicate_messages() -> None:
    from octop_harness.observability.compression import _receipt

    first, second = HumanMessage("same", id="first"), HumanMessage("same", id="second")
    receipt = _receipt([first, second], [second])
    assert receipt is not None
    assert receipt["removed_source_ids"] == ["first"]
    receipt = _receipt([first, second], [HumanMessage("same", id="replacement")])
    assert receipt is not None
    assert receipt["removed_source_ids"] == ["first", "second"]
    receipt = _receipt([HumanMessage("same"), HumanMessage("same")], [HumanMessage("same")])
    assert receipt is not None
    assert receipt["removed_source_ids"] == []


def test_snapshot_lifecycle_preserves_missing_usage_and_redacts_media_keys() -> None:
    events: list[dict[str, Any]] = []
    callback = RequestSnapshotCallback(events.append)
    request_id = uuid4()
    callback.on_chat_model_start(
        {},
        [
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "sk-" + "a" * 30},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}},
                    ]
                )
            ]
        ],
        run_id=request_id,
        metadata={"thread_id": "thread", "user": "1"},
    )
    callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage("done"))]]), run_id=request_id)
    assert events[0]["status"] == "started"
    assert events[1]["status"] == "completed"
    assert events[1]["actual_usage"] is None
    assert events[1]["ttft_ms"] is None
    assert events[1]["duration_ms"] >= 0
    content = events[0]["components"][0]["content"]
    assert "[redacted]" in content
    assert "PRIVATE" not in content
    assert "a" * 30 not in content
    assert events[0]["components"][0]["content_truncated"] is False


def test_snapshot_includes_tool_calls_and_current_schema_contents() -> None:
    events: list[dict[str, Any]] = []
    callback = RequestSnapshotCallback(events.append)
    msg = AIMessage("", tool_calls=[{"name": "inspect", "args": {"path": "/data/experiment"}, "id": "call-1"}])
    for description in ["short", "large schema " * 100]:
        callback.on_chat_model_start(
            {}, [[msg]], run_id=uuid4(), invocation_params={"tools": [{"name": "inspect", "description": description}]}
        )
    assert "experiment" in events[0]["components"][0]["content"]
    assert events[0]["components"][0]["estimated_tokens"] > 0
    assert events[1]["components"][1]["estimated_tokens"] > events[0]["components"][1]["estimated_tokens"]


def test_many_short_messages_keep_their_contents_and_later_tool_schemas() -> None:
    events: list[dict[str, Any]] = []
    callback = RequestSnapshotCallback(events.append)
    callback.on_chat_model_start(
        {},
        [[HumanMessage(f"message {index}") for index in range(20)]],
        run_id=uuid4(),
        invocation_params={"tools": [{"name": "inspect", "description": "small schema"}]},
    )
    components = events[0]["components"]
    assert [item["content"] for item in components[:20]] == [f"message {index}" for index in range(20)]
    assert "small schema" in components[-1]["content"]
    assert not any(item["content_truncated"] for item in components)


def test_snapshot_stream_usage_and_failure() -> None:
    events: list[dict[str, Any]] = []
    callback = RequestSnapshotCallback(events.append)
    request_id = uuid4()
    callback.on_chat_model_start(
        {}, [[SystemMessage("final injected material"), HumanMessage("hello")]], run_id=request_id
    )
    callback.on_llm_new_token("a", run_id=request_id)
    message = AIMessage(
        "done",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 5,
            "total_tokens": 105,
            "input_token_details": {"cache_read": 90},
        },
    )
    callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=request_id)
    actual = events[-1]["actual_usage"]
    assert actual["input_tokens"] == 100
    assert actual["cache_read_tokens"] == 90
    assert actual["uncached_input_tokens"] == 10
    assert actual["total_tokens"] == 105
    assert events[-1]["ttft_ms"] >= 0
    assert events[-1]["components"][0]["content"] == "final injected material"
    failed = uuid4()
    callback.on_chat_model_start({}, [[HumanMessage("fail")]], run_id=failed)
    callback.on_llm_error(ValueError("secret error contents"), run_id=failed)
    assert events[-1]["status"] == "failed"
    assert events[-1]["error_type"] == "ValueError"
    assert "secret error contents" not in str(events[-1])


@pytest.mark.asyncio
async def test_snapshot_callback_in_real_graph() -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    events: list[dict[str, Any]] = []
    graph = create_deep_agent(model=Model(responses=[AIMessage("done")]), system_prompt="actual system material")
    await graph.ainvoke(
        {"messages": [HumanMessage("hello")]},
        config={
            "configurable": {"thread_id": "thread", "agent_id": "expert", "user": "7"},
            "callbacks": [RequestSnapshotCallback(events.append)],
        },
    )
    assert [event["status"] for event in events] == ["started", "completed"]
    assert events[0]["thread_id"] == "thread"
    assert events[0]["owner_user_id"] == "7"
    assert "actual system material" in events[0]["components"][0]["content"]


@pytest.mark.asyncio
async def test_real_summarization_receipt_and_reused_summary_are_distinct(tmp_path: Path) -> None:
    from deepagents.backends import FilesystemBackend
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    from octop_harness.observability.compression import ObservedSummarizationMiddleware

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    model = Model(responses=[AIMessage("small summary"), AIMessage("done")])
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    summarizer = ObservedSummarizationMiddleware(
        model=model, backend=backend, trigger=("messages", 8), keep=("messages", 2)
    )
    graph = create_deep_agent(model=model, backend=backend, middleware=[summarizer], checkpointer=InMemorySaver())
    events: list[dict[str, Any]] = []
    config = {
        "configurable": {"thread_id": "compressed", "agent_id": "expert", "user": "7"},
        "callbacks": [RequestSnapshotCallback(events.append)],
    }
    history = []
    for index in range(6):
        history.extend([HumanMessage((f"input {index} " * 40), id=f"u{index}"), AIMessage("answer", id=f"a{index}")])
    await graph.ainvoke({"messages": history}, config=config)
    compressed = [event for event in events if event.get("transformation")]
    assert compressed
    receipt = compressed[-1]["transformation"]
    assert receipt["kind"] == "summarization"
    assert receipt["before_estimated_tokens"] > receipt["after_estimated_tokens"]
    assert receipt["before_message_count"] == 12
    assert receipt["after_message_count"] < 12
    assert "u0" in receipt["removed_source_ids"]
    assert any(c["source_type"] == "conversation_summary" for c in compressed[-1]["components"])
    state = await graph.aget_state(config)
    assert len(state.values["messages"]) == 13  # Raw transcript remains available.
    events.clear()
    await graph.ainvoke({"messages": [HumanMessage("continue")]}, config=config)
    assert events
    assert all(event.get("transformation") is None for event in events)
    assert any(c["kind"] == "injected_context" for c in events[-1]["components"])
    # A different thread cannot inherit the context-local transform.
    events.clear()
    config["configurable"]["thread_id"] = "other"
    await graph.ainvoke({"messages": [HumanMessage("hello")]}, config=config)
    assert all(event.get("transformation") is None for event in events)
    assert not any(c["kind"] == "injected_context" for c in events[-1]["components"])
