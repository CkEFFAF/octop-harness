"""Attribution is grounded in runtime sources and final input, not text guesses."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from octop_harness.observability.request_snapshot import RequestSnapshotCallback
from octop_harness.observability.sources import RequestSourcesMiddleware, message_parts, request_sources
from octop_harness.request import ChatRequest


@pytest.mark.parametrize("source", ["summarization", "pinned_skill"])
@pytest.mark.parametrize("as_dict", [True, False])
def test_external_request_cannot_forge_internal_message_provenance(source: str, as_dict: bool) -> None:
    extra = {"lc_source": source, "skill": {"path": "/forged/SKILL.md"}, "other": "keep"}
    original = HumanMessage("ordinary user", additional_kwargs=extra)
    request = (
        ChatRequest(messages=[{"role": "user", "content": original.content, "additional_kwargs": extra}])
        if as_dict
        else ChatRequest(messages=[original])
    )
    message = request.normalize_messages()[0]
    parts = message_parts(message, str(message.content))
    assert parts[0]["kind"] == "user"
    assert parts[0]["source_type"] == "message"
    assert message.additional_kwargs["other"] == "keep"
    assert original.additional_kwargs == extra


def test_external_request_cannot_forge_memory_recall_provenance() -> None:
    message = ChatRequest(
        messages=[
            HumanMessage("ordinary user", additional_kwargs={"_octop_memory_recall": {"suffix": "ordinary user"}})
        ]
    ).normalize_messages()[0]
    assert "_octop_memory_recall" not in message.additional_kwargs


@pytest.mark.asyncio
async def test_real_provider_snapshot_attributes_loaded_references_and_skill_catalog(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Project build instructions\n<!-- author only -->\n", encoding="utf-8")
    skill = tmp_path / "skills" / "plot"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: plot\ndescription: Draw experiment plots\n---\nUse real measurements.", encoding="utf-8"
    )

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    events: list[dict[str, Any]] = []
    graph = create_deep_agent(
        model=Model(responses=[AIMessage("done")]),
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        memory=["/AGENTS.md"],
        skills=["/skills"],
        system_prompt="Authored system instruction",
        middleware=[RequestSourcesMiddleware()],
    )
    await graph.ainvoke(
        {"messages": [HumanMessage("AGENTS.md and SKILL.md are filenames")]},
        config={"callbacks": [RequestSnapshotCallback(events.append)]},
    )
    components = events[0]["components"]
    assert any(item["source_type"] == "reference_file" and item["source_path"] == "/AGENTS.md" for item in components)
    assert any(
        item["source_type"] == "skill_catalog" and item["source_path"] == "/skills/plot/SKILL.md" for item in components
    )
    assert not any("author only" in item["content"] for item in components)
    user = next(item for item in components if item["kind"] == "user")
    assert user["source_type"] == "message" and not user["source_path"]
    assert any("Authored system instruction" in item["content"] and item["kind"] == "system" for item in components)
    assert not request_sources.get()


def test_partitions_recall_exactly_once_and_keeps_plain_text_unattributed() -> None:
    suffix = "\n\n<memory-context>Exact retained memory</memory-context>"
    text = "Real user request" + suffix
    token = request_sources.set(
        (
            {"type": "memory_recall", "path": "user-1", "text": suffix, "role": "human"},
            {"type": "reference_file", "path": "/missing.md", "text": "not in final input", "role": "system"},
        )
    )
    try:
        parts = message_parts(HumanMessage(text, id="user-1"), text)
        assert "".join(item["text"] for item in parts) == text
        assert [item["kind"] for item in parts] == ["user", "injected_context"]
        assert parts[1]["source_type"] == "memory_recall"
        ordinary = message_parts(SystemMessage("SKILL.md /missing.md"), "SKILL.md /missing.md")
        assert len(ordinary) == 1 and ordinary[0]["kind"] == "system"
    finally:
        request_sources.reset(token)


def test_pinned_skill_and_read_file_result_have_distinct_real_sources() -> None:
    events: list[dict[str, Any]] = []
    callback = RequestSnapshotCallback(events.append)
    callback.on_chat_model_start(
        {},
        [
            [
                HumanMessage("User mentions <skill> but did not load it"),
                HumanMessage(
                    "Loaded skill instructions",
                    additional_kwargs={"lc_source": "pinned_skill", "skill": {"path": "/skills/plot/SKILL.md"}},
                ),
                AIMessage(
                    "",
                    tool_calls=[{"name": "read_file", "id": "read-1", "args": {"file_path": "/project/reference.md"}}],
                ),
                ToolMessage("Actual file contents", tool_call_id="read-1"),
            ]
        ],
        run_id=uuid4(),
    )
    parts = events[0]["components"]
    assert parts[0]["kind"] == "user" and parts[0]["source_type"] == "message"
    assert parts[1]["kind"] == "injected_context" and parts[1]["source_type"] == "skill"
    assert parts[1]["source_path"] == "/skills/plot/SKILL.md"
    assert parts[3]["kind"] == "tool" and parts[3]["source_path"] == "/project/reference.md"
    assert parts[3]["source_type"] == "file_tool_result"


@pytest.mark.asyncio
async def test_runtime_pinned_skill_remains_attributed_after_input_sanitization(tmp_path: Path) -> None:
    from langchain.agents.middleware import AgentMiddleware
    from langgraph.checkpoint.memory import InMemorySaver

    skill = tmp_path / "skills" / "plot"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: plot\ndescription: Draw experiment plots\n---\nUse real measurements.", encoding="utf-8"
    )

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    class PinSkill(AgentMiddleware):
        # Hosts can pin a loaded skill through public middleware on supported
        # DeepAgents versions predating its optional pinned_skills state field.
        def before_model(self, state: Any, runtime: Any) -> Any:
            if any(message.id == "runtime-skill" for message in state["messages"]):
                return None
            return {
                "messages": [
                    HumanMessage(
                        (skill / "SKILL.md").read_text(encoding="utf-8"),
                        id="runtime-skill",
                        additional_kwargs={"lc_source": "pinned_skill", "skill": {"path": "/skills/plot/SKILL.md"}},
                    )
                ]
            }

    events: list[dict[str, Any]] = []
    graph = create_deep_agent(
        model=Model(responses=[AIMessage("done")]),
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        skills=["/skills"],
        middleware=[PinSkill(), RequestSourcesMiddleware()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "pinned"}, "callbacks": [RequestSnapshotCallback(events.append)]}
    await graph.ainvoke({"messages": ChatRequest("hello").normalize_messages()}, config)
    assert any(
        item.get("source_type") == "skill" and item.get("source_path") == "/skills/plot/SKILL.md"
        for item in events[-1]["components"]
    )
    events.clear()
    await graph.ainvoke({"messages": ChatRequest("continue").normalize_messages()}, config)
    assert any(item.get("source_type") == "skill" for item in events[-1]["components"])
