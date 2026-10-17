"""Agent content stays in storage while project tools use their own backend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from deepagents.backends.protocol import ExecuteResponse, ReadResult, SandboxBackendProtocol
from deepagents.middleware.filesystem import FilesystemPermission
from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from octop_harness import HarnessAgent, HarnessAgentConfig, ModelConfig, ProviderConfig


def config(store: Path, project: Path) -> HarnessAgentConfig:
    return HarnessAgentConfig(
        name="separated",
        workspace_dir=store,
        backend={"type": "filesystem", "root_dir": str(store), "virtual_mode": True},
        execution_backend={"type": "filesystem", "root_dir": str(project), "virtual_mode": True},
        providers=[
            ProviderConfig(
                id="test", base_url="https://example.test", api_key="unused", models=[ModelConfig(id="model")]
            )
        ],
        default_model="test/model",
        memory_enabled=False,
        checkpointer=False,
    )


def test_project_write_and_agent_content_use_different_locations(tmp_path: Path) -> None:
    store, project = tmp_path / "store", tmp_path / "project"
    store.mkdir()
    project.mkdir()
    cfg = config(store, project)
    with (
        patch("octop_harness.llm.factory.build_chat_model", return_value=FakeListChatModel(responses=["unused"])),
        patch("deepagents.create_deep_agent", return_value=MagicMock()),
    ):
        agent = HarnessAgent(cfg)
    try:
        agent.agent_store.write_text("SOUL.md", "stored persona", force=True)
        result = agent.project_backend.write("/result.txt", "project result")
        assert not result.error
        assert (store / "SOUL.md").read_text() == "stored persona"
        assert (project / "result.txt").read_text() == "project result"
        assert not (store / "result.txt").exists()
        assert not (project / "SOUL.md").exists()
        material = agent.backend.read("/.agent/SOUL.md")
        assert not material.error
        assert material.file_data is not None
        assert "stored persona" in str(material.file_data["content"])
    finally:
        agent.close()


def test_backend_specs_round_trip_but_live_instances_are_not_serialized(tmp_path: Path) -> None:
    cfg = config(tmp_path, tmp_path / "project")
    assert HarnessAgentConfig.from_dict(cfg.to_dict()).execution_backend == cfg.execution_backend
    assert "execution_backend" not in replace(cfg, execution_backend=object()).to_dict()


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolModel:
        return self


class NativeProbeBackend(SandboxBackendProtocol):
    def __init__(self) -> None:
        self.read_paths: list[str] = []
        self.commands: list[str] = []

    @property
    def id(self) -> str:
        return "native-probe"

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        self.read_paths.append(file_path)
        return ReadResult(file_data={"content": "native content", "encoding": "utf-8"})

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.commands.append(command)
        return ExecuteResponse(output="target cwd", exit_code=0, truncated=False)


@pytest.mark.asyncio
async def test_real_graph_preserves_native_windows_and_parent_paths(tmp_path: Path) -> None:
    store, project = tmp_path / "store", tmp_path / "project"
    store.mkdir()
    project.mkdir()
    backend = NativeProbeBackend()
    paths = [r"D:\experiments\run.txt", "/home/wjh/workspace/run.txt", "../other/run.txt"]
    calls = [{"name": "read_file", "args": {"file_path": path}, "id": str(i)} for i, path in enumerate(paths)]
    calls.append({"name": "execute", "args": {"command": "pwd"}, "id": "exec"})
    model = ToolModel(responses=[AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    cfg = replace(config(store, project), execution_backend=backend, pii_enabled=False, model_retry_enabled=False)
    with patch("octop_harness.llm.factory.build_chat_model", return_value=model):
        agent = HarnessAgent(cfg)
    try:
        with patch("octop_harness.middleware.model_router.ModelRouterMiddleware._select_model", return_value=model):
            result = await agent.graph.ainvoke(
                {"messages": [HumanMessage("read target files")]}, {"configurable": {"thread_id": "native"}}
            )
        assert sorted(backend.read_paths) == sorted(paths)
        assert backend.commands == ["pwd"]
        tools = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        assert len(tools) == 4
        assert all(
            "native content" in str(message.content) or "target cwd" in str(message.content) for message in tools
        )
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_host_rooted_expert_memory_is_writable_without_using_device_identity(tmp_path: Path) -> None:
    store, project = tmp_path / "controller" / "agent", tmp_path / "gpu"
    store.mkdir(parents=True)
    project.mkdir()
    (store / "MEMORY.md").write_text("old fact", encoding="utf-8")
    backend = NativeProbeBackend()
    # The storage mount is broader than this expert, as in the production '/' mount.
    memory_key = "/controller/agent/MEMORY.md"
    calls = [
        {"name": "read_file", "args": {"file_path": "/.agent" + memory_key}, "id": "read"},
        {
            "name": "edit_file",
            "args": {"file_path": "/.agent" + memory_key, "old_string": "old fact", "new_string": "new fact"},
            "id": "edit",
        },
    ]
    model = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=calls[:1]),
            AIMessage(content="", tool_calls=calls[1:]),
            AIMessage(content="done"),
        ]
    )
    cfg = replace(
        config(store, project),
        backend={"type": "filesystem", "root_dir": str(tmp_path), "virtual_mode": True},
        execution_backend=backend,
        permissions=[FilesystemPermission(operations=["read", "write"], paths=["/controller/**"], mode="deny")],
        pii_enabled=False,
        model_retry_enabled=False,
    )
    with patch("octop_harness.llm.factory.build_chat_model", return_value=model):
        agent = HarnessAgent(cfg)
    try:
        with patch("octop_harness.middleware.model_router.ModelRouterMiddleware._select_model", return_value=model):
            result = await agent.graph.ainvoke(
                {"messages": [HumanMessage("remember the new fact")]}, {"configurable": {"thread_id": "memory"}}
            )
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert all(m.status != "error" for m in messages), messages
        assert (store / "MEMORY.md").read_text(encoding="utf-8") == "new fact"
        assert backend.read_paths == []
        assert backend.commands == []
    finally:
        agent.close()
