"""Run the real graph with a memory-only native-path execution backend."""

from __future__ import annotations

from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends.protocol import ExecuteResponse, ReadResult, SandboxBackendProtocol
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from octop_harness.backends.execution import StorageExecutionBackend
from octop_harness.middleware.native_filesystem import NativeFilesystemMiddleware


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Model:
        return self


class NativeBackend(SandboxBackendProtocol):
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.commands: list[str] = []

    @property
    def id(self) -> str:
        return "memory-native"

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        self.paths.append(file_path)
        return ReadResult(file_data={"content": "native file", "encoding": "utf-8"})

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.commands.append(command)
        return ExecuteResponse(output="native command", exit_code=0, truncated=False)


@pytest.mark.asyncio
async def test_native_tools_in_full_deepagent_graph() -> None:
    paths = [r"D:\experiment\log.txt", "/home/wjh/workspace/log.txt", "../outside/log.txt"]
    calls = [{"name": "read_file", "args": {"file_path": path}, "id": str(i)} for i, path in enumerate(paths)]
    calls.append({"name": "execute", "args": {"command": "pwd"}, "id": "exec"})
    model = Model(responses=[AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    project = NativeBackend()
    storage = NativeBackend()
    composite = StorageExecutionBackend(project, storage)
    graph = create_deep_agent(
        model=model,
        backend=composite,
        middleware=[NativeFilesystemMiddleware(composite)],
    )
    result = await graph.ainvoke({"messages": [HumanMessage("inspect execution target")]})
    assert sorted(project.paths) == sorted(paths)
    assert project.commands == ["pwd"]
    assert storage.paths == []
    messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(messages) == 4
    assert all("native" in str(m.content) for m in messages)


@pytest.mark.asyncio
async def test_native_tool_errors_are_visible_as_failed_tool_messages() -> None:
    class MissingBackend(NativeBackend):
        async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
            return ReadResult(error="OS permission denied")

    model = Model(
        responses=[
            AIMessage("", tool_calls=[{"name": "read_file", "args": {"file_path": "/private"}, "id": "denied"}]),
            AIMessage("done"),
        ]
    )
    backend = MissingBackend()
    graph = create_deep_agent(model=model, backend=backend, middleware=[NativeFilesystemMiddleware(backend)])
    result = await graph.ainvoke({"messages": [HumanMessage("inspect")]})
    tool_message = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert tool_message.status == "error"
    assert "OS permission denied" in tool_message.content
