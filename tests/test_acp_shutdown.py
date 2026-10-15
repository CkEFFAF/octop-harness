from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from octop_harness.acp import service
from octop_harness.acp.models import ACPConfig


def test_shutdown_closes_registered_service_after_default_loop_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = []

    class FakeService:
        async def close_all_sessions(self) -> None:
            closed.append(True)

    asyncio.run(asyncio.sleep(0))
    monkeypatch.setattr(service, "_services", {"test": FakeService()})
    service._shutdown_services()
    assert closed == [True]
    assert not service._services


@pytest.mark.asyncio
async def test_close_all_cancels_active_prompts_and_closes_every_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = service.ACPService(config=ACPConfig())
    killed: list[int] = []
    closed: list[int] = []
    monkeypatch.setattr(service, "_kill_process_tree", killed.append)
    conversations = []
    for index in range(2):
        stack = AsyncExitStack()
        stack.callback(closed.append, index)
        conversation = service._Conversation(
            thread_id=str(index),
            runner="test",
            acp_session_id=str(index),
            cwd=".",
            conn=SimpleNamespace(close_session=AsyncMock()),
            process=SimpleNamespace(pid=index),
            client=SimpleNamespace(),
            exit_stack=stack,
            turn_lock=asyncio.Lock(),
            prompt_task=asyncio.create_task(asyncio.Event().wait()),
        )
        conversations.append(conversation)
        instance._sessions[(str(index), "test")] = conversation
    await asyncio.sleep(0)
    await instance.close_all_sessions()
    assert not instance._sessions
    assert sorted(killed) == sorted(closed) == [0, 1]
    for conversation in conversations:
        assert conversation.prompt_task is not None and conversation.prompt_task.cancelled()
        conversation.conn.close_session.assert_awaited_once_with(session_id=conversation.acp_session_id)


@pytest.mark.asyncio
async def test_close_all_continues_after_one_session_cleanup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = service.ACPService(config=ACPConfig())
    instance._sessions = {("first", "test"): "first", ("second", "test"): "second"}  # type: ignore[dict-item]
    seen = []

    async def close(conversation: object) -> None:
        seen.append(conversation)
        if conversation == "first":
            raise RuntimeError("broken transport cleanup")

    monkeypatch.setattr(instance, "_close_conversation", close)
    await instance.close_all_sessions()
    assert seen == ["first", "second"]
    assert not instance._sessions
