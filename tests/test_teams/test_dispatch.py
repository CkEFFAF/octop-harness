"""Dispatch contracts exercised through both real tool construction paths."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from octop_harness.config import HarnessAgentConfig
from octop_harness.middleware.peer import PeerAgentMiddleware
from octop_harness.registry import AgentEntry, AgentRegistry
from octop_harness.teams import DispatchOptions, DispatchReceipt, PeerCall, PeerSession, TeamManager


class Processor:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def compose_followup(self, msg: Any, *, result_text: str | None, error_text: str | None) -> str:
        return result_text or error_text or "empty"

    async def on_reply(self, event: Any) -> None:
        self.events.append(event)


def team_with_child(tmp_path: Any, *, allowlist: tuple[str, ...] | None = None) -> tuple[TeamManager, Any, Processor]:
    registry = AgentRegistry()
    child = MagicMock()
    child.call = AsyncMock(return_value={"messages": [{"role": "assistant", "content": "blocked in prose"}]})
    for name in ("parent", "child"):
        registry.add(
            AgentEntry(
                agent_id=name,
                agent=child,
                config=HarnessAgentConfig(
                    name=name, workspace_dir=tmp_path / name, team_peers=allowlist if name == "parent" else None
                ),
                metadata={"user_id": 7},
                tags=[],
                created_at=datetime.now(tz=UTC),
            )
        )
    team = TeamManager(registry)
    processor = Processor()
    team.set_processor(processor)
    return team, child, processor


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["middleware", "manager"])
@pytest.mark.parametrize("mode", ["sync", "async", "both"])
@pytest.mark.parametrize("policy", ["inherit", "member_default", "explicit"])
async def test_real_tool_dispatch_options_and_receipt(tmp_path: Any, builder: str, mode: str, policy: str) -> None:
    team, child, _ = team_with_child(tmp_path)
    seen: list[tuple[PeerCall, DispatchReceipt]] = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.append((call, receipt))

    async def prepare(call: PeerCall) -> PeerSession:
        assert seen[0][1].transport_status == "accepted"
        assert call.dispatch_options.resource_ids == ("res_one",)
        assert call.dispatch_options.target_policy == policy
        assert call.invocation_id and call.invocation_id != call.job_id
        return PeerSession(
            receipt=DispatchReceipt(
                invocation_id=call.invocation_id,
                target_policy=call.dispatch_options.target_policy,
                requested_target=call.dispatch_options.requested_target(),
                resolved_target={"device_id": "actual-device"},
            )
        )

    team.bind_peer_session(prepare=prepare, lifecycle=lifecycle)
    tools = (
        PeerAgentMiddleware(team, peer_invoke_mode=mode).tools
        if builder == "middleware"
        else team.team_tools(peer_invoke_mode=mode)
    )
    tool = next(tool for tool in tools if tool.name == "ask_agent")
    values: dict[str, Any] = {
        "expert": "child",
        "message": "task",
        "target_policy": policy,
        "resource_ids": ["res_one"],
    }
    if policy == "explicit":
        values.update(device_id="requested-device")
    with patch(
        "octop_harness.teams.tools.get_config",
        return_value={"configurable": {"agent_id": "parent", "user": 7, "thread_id": "root-thread"}},
    ):
        result = json.loads(await tool.ainvoke(values))
    await team.aclose()
    assert result["invocation_id"] == seen[0][0].invocation_id
    assert [receipt.transport_status for _, receipt in seen] == ["accepted", "running", "completed"]
    assert seen[-1][1].resolved_target == {"device_id": "actual-device"}
    assert seen[-1][1].task_outcome == "unknown"
    if mode == "async":
        assert result["dispatch_receipt"]["transport_status"] == "accepted"
        assert result["job_id"] != result["invocation_id"]
    else:
        assert "job_id" not in result
        assert result["response"] == "blocked in prose"
        assert result["thread_id"].endswith(result["invocation_id"])
        assert result["dispatch_receipt"]["resolved_target"]["device_id"] == "actual-device"
    assert child.call.await_count == (2 if mode == "async" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "both"])
@pytest.mark.parametrize(
    "bad",
    [
        {"mystery": "lost"},
        {"invocation_id": "forged"},
        {"device_id": "unqualified"},
        {"target_policy": "explicit"},
        {"target_policy": "explicit", "target_kind": "local", "device_id": "device"},
        {"resource_ids": ["/private/file"]},
        {"resource_ids": ["same", "same"]},
        {"resource_ids": ["x"] * 33},
        {"resource_ids": [2]},
        {"resource_ids": [""]},
    ],
)
async def test_strict_actual_schemas_reject_before_call(tmp_path: Any, mode: str, bad: dict[str, Any]) -> None:
    team, child, _ = team_with_child(tmp_path)
    tool = next(tool for tool in PeerAgentMiddleware(team, peer_invoke_mode=mode).tools if tool.name == "ask_agent")
    with pytest.raises(ValidationError):
        await tool.ainvoke({"expert": "child", "message": "task", **bad})
    child.call.assert_not_awaited()
    await team.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["accepted", "prepare", "running"])
async def test_authorization_and_persistence_fail_closed(tmp_path: Any, failure: str) -> None:
    team, child, _ = team_with_child(tmp_path)
    seen = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.append(receipt.transport_status)
        if receipt.transport_status == failure:
            raise RuntimeError("do not execute")

    async def prepare(call: PeerCall) -> None:
        if failure == "prepare":
            raise PermissionError("resource not authorized")

    team.bind_peer_session(prepare=prepare, lifecycle=lifecycle)
    with pytest.raises(RuntimeError):
        await team.call_peer(from_agent_id="parent", to_agent_id="child", message="task", user_id=7)
    child.call.assert_not_awaited()
    assert seen[-1] == "failed"
    await team.aclose()


@pytest.mark.asyncio
async def test_terminal_persistence_failure_preserves_response_as_unknown(tmp_path: Any) -> None:
    team, _child, _ = team_with_child(tmp_path)

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        if receipt.transport_status == "completed":
            raise RuntimeError("commit failed")

    team.bind_peer_session(lifecycle=lifecycle)
    result = await team.call_peer(from_agent_id="parent", to_agent_id="child", message="task", user_id=7)
    assert result.response == "blocked in prose"
    assert result.receipt.transport_status == "unknown"
    assert result.receipt.error_code == "PEER_RECEIPT_PERSISTENCE_FAILED"
    await team.aclose()


@pytest.mark.asyncio
async def test_task_outcome_only_from_typed_result(tmp_path: Any) -> None:
    team, child, _ = team_with_child(tmp_path)
    child.call.return_value["dispatch_receipt"] = {
        "task_outcome": "blocked",
        "error_code": "DEVICE_OFFLINE",
        "retryable": True,
    }
    result = await team.call_peer(from_agent_id="parent", to_agent_id="child", message="task", user_id=7)
    assert result.receipt.transport_status == "completed"
    assert result.receipt.task_outcome == "blocked"
    assert result.receipt.retryable is True
    await team.aclose()


@pytest.mark.asyncio
async def test_background_acceptance_is_awaited_and_cancellation_retained(tmp_path: Any) -> None:
    team, child, processor = team_with_child(tmp_path)
    entered = asyncio.Event()
    seen: dict[str, list[str]] = {}

    async def invoke(request: Any) -> Any:
        entered.set()
        await asyncio.Event().wait()

    child.call.side_effect = invoke

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.setdefault(call.invocation_id, []).append(receipt.transport_status)

    team.bind_peer_session(lifecycle=lifecycle)
    first = await team.asubmit_peer(
        from_agent_id="parent", to_agent_id="child", message="one", user_id=7, source_thread_id="root"
    )
    second = await team.asubmit_peer(
        from_agent_id="parent", to_agent_id="child", message="two", user_id=7, source_thread_id="root"
    )
    assert seen[first.invocation_id] == ["accepted"]
    assert seen[second.invocation_id] == ["accepted"]
    await entered.wait()
    assert team.inbox.cancel(first.job_id)
    assert team.inbox.cancel(second.job_id)
    await team.aclose()
    assert seen[first.invocation_id] == ["accepted", "running", "unknown"]
    assert seen[second.invocation_id] == ["accepted", "cancelled"]
    assert {event.status for event in processor.events} == {"cancelled"}


@pytest.mark.asyncio
async def test_direct_call_cannot_bypass_allowlist(tmp_path: Any) -> None:
    team, child, _ = team_with_child(tmp_path, allowlist=())
    with pytest.raises(ValueError, match="allowed peer"):
        await team.call_peer(from_agent_id="parent", to_agent_id="child", message="task", user_id=7)
    with pytest.raises(ValueError, match="allowed peer"):
        await team.asubmit_peer(
            from_agent_id="parent", to_agent_id="child", message="task", user_id=7, source_thread_id="root"
        )
    child.call.assert_not_awaited()
    await team.aclose()


def test_default_policy_and_explicit_local() -> None:
    assert DispatchOptions().target_policy == "member_default"
    assert DispatchOptions(target_policy="explicit", target_kind="local").requested_target()["target_kind"] == "local"


def test_native_roster_default_metadata_requires_authenticated_owner() -> None:
    from octop_harness.teams.tools import _agent_list_item

    team = MagicMock()
    team.peer_display_name.return_value = "shared"
    entry = MagicMock()
    entry.agent_id = "shared-expert"
    entry.metadata = {
        "octop_owner_user_id": 8,
        "default_target": {"device_id": "private-device"},
        "model_ref": "private-model",
    }
    for viewer in (None, 7, 8):
        with patch("octop_harness.teams.tools.get_config", return_value={"configurable": {"user": viewer}}):
            public = _agent_list_item(team, entry, "en")
        assert public["default_target"] == ({"device_id": "private-device"} if viewer == 8 else None)
        assert public["model_ref"] == ("private-model" if viewer == 8 else None)


@pytest.mark.asyncio
async def test_sync_failure_and_cancellation_are_observed(tmp_path: Any) -> None:
    team, child, _ = team_with_child(tmp_path)
    seen = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.append(receipt.transport_status)

    team.bind_peer_session(lifecycle=lifecycle)
    child.call.side_effect = TimeoutError("response lost")
    with pytest.raises(RuntimeError):
        await team.call_peer(from_agent_id="parent", to_agent_id="child", message="task", user_id=7)
    assert seen == ["accepted", "running", "unknown"]
    seen.clear()
    child.call.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await team.call_peer(from_agent_id="parent", to_agent_id="child", message="task", user_id=7)
    assert seen == ["accepted", "running", "unknown"]
    await team.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_actual_toolnode_runtime_identity_replay_and_conflict(tmp_path: Any, mode: str) -> None:
    team, child, _ = team_with_child(tmp_path)
    tool = next(tool for tool in PeerAgentMiddleware(team, peer_invoke_mode=mode).tools if tool.name == "ask_agent")
    assert "runtime" not in tool.tool_call_schema.model_fields
    assert "invocation_key" not in tool.tool_call_schema.model_fields
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([tool], handle_tool_errors=False))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    node = graph.compile()
    config = {"configurable": {"agent_id": "parent", "user": 7, "thread_id": "root"}}

    async def invoke(message: str, key: str = "provider-tool-call") -> Any:
        result = await node.ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ask_agent",
                                "args": {"expert": "child", "message": message},
                                "id": key,
                            }
                        ],
                    )
                ]
            },
            config=config,
        )
        return json.loads(result["messages"][-1].content)

    first = await invoke("task")
    replay = await invoke("task")
    assert replay["invocation_id"] == first["invocation_id"]
    conflict = await invoke("changed task")
    assert conflict["dispatch_receipt"]["error_code"] == "INVOCATION_CONFLICT"
    assert child.call.await_count <= (2 if mode == "async" else 1)
    next_call = await invoke("task", "next-provider-call")
    assert next_call["invocation_id"] != first["invocation_id"]
    await team.aclose()
    assert child.call.await_count == (4 if mode == "async" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_durable_claim_replay_returns_link_without_executing(tmp_path: Any, background: bool) -> None:
    team, child, _ = team_with_child(tmp_path)

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> DispatchReceipt:
        assert receipt.transport_status == "accepted"
        return DispatchReceipt(
            invocation_id=call.invocation_id,
            transport_status="completed",
            task_outcome="succeeded",
            child_thread_id="persisted-child",
            source_thread_id="root",
            root_run_id="persisted-root",
        )

    team.bind_peer_session(lifecycle=lifecycle)
    kwargs = dict(
        from_agent_id="parent",
        to_agent_id="child",
        message="task",
        user_id=7,
        source_thread_id="root",
        invocation_key="same-runtime-call",
    )
    result = await team.asubmit_peer(**kwargs) if background else await team.call_peer(**kwargs)
    assert result.status == "completed"
    assert result.thread_id == "persisted-child"
    assert result.receipt.task_outcome == "succeeded"
    child.call.assert_not_awaited()
    await team.aclose()


@pytest.mark.asyncio
async def test_concurrent_same_key_only_executes_once_and_retry_links_new_attempt(tmp_path: Any) -> None:
    team, child, _ = team_with_child(tmp_path)
    gate = asyncio.Event()
    started = asyncio.Event()
    seen = []

    async def invoke(request: Any) -> Any:
        started.set()
        await gate.wait()
        return {"messages": [{"role": "assistant", "content": "result"}]}

    async def prepare(call: PeerCall) -> None:
        seen.append(call)

    team.bind_peer_session(prepare=prepare)
    child.call.side_effect = invoke
    kwargs = dict(
        from_agent_id="parent",
        to_agent_id="child",
        message="task",
        user_id=7,
        source_thread_id="root",
        invocation_key="runtime-key",
    )
    first = asyncio.create_task(team.call_peer(**kwargs))
    await started.wait()
    second = asyncio.create_task(team.call_peer(**kwargs))
    await asyncio.sleep(0)
    gate.set()
    one, two = await asyncio.gather(first, second)
    assert one.invocation_id == two.invocation_id
    assert one.response == two.response == "result"
    assert child.call.await_count == 1
    retry = await team.call_peer(**{**kwargs, "invocation_key": "retry-key", "retry_of": one.invocation_id})
    assert retry.invocation_id != one.invocation_id
    assert seen[-1].retry_of == one.invocation_id
    await team.aclose()


@pytest.mark.asyncio
async def test_cancel_worker_before_dispatch_retains_accepted_attempt(tmp_path: Any) -> None:
    team, child, processor = team_with_child(tmp_path)
    seen = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.append(receipt.transport_status)

    team.bind_peer_session(lifecycle=lifecycle)
    result = await team.asubmit_peer(
        from_agent_id="parent", to_agent_id="child", message="task", user_id=7, source_thread_id="root"
    )
    team.close()
    await team.aclose()
    assert seen == ["accepted", "cancelled"]
    assert processor.events[0].status == "cancelled"
    assert result.receipt.transport_status == "cancelled"
    child.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_rewritten_prompt_only_reaches_history_hook_not_lifecycle_fingerprint(tmp_path: Any) -> None:
    team, child, _ = team_with_child(tmp_path)
    lifecycle_messages = []
    history_messages = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        lifecycle_messages.append(call.message)
        assert call.message == "immutable original assignment"

    async def prepare(call: PeerCall) -> PeerSession:
        return PeerSession(message=f"host supplied context\n{call.message}")

    async def after(call: PeerCall, thread_id: str, payload: Any) -> None:
        history_messages.append(call.message)

    team.bind_peer_session(prepare=prepare, after=after, lifecycle=lifecycle)
    result = await team.call_peer(
        from_agent_id="parent", to_agent_id="child", user_id=7, message="immutable original assignment"
    )
    assert result.receipt.transport_status == "completed"
    assert lifecycle_messages == ["immutable original assignment"] * 3
    assert history_messages == ["host supplied context\nimmutable original assignment"]
    assert child.call.await_args.args[0].messages == history_messages[0]
    await team.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("durable_host", [False, True])
async def test_completed_result_eviction_never_reexecutes_old_key(tmp_path: Any, durable_host: bool) -> None:
    team, child, _ = team_with_child(tmp_path)
    durable = {}

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> DispatchReceipt | None:
        if receipt.transport_status == "accepted" and call.invocation_id in durable:
            return durable[call.invocation_id]
        durable[call.invocation_id] = receipt
        return None

    if durable_host:
        team.bind_peer_session(lifecycle=lifecycle)
    kwargs = dict(from_agent_id="parent", to_agent_id="child", message="task", user_id=7, source_thread_id="root")
    first = await team.call_peer(**kwargs, invocation_key="0")
    assert first.response == "blocked in prose"
    for key in range(1, 258):
        await team.call_peer(**kwargs, invocation_key=str(key))
    replay = await team.call_peer(**kwargs, invocation_key="0")
    assert replay.invocation_id == first.invocation_id
    assert replay.receipt.transport_status == "completed"
    assert replay.thread_id == first.thread_id
    assert not replay.response
    assert child.call.await_count == 258
    if durable_host:
        assert len(team._dispatch_claims) <= 256
    else:
        full_results = [future.result() for _, future in team._dispatch_claims.values() if future.done()]
        assert sum(result.response is not None for result in full_results) == 256
    await team.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_cancel_created_task_before_coroutine_starts_is_persisted(tmp_path: Any, shutdown: bool) -> None:
    team, child, processor = team_with_child(tmp_path)
    seen = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.append(receipt.transport_status)

    team.bind_peer_session(lifecycle=lifecycle)
    result = await team.asubmit_peer(
        from_agent_id="parent", to_agent_id="child", message="task", user_id=7, source_thread_id="root"
    )
    # Let the dispatcher create its task, then cancel before the newly-created
    # coroutine gets its first timeslice. CancelledError cannot enter it yet.
    await asyncio.sleep(0)
    assert result.job_id in team.inbox._tasks
    assert child.call.await_count == 0
    if shutdown:
        team.close()
    else:
        assert team.inbox.cancel(result.job_id)
    await team.aclose()
    assert seen == ["accepted", "cancelled"]
    assert len(processor.events) == 1
    assert processor.events[0].status == "cancelled"
    child.call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_public_cancel_persists_request_then_keeps_entered_child_unknown(tmp_path: Any, background: bool) -> None:
    team, child, _ = team_with_child(tmp_path)
    entered = asyncio.Event()
    seen = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.append((call.invocation_id, receipt.transport_status, receipt.error_code))

    async def invoke(request: Any) -> Any:
        entered.set()
        await asyncio.Event().wait()

    team.bind_peer_session(lifecycle=lifecycle)
    child.call.side_effect = invoke
    kwargs = dict(from_agent_id="parent", to_agent_id="child", message="task", user_id=7, source_thread_id="root")
    if background:
        peer = await team.asubmit_peer(**kwargs)
        invocation_id = peer.invocation_id
        pending = None
    else:
        pending = asyncio.create_task(team.call_peer(**kwargs))
    await entered.wait()
    if not background:
        invocation_id = seen[0][0]
    receipt = await team.acancel_peer(invocation_id)
    assert receipt.transport_status == "unknown"
    assert receipt.error_code == "PEER_CANCEL_REQUESTED"
    assert [(state, code) for _, state, code in seen] == [
        ("accepted", None),
        ("running", None),
        ("running", "PEER_CANCEL_REQUESTED"),
        ("unknown", "PEER_CANCEL_REQUESTED"),
    ]
    if pending is not None:
        assert pending.cancelled()
    await team.aclose()


@pytest.mark.asyncio
async def test_public_cancel_request_failure_leaves_execution_running(tmp_path: Any) -> None:
    team, child, _ = team_with_child(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.append(call.invocation_id)
        if receipt.error_code == "PEER_CANCEL_REQUESTED":
            raise RuntimeError("cancellation request not committed")

    async def invoke(request: Any) -> Any:
        entered.set()
        await release.wait()
        return {"messages": [{"role": "assistant", "content": "completed"}]}

    team.bind_peer_session(lifecycle=lifecycle)
    child.call.side_effect = invoke
    pending = asyncio.create_task(
        team.call_peer(from_agent_id="parent", to_agent_id="child", message="task", user_id=7)
    )
    await entered.wait()
    with pytest.raises(RuntimeError, match="not committed"):
        await team.acancel_peer(seen[0])
    assert not pending.done()
    release.set()
    result = await pending
    assert result.receipt.transport_status == "completed"
    await team.aclose()


@pytest.mark.asyncio
async def test_public_late_cancel_observation_cannot_regress_completed_result(tmp_path: Any) -> None:
    team, child, _ = team_with_child(tmp_path)
    seen = []

    async def lifecycle(call: PeerCall, receipt: DispatchReceipt) -> None:
        seen.append((receipt.transport_status, receipt.error_code))

    team.bind_peer_session(lifecycle=lifecycle)
    result = await team.call_peer(from_agent_id="parent", to_agent_id="child", message="task", user_id=7)
    late = await team.acancel_peer(result.invocation_id)
    assert late.transport_status == "completed"
    assert seen[-1] == ("completed", "PEER_CANCEL_REQUESTED")
    assert result.receipt.error_code is None
    assert child.call.await_count == 1
    await team.aclose()
