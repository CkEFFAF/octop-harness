"""Peer tools: ``agent_list`` and ``ask_agent`` bound to a :class:`~octop_harness.teams.team_manager.TeamManager`.

Mounted by :class:`~octop_harness.middleware.peer.PeerAgentMiddleware` when
``HarnessAgentConfig.team_enabled`` is True and the agent is created through
:class:`~octop_harness.manager.HarnessAgentManager`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.config import get_config
from langgraph.prebuilt import ToolRuntime
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from octop_harness.backends.utils import DEFENSIVE_OP_ERRORS
from octop_harness.teams.dispatch import DispatchOptions, PeerDispatchError, TargetPolicy
from octop_harness.teams.profile import peer_description, peer_guidance_cards
from octop_harness.teams.util import (
    PeerInvokeMode,
    clamp_peer_invoke_mode,
    coerce_user_id,
    parse_peer_invoke_mode,
)

if TYPE_CHECKING:
    from octop_harness.registry import AgentEntry
    from octop_harness.teams.inbox import PeerResult
    from octop_harness.teams.profile import Language
    from octop_harness.teams.team_manager import TeamManager


class _DispatchInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", strict=True)
    # An Any payload avoids exposing ToolRuntime's stores/callbacks as JSON
    # schema. Both markers keep this field hidden and injected by ToolNode.
    runtime: Annotated[Any, InjectedToolArg, ToolRuntime] = None
    target_policy: TargetPolicy = Field(
        default="member_default",
        description=(
            "member_default: use the expert's configured device (default). "
            "inherit: explicitly use the caller's current execution device. "
            "explicit: request device_id or target_kind=local for this invocation. "
            "The host authorizes the request; inspect resolved_target in the receipt for the actual device."
        ),
    )
    device_id: str | None = Field(default=None, description="Requested device ID when target_policy=explicit")
    target_kind: Literal["device", "local"] | None = Field(default=None, description="Explicit device or local target")
    working_dir: str | None = Field(default=None, description="Requested working directory for an explicit target")
    resource_ids: list[str] = Field(
        default_factory=list,
        max_length=32,
        description="Up to 32 unique registered resource IDs to request for this invocation; no paths or owner claims",
    )
    retry_of: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Explicit new attempt linked to a previous invocation; the host validates retry authority",
    )

    @model_validator(mode="after")
    def validate_dispatch(self) -> _DispatchInput:
        if self.runtime is not None and not isinstance(self.runtime, ToolRuntime):
            raise ValueError("runtime must be supplied by the tool execution host")
        DispatchOptions(
            target_policy=self.target_policy,
            device_id=self.device_id,
            target_kind=self.target_kind,
            working_dir=self.working_dir,
            resource_ids=tuple(self.resource_ids),
        )
        return self


class AskAgentInput(_DispatchInput):
    expert: str = Field(
        validation_alias=AliasChoices("expert", "agent"),
        description="Target expert name or short id",
    )
    message: str = Field(description="Task or question for the target expert")
    mode: Literal["sync", "background"] = Field(
        default="sync",
        description=(
            "sync: wait for an immediate answer (default). "
            "background: delegate long work and get a proactive follow-up "
            "(only honored when the host enabled async delivery)."
        ),
    )
    user_question: str | None = Field(
        default=None,
        description="When mode=background, the user's original question for the follow-up reply",
    )


class AskAgentSyncInput(_DispatchInput):
    expert: str = Field(
        validation_alias=AliasChoices("expert", "agent"),
        description="Target expert name or short id",
    )
    message: str = Field(description="Task or question for the target expert")


class AskAgentAsyncInput(_DispatchInput):
    expert: str = Field(
        validation_alias=AliasChoices("expert", "agent"),
        description="Target expert name or short id",
    )
    message: str = Field(description="Task or question for the target expert")
    user_question: str | None = Field(
        default=None,
        description="The user's original question, used when composing the follow-up",
    )


def _tool_ctx() -> tuple[str, str | int, str | None, str | None]:
    cfg = get_config().get("configurable") or {}
    agent_id = cfg.get("agent_id")
    user_id = coerce_user_id(cfg.get("user"))
    if not agent_id:
        raise ValueError("missing configurable.agent_id")
    if user_id is None:
        raise ValueError("missing configurable.user")
    session_key = cfg.get("session_key")
    thread_id = cfg.get("thread_id")
    return (
        str(agent_id),
        user_id,
        str(session_key) if session_key else None,
        str(thread_id) if thread_id else None,
    )


def _ok(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _err(exc: BaseException) -> str:
    return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _peer_payload(peer: PeerResult) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": peer.mode, "agent_id": peer.agent_id, "name": peer.name}
    for key in ("job_id", "thread_id", "response", "status", "message", "invocation_id"):
        value = getattr(peer, key)
        if value is not None and (value != "" or key == "response"):
            payload[key] = value
    if peer.job_id:
        payload["short_id"] = peer.job_id[-6:]
    if peer.receipt is not None:
        payload["dispatch_receipt"] = peer.receipt.to_dict()
    return payload


def _use_async_invoke(
    forced: str,
    mode: Literal["sync", "background"],
    enabled: bool,
) -> bool:
    if forced == "async":
        return True
    if forced == "sync":
        return False
    return mode == "background" and enabled


def _request_peer_invoke_mode() -> PeerInvokeMode | None:
    """Honor a request-scoped ``peer_invoke_mode`` (team member dispatch)."""
    try:
        cfg = get_config().get("configurable") or {}
    except Exception:  # no request config outside a graph run; pylint: disable=broad-except
        return None
    return parse_peer_invoke_mode(cfg.get("peer_invoke_mode"))


def _agent_list_item(
    team: TeamManager,
    entry: AgentEntry,
    language: Language,
) -> dict[str, Any]:
    meta = entry.metadata or {}
    item: dict[str, Any] = {
        "agent_id": entry.agent_id,
        "short_id": entry.agent_id[-6:],
        "name": team.peer_display_name(entry),
        "description": peer_description(meta, language),
    }
    cards = peer_guidance_cards(meta, language)
    if cards:
        item["guidance_cards"] = cards
    # Host-enriched execution defaults are private owner data, including for
    # shared text-only experts. Missing request identity fails closed.
    item.update(default_target=None, model_ref=None)
    owner = meta.get("octop_owner_user_id", meta.get("user_id"))
    try:
        viewer = (get_config().get("configurable") or {}).get("user")
    except RuntimeError:
        viewer = None
    if owner is not None and viewer is not None and str(owner) == str(viewer):
        item["default_target"] = meta.get("default_target")
        item["model_ref"] = meta.get("model_ref")
    return item


def build_team_tools(
    team: TeamManager,
    *,
    peer_invoke_mode: PeerInvokeMode = "both",
) -> list[StructuredTool]:
    """Return ``agent_list`` and ``ask_agent`` tools bound to *team*.

    ``peer_invoke_mode`` selects the invoke surface:
    ``both`` keeps ``mode=sync|background``; ``sync`` / ``async`` hide the
    other path so the model cannot pick the wrong one.
    """

    async def agent_list() -> str:
        try:
            from_agent_id, user_id, _, _ = _tool_ctx()
            peers = team.list_peers(user_id, exclude_agent_id=from_agent_id)
            language = team.caller_language(from_agent_id)
            return _ok([_agent_list_item(team, entry, language) for entry in peers])
        except DEFENSIVE_OP_ERRORS as exc:
            return _err(exc)

    async def ask_agent(
        expert: str,
        message: str,
        mode: Literal["sync", "background"] = "sync",
        user_question: str | None = None,
        target_policy: TargetPolicy = "member_default",
        device_id: str | None = None,
        target_kind: Literal["device", "local"] | None = None,
        working_dir: str | None = None,
        resource_ids: list[str] | None = None,
        retry_of: str | None = None,
        runtime: ToolRuntime[Any, Any] | None = None,
    ) -> str:
        try:
            dispatch_options = DispatchOptions(
                target_policy=target_policy,
                device_id=device_id,
                target_kind=target_kind,
                working_dir=working_dir,
                resource_ids=tuple(resource_ids or ()),
            )
            from_agent_id, user_id, session_key, thread_id = _tool_ctx()
            entry = team.resolve_peer(user_id, expert, exclude_agent_id=from_agent_id)
            if entry is None:
                return _ok({"error": f"expert not found: {expert}"})

            configured = team.peer_invoke_mode_for(from_agent_id) or peer_invoke_mode
            surface = clamp_peer_invoke_mode(peer_invoke_mode, configured)
            forced = clamp_peer_invoke_mode(_request_peer_invoke_mode(), surface)
            use_async = _use_async_invoke(forced, mode, team.enabled)
            if use_async and not team.enabled:
                return _ok({"error": "async peer invoke is not enabled"})
            if use_async:
                peer = await team.asubmit_peer(
                    from_agent_id=from_agent_id,
                    to_agent_id=entry.agent_id,
                    message=message,
                    user_id=user_id,
                    source_thread_id=thread_id,
                    original_user_prompt=user_question,
                    metadata={"session_key": session_key} if session_key else None,
                    dispatch_options=dispatch_options,
                    invocation_key=runtime.tool_call_id if runtime else None,
                    retry_of=retry_of,
                )
            else:
                peer = await team.call_peer(
                    from_agent_id=from_agent_id,
                    to_agent_id=entry.agent_id,
                    message=message,
                    user_id=user_id,
                    source_thread_id=thread_id,
                    session_key=session_key,
                    dispatch_options=dispatch_options,
                    invocation_key=runtime.tool_call_id if runtime else None,
                    retry_of=retry_of,
                )

            return _ok(_peer_payload(peer))
        except PeerDispatchError as exc:
            return _ok(
                {
                    "error": str(exc),
                    "invocation_id": exc.receipt.invocation_id,
                    "dispatch_receipt": exc.receipt.to_dict(),
                }
            )
        except DEFENSIVE_OP_ERRORS as exc:
            return _err(exc)

    if peer_invoke_mode == "async":
        ask_schema: type[BaseModel] = AskAgentAsyncInput
        ask_description = (
            "Dispatch work to another expert asynchronously and return immediately. "
            "When the user writes @Name, prefer this tool. "
            "The teammate works in the background; you will be called again when they finish."
        )
    elif peer_invoke_mode == "sync":
        ask_schema = AskAgentSyncInput
        ask_description = (
            "Collaborate with another expert and wait for their answer. "
            "When the user writes @Name, prefer this tool over answering as yourself."
        )
    else:
        ask_schema = AskAgentInput
        ask_description = (
            "Collaborate with another expert when their expertise fits better than "
            "answering alone. When the user writes @Name (or a similar name), "
            "prefer this tool over answering as yourself. "
            "Approximate names are fine — match the closest teammate. "
            "Use mode=sync for quick tasks. "
            "Use mode=background for slow work (user gets a proactive follow-up)."
        )

    return [
        StructuredTool.from_function(
            coroutine=agent_list,
            name="agent_list",
            description=(
                "List experts owned by the current user (name, description, short id, "
                "and optional guidance cards). "
                "Call before ask_agent when the best collaborator is unclear."
            ),
        ),
        StructuredTool.from_function(
            coroutine=ask_agent,
            name="ask_agent",
            description=ask_description,
            args_schema=ask_schema,
        ),
    ]


__all__ = ["AskAgentAsyncInput", "AskAgentInput", "AskAgentSyncInput", "build_team_tools"]
