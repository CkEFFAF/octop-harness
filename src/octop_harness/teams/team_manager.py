"""Peer discovery and collaboration via :class:`~octop_harness.registry.AgentRegistry`."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from octop_harness.messages import extract_call_response, is_model_retry_failure_text
from octop_harness.registry import AgentEntry, AgentRegistry
from octop_harness.request import ChatRequest
from octop_harness.teams.dispatch import (
    DispatchOptions,
    DispatchReceipt,
    PeerDispatchError,
    TaskOutcome,
    TransportStatus,
)
from octop_harness.teams.inbox import (
    DEFAULT_INBOX_MAX_CONCURRENCY,
    HarnessAgentInboxManager,
    InboxMessage,
    PeerResult,
)
from octop_harness.teams.processor import TeamProcessor
from octop_harness.teams.profile import Language
from octop_harness.teams.tools import build_team_tools
from octop_harness.teams.util import (
    PEER_IDENTITY_CONFIG_KEYS,
    PeerCall,
    PeerInvokeMode,
    PeerSession,
    build_one_shot_request,
    derive_peer_thread_id,
    parse_team_peers,
)

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)
_COMPLETED_DISPATCH_CACHE = 256


def _request_team_peers() -> tuple[str, ...] | None:
    """Read a request-scoped ``team_peers`` allowlist from LangGraph config."""
    try:
        from langgraph.config import get_config

        cfg = get_config().get("configurable") or {}
    except Exception:  # no request config outside a graph run; pylint: disable=broad-except
        return None
    return parse_team_peers(cfg.get("team_peers"))


EnrichRequest = Callable[[str, ChatRequest], Awaitable[ChatRequest]]
AfterCall = Callable[[], Awaitable[None]]
PeerEnrich = Callable[[AgentEntry], None]
PreparePeerSession = Callable[[PeerCall], Awaitable[PeerSession | None]]
AfterPeerCall = Callable[[PeerCall, str, dict[str, Any]], Awaitable[None]]
PeerLifecycle = Callable[[PeerCall, DispatchReceipt], Awaitable[DispatchReceipt | None]]


class TeamManager:
    """Peer listing, sync/async collaboration, and inbox wiring."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry
        self._inbox: HarnessAgentInboxManager | None = None
        self._enrich_request: EnrichRequest | None = None
        self._after_call: AfterCall | None = None
        self._peer_enrich: PeerEnrich | None = None
        self._prepare_peer: PreparePeerSession | None = None
        self._after_peer: AfterPeerCall | None = None
        self._peer_lifecycle: PeerLifecycle | None = None
        self._dispatch_receipts: dict[str, DispatchReceipt] = {}
        self._accepted_replays: set[str] = set()
        self._dispatch_claims: dict[str, tuple[tuple[Any, ...], asyncio.Future[PeerResult]]] = {}
        self._dispatch_calls: dict[str, PeerCall] = {}
        self._dispatch_tasks: dict[str, asyncio.Task[Any]] = {}

    def bind_invocation_hooks(
        self,
        *,
        enrich_request: EnrichRequest | None = None,
        after_call: AfterCall | None = None,
    ) -> None:
        """Optional hooks from :class:`~octop_harness.manager.HarnessAgentManager` (langfuse, etc.)."""
        self._enrich_request = enrich_request
        self._after_call = after_call

    def bind_peer_enrich(self, enrich: PeerEnrich | None) -> None:
        """Host hook to refresh ``AgentEntry.metadata`` before roster / ``agent_list``.

        Called once per returned peer on :meth:`list_peers`. Hosts that snapshot
        description or guidance cards at agent start should re-read them here so
        edits apply without a restart.
        """
        self._peer_enrich = enrich

    def bind_peer_session(
        self,
        *,
        prepare: PreparePeerSession | None = None,
        after: AfterPeerCall | None = None,
        lifecycle: PeerLifecycle | None = None,
    ) -> None:
        """Host hooks to map a peer call onto a durable callee session.

        *prepare* may rewrite ``thread_id`` / ``session_key`` / ``message``
        / ``configurable`` (for example Octop's ``threads`` table and a
        group-chat transcript). Identity keys in ``configurable``
        (``agent_id``, ``user``, ``thread_id``, ``source``) are ignored.
        *after* sees the message actually delivered to the callee. Fallback
        child threads and sessions are isolated per invocation. *lifecycle*
        is awaited before accepted/running execution; failures prevent the
        child call. On accepted, returning a receipt signals a durable replay
        and skips execution. Terminal persistence failures retain the response
        with an unknown receipt, rather than claiming a committed completion.
        """
        self._prepare_peer = prepare
        self._after_peer = after
        self._peer_lifecycle = lifecycle

    # ------------------------------------------------------------------
    # Inbox / processor lifecycle
    # ------------------------------------------------------------------

    def set_processor(
        self,
        processor: TeamProcessor | None,
        *,
        max_concurrency: int = DEFAULT_INBOX_MAX_CONCURRENCY,
    ) -> None:
        """Enable (or disable) inbox-driven async collaboration."""
        if processor is None:
            if self._inbox is not None:
                self._inbox.cancel_worker()
            self._inbox = None
            return
        self._inbox = HarnessAgentInboxManager(
            call_agent=self._call_agent,
            processor=processor,
            invoke_target=self._invoke_inbox_target,
            max_concurrency=max_concurrency,
            observe_terminal=self._observe_inbox_terminal,
        )

    @property
    def enabled(self) -> bool:
        """``True`` when an inbox/TeamProcessor is configured (async available)."""
        return self._inbox is not None

    @property
    def inbox(self) -> HarnessAgentInboxManager | None:
        return self._inbox

    def close(self) -> None:
        if self._inbox is not None:
            self._inbox.cancel_worker()

    async def aclose(self) -> None:
        if self._inbox is not None:
            await self._inbox.shutdown()

    async def acancel_peer(self, invocation_id: str) -> DispatchReceipt | None:
        """Persist a cancellation request before interrupting an invocation.

        Queued work that never entered the child becomes cancelled. Entered
        child execution remains unknown unless its executor confirms it stopped.
        A completed invocation keeps its result when cancellation arrives late.
        """
        call = self._dispatch_calls.get(invocation_id)
        claim = self._dispatch_claims.get(invocation_id)
        receipt = self._dispatch_receipts.get(invocation_id)
        if receipt is None and claim is not None and claim[1].done():
            receipt = claim[1].result().receipt
        if call is None or receipt is None:
            return None
        requested = replace(receipt, error_code="PEER_CANCEL_REQUESTED")
        if self._peer_lifecycle is not None:
            await self._peer_lifecycle(call, requested)
        if receipt.transport_status in {"completed", "failed", "cancelled", "unknown"}:
            return receipt
        task = self._dispatch_tasks.get(invocation_id)
        if call.job_id and self._inbox is not None:
            self._inbox.cancel(call.job_id)
        elif task is not None:
            task.cancel()
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)
        if claim is not None and claim[1].done():
            return claim[1].result().receipt
        return self._dispatch_receipts.get(invocation_id, receipt)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def team_tools(
        self,
        *,
        agent_id: str | None = None,
        peer_invoke_mode: PeerInvokeMode | None = None,
    ) -> list[StructuredTool]:
        """Return ``agent_list`` / ``ask_agent`` bound to this manager.

        Kept for hosts that still append these to ``config.tools``. Agents
        created through ``HarnessAgentManager`` with ``team_enabled`` already
        get the same tools from ``PeerAgentMiddleware`` — do not add both, or the
        model will see duplicates.

        Pass *agent_id* (or an explicit *peer_invoke_mode*) so the schema
        matches that agent's ``peer_invoke_mode`` instead of defaulting to
        ``both``.
        """
        mode = peer_invoke_mode
        if mode is None and agent_id is not None:
            mode = self.peer_invoke_mode_for(agent_id)
        return build_team_tools(self, peer_invoke_mode=mode or "both")

    # ------------------------------------------------------------------
    # Peer discovery
    # ------------------------------------------------------------------

    def list_peers(
        self,
        user_id: str | int,
        *,
        exclude_agent_id: str | None = None,
        team_peers: Sequence[str] | None = None,
    ) -> list[AgentEntry]:
        wanted = str(user_id)
        by_id: dict[str, AgentEntry] = {}
        for entry in self._registry.list():
            meta_uid = entry.metadata.get("user_id")
            if meta_uid is None or str(meta_uid) == wanted:
                by_id[entry.agent_id] = entry
        peers = list(by_id.values())
        if exclude_agent_id is not None:
            peers = [e for e in peers if e.agent_id != exclude_agent_id]
        allowlist = self._peer_allowlist(exclude_agent_id)
        if allowlist is not None:
            peers = self._filter_peers(peers, allowlist)
        override: Sequence[str] | None = team_peers
        if override is None:
            override = _request_team_peers()
        if override is not None:
            names = tuple(str(n).strip().lstrip("@") for n in override if str(n).strip())
            peers = self._filter_peers(peers, names)
        self._apply_peer_enrich(peers)
        return peers

    def peer_invoke_mode_for(self, agent_id: str) -> PeerInvokeMode | None:
        """Return the calling agent's configured ``peer_invoke_mode``, if any."""
        try:
            return self._registry.get(agent_id).config.peer_invoke_mode
        except KeyError:
            return None

    def caller_language(self, from_agent_id: str) -> Language:
        """UI language of the calling agent (``agent_list`` localization)."""
        try:
            lang = self._registry.get(from_agent_id).config.language
        except KeyError:
            return "zh"
        return "en" if lang == "en" else "zh"

    def _apply_peer_enrich(self, peers: list[AgentEntry]) -> None:
        enrich = self._peer_enrich
        if enrich is None:
            return
        for entry in peers:
            try:
                enrich(entry)
            except Exception:  # peer metadata is best-effort; pylint: disable=broad-except
                logger.warning(
                    "peer metadata enrich failed for %s",
                    entry.agent_id,
                    exc_info=True,
                )

    def _peer_allowlist(self, from_agent_id: str | None) -> tuple[str, ...] | None:
        """Return the calling agent's ``team_peers`` allowlist, or ``None`` for all."""
        if not from_agent_id:
            return None
        try:
            entry = self._registry.get(from_agent_id)
        except KeyError:
            return None
        names = entry.config.team_peers
        if names is None:
            return None
        cleaned = tuple(n.strip().lstrip("@") for n in names if str(n).strip())
        return cleaned

    def _filter_peers(self, peers: list[AgentEntry], names: Sequence[str]) -> list[AgentEntry]:
        if not names:
            return []
        out: list[AgentEntry] = []
        seen: set[str] = set()
        for raw in names:
            matched = self._match_peer(peers, raw)
            if matched is not None and matched.agent_id not in seen:
                seen.add(matched.agent_id)
                out.append(matched)
        return out

    def resolve_peer(
        self,
        user_id: str | int,
        query: str,
        *,
        exclude_agent_id: str | None = None,
    ) -> AgentEntry | None:
        rows = self.list_peers(user_id, exclude_agent_id=exclude_agent_id)
        return self._match_peer(rows, query)

    @staticmethod
    def _peer_alias_names(entry: AgentEntry) -> list[str]:
        """Human names plus harness ``config.name`` (memory namespace)."""
        names: list[str] = []
        meta = entry.metadata or {}
        for key in ("display_name", "name"):
            raw = meta.get(key)
            if isinstance(raw, str) and raw.strip():
                names.append(raw.strip())
        cfg_name = entry.config.name if entry.config else ""
        if cfg_name:
            names.append(cfg_name)
        seen: set[str] = set()
        out: list[str] = []
        for name in names:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
        return out

    @classmethod
    def _match_peer(cls, rows: list[AgentEntry], query: str) -> AgentEntry | None:
        q = query.strip().lstrip("@")
        if not q:
            return None
        ql = q.lower()
        for row in rows:
            if row.agent_id == q or row.agent_id.endswith(q):
                return row
        for row in rows:
            if any(name.lower() == ql for name in cls._peer_alias_names(row)):
                return row
        partial = [row for row in rows if any(ql in name.lower() for name in cls._peer_alias_names(row))]
        return partial[0] if len(partial) == 1 else None

    @classmethod
    def peer_display_name(cls, entry: AgentEntry) -> str:
        aliases = cls._peer_alias_names(entry)
        return aliases[0] if aliases else entry.agent_id[-6:]

    def _resolve_peer_entry(
        self,
        from_agent_id: str,
        to_agent_id: str,
        user_id: str | int,
    ) -> AgentEntry:
        if to_agent_id == from_agent_id:
            raise ValueError("cannot collaborate with self")
        entry = self._registry.get(to_agent_id)
        meta_uid = entry.metadata.get("user_id")
        if meta_uid is not None and str(meta_uid) != str(user_id):
            raise ValueError(f"agent {to_agent_id!r} not owned by user")
        if not any(peer.agent_id == to_agent_id for peer in self.list_peers(user_id, exclude_agent_id=from_agent_id)):
            raise ValueError(f"agent {to_agent_id!r} is not an allowed peer")
        return entry

    # ------------------------------------------------------------------
    # Collaboration
    # ------------------------------------------------------------------

    def _claim_dispatch(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        user_id: str | int,
        source_thread_id: str | None,
        session_key: str | None,
        invocation_key: str | None,
        fingerprint: tuple[Any, ...],
    ) -> tuple[str, asyncio.Future[PeerResult], bool]:
        if invocation_key is not None and (
            not isinstance(invocation_key, str) or not invocation_key.strip() or len(invocation_key) > 512
        ):
            raise ValueError("invalid invocation_key")
        scope = json.dumps([str(user_id), from_agent_id, source_thread_id, session_key, invocation_key])
        invocation_id = uuid.uuid5(uuid.NAMESPACE_URL, scope).hex if invocation_key else uuid.uuid4().hex
        existing = self._dispatch_claims.get(invocation_id)
        if existing is not None:
            if existing[0] != fingerprint:
                receipt = DispatchReceipt(
                    invocation_id=invocation_id, transport_status="failed", error_code="INVOCATION_CONFLICT"
                )
                raise PeerDispatchError("Invocation key reused with changed request", receipt)
            return invocation_id, existing[1], True
        future: asyncio.Future[PeerResult] = asyncio.get_running_loop().create_future()
        self._dispatch_claims[invocation_id] = (fingerprint, future)
        return invocation_id, future, False

    def _failed_dispatch_result(self, entry: AgentEntry, invocation_id: str, mode: str) -> PeerResult:
        receipt = self._dispatch_receipts.get(invocation_id)
        return PeerResult(
            mode="background" if mode == "background" else "sync",
            agent_id=entry.agent_id,
            name=self.peer_display_name(entry),
            invocation_id=invocation_id,
            receipt=receipt,
            status=receipt.transport_status if receipt else "unknown",
            thread_id=receipt.child_thread_id if receipt else None,
        )

    def _store_dispatch_result(self, invocation_id: str, result: PeerResult) -> None:
        self._dispatch_claims[invocation_id][1].set_result(result)
        self._prune_dispatch_claims()

    def _prune_dispatch_claims(self) -> None:
        completed = [
            key
            for key, (_, future) in self._dispatch_claims.items()
            if future.done()
            and (receipt := future.result().receipt) is not None
            and receipt.transport_status in {"completed", "failed", "cancelled", "unknown"}
        ]
        for key in completed[:-_COMPLETED_DISPATCH_CACHE]:
            fingerprint, future = self._dispatch_claims[key]
            if self._peer_lifecycle is not None:
                # The host's atomic acceptance remains authoritative after
                # eviction; it returns the persisted receipt on another replay.
                self._dispatch_claims.pop(key)
                self._dispatch_calls.pop(key, None)
            elif future.result().response is not None or future.result().message is not None:
                compact = replace(future.result(), response=None, message=None)
                tombstone: asyncio.Future[PeerResult] = asyncio.get_running_loop().create_future()
                tombstone.set_result(compact)
                self._dispatch_claims[key] = (fingerprint, tombstone)

    async def _call_agent(
        self,
        agent_id: str,
        request: ChatRequest | str | dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve *agent_id* from the registry and invoke ``entry.agent.call``."""
        entry = self._registry.get(agent_id)
        req = ChatRequest.coerce(request)
        if self._enrich_request is not None:
            req = await self._enrich_request(agent_id, req)
        elif not req.agent_id:
            req.agent_id = agent_id
        try:
            return await entry.agent.call(req)
        finally:
            if self._after_call is not None:
                await self._after_call()

    async def _invoke_inbox_target(self, msg: InboxMessage) -> dict[str, Any]:
        sk_raw = msg.metadata.get("session_key") if msg.metadata else None
        sk = sk_raw if isinstance(sk_raw, str) and sk_raw.strip() else None
        _req, result = await self._invoke_peer(
            from_agent_id=msg.source_agent_id,
            to_agent_id=msg.target_agent_id,
            message=msg.message,
            user_id=msg.user_id,
            source="inbox",
            source_thread_id=msg.source_thread_id,
            source_session_key=sk,
            job_id=msg.id,
            invocation_id=msg.invocation_id,
            dispatch_options=msg.dispatch_options,
            accepted=msg.accepted,
            retry_of=msg.retry_of,
        )
        msg.receipt = self._dispatch_receipts.get(msg.invocation_id)
        return result

    def _new_receipt(self, call: PeerCall) -> DispatchReceipt:
        return DispatchReceipt(
            invocation_id=call.invocation_id,
            target_policy=call.dispatch_options.target_policy,
            requested_target=call.dispatch_options.requested_target(),
            source_thread_id=call.source_thread_id,
        )

    async def _observe_peer(
        self, call: PeerCall, receipt: DispatchReceipt, *, terminal: bool = False
    ) -> DispatchReceipt:
        try:
            if self._peer_lifecycle is not None:
                previous = await self._peer_lifecycle(call, receipt)
                if receipt.transport_status == "accepted" and previous is not None:
                    if previous.invocation_id != call.invocation_id or previous.target_policy != receipt.target_policy:
                        raise ValueError("replayed receipt identity or target policy changed")
                    self._accepted_replays.add(call.invocation_id)
                    receipt = previous
        except Exception:
            if not terminal:
                raise
            logger.warning("peer terminal receipt persistence failed id=%s", call.invocation_id, exc_info=True)
            receipt = replace(receipt, transport_status="unknown", error_code="PEER_RECEIPT_PERSISTENCE_FAILED")
        self._dispatch_receipts[call.invocation_id] = receipt
        return receipt

    async def _observe_inbox_terminal(self, msg: InboxMessage, status: str) -> None:
        previous = self._dispatch_receipts.pop(msg.invocation_id, None)
        if not msg.invocation_id:
            return
        if previous is not None and previous.transport_status in {"completed", "failed", "cancelled", "unknown"}:
            msg.receipt = previous
            self._update_background_result(msg)
            return
        call = PeerCall(
            from_agent_id=msg.source_agent_id,
            to_agent_id=msg.target_agent_id,
            user_id=msg.user_id,
            message=msg.message,
            source_thread_id=msg.source_thread_id,
            source_session_key=msg.metadata.get("session_key"),
            job_id=msg.id,
            invocation_id=msg.invocation_id,
            dispatch_options=msg.dispatch_options,
            retry_of=msg.retry_of,
        )
        if previous is None:
            previous = self._new_receipt(call)
        terminal_status: TransportStatus = "cancelled" if status == "cancelled" else "failed"
        msg.receipt = await self._observe_peer(call, replace(previous, transport_status=terminal_status), terminal=True)
        self._dispatch_receipts.pop(msg.invocation_id, None)
        self._update_background_result(msg)

    def _update_background_result(self, msg: InboxMessage) -> None:
        claim = self._dispatch_claims.get(msg.invocation_id)
        if claim is not None and claim[1].done():
            peer = claim[1].result()
            peer.receipt = msg.receipt
            if msg.receipt is not None:
                peer.status = msg.receipt.transport_status
                peer.thread_id = msg.receipt.child_thread_id
        self._prune_dispatch_claims()

    async def _invoke_peer(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        message: str,
        user_id: str | int,
        source: str,
        source_thread_id: str | None,
        source_session_key: str | None,
        job_id: str | None = None,
        invocation_id: str = "",
        dispatch_options: DispatchOptions | None = None,
        accepted: bool = False,
        retry_of: str | None = None,
    ) -> tuple[ChatRequest, dict[str, Any]]:
        call = PeerCall(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            user_id=user_id,
            message=message,
            source_thread_id=source_thread_id,
            source_session_key=source_session_key,
            job_id=job_id,
            invocation_id=invocation_id or uuid.uuid4().hex,
            dispatch_options=dispatch_options or DispatchOptions(),
            retry_of=retry_of,
        )
        self._dispatch_calls[call.invocation_id] = call
        task = asyncio.current_task()
        if task is not None:
            self._dispatch_tasks[call.invocation_id] = task
        receipt = self._dispatch_receipts.get(call.invocation_id) or self._new_receipt(call)
        stage = "PEER_ACCEPTANCE_FAILED"
        try:
            if not accepted:
                receipt = await self._observe_peer(call, receipt)
                if call.invocation_id in self._accepted_replays:
                    self._accepted_replays.discard(call.invocation_id)
                    request = build_one_shot_request(
                        user_id=user_id,
                        agent_id=to_agent_id,
                        text=message,
                        source=source,
                        thread_id=receipt.child_thread_id,
                    )
                    return request, {"dispatch_receipt": receipt.to_dict()}
            stage = "PEER_PREPARATION_FAILED"
            if self._prepare_peer is None and (
                call.dispatch_options.target_policy == "explicit" or call.dispatch_options.resource_ids
            ):
                raise ValueError("target and resource requests require a host preparation hook")
            request = await self._build_peer_request(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                message=message,
                user_id=user_id,
                source=source,
                source_thread_id=source_thread_id,
                source_session_key=source_session_key,
                job_id=job_id,
                invocation_id=call.invocation_id,
                dispatch_options=call.dispatch_options,
                retry_of=retry_of,
            )
            receipt = self._dispatch_receipts.get(call.invocation_id) or receipt
            receipt = replace(receipt, child_thread_id=str(request.thread_id))
            stage = "PEER_START_PERSISTENCE_FAILED"
            receipt = await self._observe_peer(call, replace(receipt, transport_status="running"))
            stage = "PEER_CALL_FAILED"
            result = await self._call_agent(to_agent_id, request)
            payload = result if isinstance(result, dict) else {}
            delivered = request.messages if isinstance(request.messages, str) else message
            if self._after_peer is not None:
                try:
                    await self._after_peer(replace(call, message=delivered), str(request.thread_id or ""), payload)
                except Exception:
                    logger.warning("peer after-call hook failed id=%s", call.invocation_id, exc_info=True)
            receipt = self._result_receipt(receipt, payload)
            await self._observe_peer(call, receipt, terminal=True)
            return request, payload
        except asyncio.CancelledError:
            entered_child = stage == "PEER_CALL_FAILED"
            await self._observe_peer(
                call,
                replace(
                    receipt,
                    transport_status="unknown" if entered_child else "cancelled",
                    error_code="PEER_CANCEL_REQUESTED",
                ),
                terminal=True,
            )
            raise
        except Exception as exc:
            status: TransportStatus = (
                "unknown" if isinstance(exc, TimeoutError) and stage == "PEER_CALL_FAILED" else "failed"
            )
            receipt = await self._observe_peer(
                call, replace(receipt, transport_status=status, error_code=stage), terminal=True
            )
            raise PeerDispatchError(f"Peer dispatch failed: {stage}", receipt) from exc
        finally:
            self._dispatch_tasks.pop(call.invocation_id, None)

    @staticmethod
    def _result_receipt(receipt: DispatchReceipt, result: dict[str, Any]) -> DispatchReceipt:
        receipt = replace(receipt, transport_status="completed")
        if is_model_retry_failure_text(extract_call_response(result)):
            return replace(receipt, transport_status="failed", error_code="PEER_MODEL_FAILED")
        # Only the trusted agent/host structured result supplies a task outcome.
        raw = result.get("dispatch_receipt")
        if isinstance(raw, DispatchReceipt):
            raw = raw.to_dict()
        if isinstance(raw, dict):
            outcome = raw.get("task_outcome", "unknown")
            if isinstance(outcome, str) and outcome in {"unknown", "succeeded", "blocked", "failed", "needs_input"}:
                receipt = replace(receipt, task_outcome=cast(TaskOutcome, outcome))
            if isinstance(raw.get("error_code"), str):
                receipt = replace(receipt, error_code=raw["error_code"])
            if isinstance(raw.get("retryable"), bool):
                receipt = replace(receipt, retryable=raw["retryable"])
        return receipt

    async def _build_peer_request(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        message: str,
        user_id: str | int,
        source: str,
        source_thread_id: str | None,
        source_session_key: str | None,
        job_id: str | None = None,
        invocation_id: str = "",
        dispatch_options: DispatchOptions | None = None,
        retry_of: str | None = None,
    ) -> ChatRequest:
        thread_id = derive_peer_thread_id(source_thread_id, to_agent_id) if source_thread_id else None
        if thread_id and invocation_id:
            thread_id = f"{thread_id}~{invocation_id}"
        session_key = (
            f"{source_session_key}~{invocation_id}" if source_session_key and invocation_id else source_session_key
        )
        extra: dict[str, Any] = {}
        if self._prepare_peer is not None:
            call = PeerCall(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                user_id=user_id,
                message=message,
                source_thread_id=source_thread_id,
                source_session_key=source_session_key,
                job_id=job_id,
                invocation_id=invocation_id,
                dispatch_options=dispatch_options or DispatchOptions(),
                retry_of=retry_of,
            )
            prepared = await self._prepare_peer(call)
            if prepared is not None:
                if prepared.receipt is not None:
                    if prepared.receipt.invocation_id != invocation_id:
                        raise ValueError("prepared receipt invocation identity changed")
                    if prepared.receipt.target_policy != call.dispatch_options.target_policy:
                        raise ValueError("prepared receipt target policy changed")
                    self._dispatch_receipts[invocation_id] = prepared.receipt
                if prepared.thread_id:
                    thread_id = prepared.thread_id
                if prepared.session_key:
                    session_key = prepared.session_key
                if prepared.configurable:
                    extra.update(
                        (key, value)
                        for key, value in prepared.configurable.items()
                        if key not in PEER_IDENTITY_CONFIG_KEYS
                    )
                if prepared.message:
                    message = prepared.message
        return build_one_shot_request(
            user_id=user_id,
            agent_id=to_agent_id,
            text=message,
            source=source,
            thread_id=thread_id,
            session_key=session_key,
            configurable=extra or None,
        )

    async def ask_peer_sync(
        self,
        *,
        from_agent_id: str,
        agent_query: str,
        message: str,
        user_id: str | int,
        source: str = "ask_agent",
        dispatch_options: DispatchOptions | None = None,
        invocation_key: str | None = None,
        retry_of: str | None = None,
    ) -> PeerResult:
        """Sync peer call after resolving *agent_query* — used by ``ask_agent``."""
        entry = self.resolve_peer(user_id, agent_query, exclude_agent_id=from_agent_id)
        if entry is None:
            raise ValueError(f"expert not found: {agent_query}")
        return await self.call_peer(
            from_agent_id=from_agent_id,
            to_agent_id=entry.agent_id,
            message=message,
            user_id=user_id,
            source=source,
            dispatch_options=dispatch_options,
            invocation_key=invocation_key,
            retry_of=retry_of,
        )

    async def call_peer(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        message: str,
        user_id: str | int,
        source: str = "ask_agent",
        source_thread_id: str | None = None,
        session_key: str | None = None,
        dispatch_options: DispatchOptions | None = None,
        invocation_key: str | None = None,
        retry_of: str | None = None,
    ) -> PeerResult:
        """Await a peer, coalescing the same trusted ``invocation_key``.

        Omission creates a new attempt. ``retry_of`` requests a predecessor
        link validated by the host; it never grants authorization itself.
        """
        entry = self._resolve_peer_entry(
            from_agent_id,
            to_agent_id,
            user_id,
        )
        options = dispatch_options or DispatchOptions()
        invocation_id, future, replay = self._claim_dispatch(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            user_id=user_id,
            source_thread_id=source_thread_id,
            session_key=session_key,
            invocation_key=invocation_key,
            fingerprint=(to_agent_id, message, options, retry_of, "sync", source),
        )
        if replay:
            return await asyncio.shield(future)
        try:
            request, result = await self._invoke_peer(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                message=message,
                user_id=user_id,
                source=source,
                source_thread_id=source_thread_id,
                source_session_key=session_key,
                invocation_id=invocation_id,
                dispatch_options=options,
                retry_of=retry_of,
            )
            receipt = self._dispatch_receipts.get(invocation_id)
            peer = PeerResult(
                mode="sync",
                agent_id=to_agent_id,
                name=self.peer_display_name(entry),
                thread_id=receipt.child_thread_id if receipt and receipt.child_thread_id else request.thread_id,
                response=extract_call_response(result),
                invocation_id=invocation_id,
                receipt=receipt,
                status=receipt.transport_status if receipt else None,
            )
            self._store_dispatch_result(invocation_id, peer)
            return peer
        finally:
            if not future.done():
                self._store_dispatch_result(invocation_id, self._failed_dispatch_result(entry, invocation_id, "sync"))
            self._dispatch_receipts.pop(invocation_id, None)

    def submit_peer(self, **kwargs: Any) -> PeerResult:
        """Deprecated: background acceptance requires the awaited :meth:`asubmit_peer`."""
        raise RuntimeError("use await asubmit_peer")

    def _enqueue_peer(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        message: str,
        user_id: str | int,
        source_thread_id: str | None,
        original_user_prompt: str | None = None,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        dispatch_options: DispatchOptions | None = None,
        invocation_id: str,
        retry_of: str | None = None,
    ) -> PeerResult:
        """Enqueue async collaboration; returns inbox id immediately."""
        if self._inbox is None:
            raise RuntimeError("inbox not enabled: construct the manager with a team_processor")
        entry = self._resolve_peer_entry(from_agent_id, to_agent_id, user_id)
        jid = self._inbox.enqueue(
            target_agent_id=to_agent_id,
            source_agent_id=from_agent_id,
            source_thread_id=source_thread_id,
            message=message,
            user_id=user_id,
            original_user_prompt=original_user_prompt,
            job_id=job_id,
            metadata=metadata,
            invocation_id=invocation_id,
            dispatch_options=dispatch_options,
            accepted=True,
            retry_of=retry_of,
        )
        return PeerResult(
            mode="background",
            agent_id=to_agent_id,
            name=self.peer_display_name(entry),
            job_id=jid,
            status="queued",
            invocation_id=invocation_id,
            receipt=self._dispatch_receipts.get(invocation_id)
            or DispatchReceipt(
                invocation_id=invocation_id,
                target_policy=(dispatch_options or DispatchOptions()).target_policy,
                requested_target=(dispatch_options or DispatchOptions()).requested_target(),
            ),
            message=(
                "Background task started. Tell the user work is in progress; "
                "they will receive a follow-up reply when it completes."
            ),
        )

    async def asubmit_peer(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        message: str,
        user_id: str | int,
        source_thread_id: str | None,
        original_user_prompt: str | None = None,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        dispatch_options: DispatchOptions | None = None,
        invocation_key: str | None = None,
        retry_of: str | None = None,
    ) -> PeerResult:
        """Persist acceptance before queueing background work or returning its receipt."""
        if self._inbox is None:
            raise RuntimeError("inbox not enabled: construct the manager with a team_processor")
        entry = self._resolve_peer_entry(from_agent_id, to_agent_id, user_id)
        options = dispatch_options or DispatchOptions()
        invocation_id, future, replay = self._claim_dispatch(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            user_id=user_id,
            source_thread_id=source_thread_id,
            session_key=(metadata or {}).get("session_key"),
            invocation_key=invocation_key,
            fingerprint=(to_agent_id, message, options, retry_of, "background", original_user_prompt),
        )
        if replay:
            return await asyncio.shield(future)
        jid = job_id or uuid.uuid4().hex
        call = PeerCall(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            user_id=user_id,
            message=message,
            source_thread_id=source_thread_id,
            source_session_key=(metadata or {}).get("session_key"),
            job_id=jid,
            invocation_id=invocation_id,
            dispatch_options=options,
            retry_of=retry_of,
        )
        self._dispatch_calls[invocation_id] = call
        receipt = self._new_receipt(call)
        try:
            receipt = await self._observe_peer(call, receipt)
            if invocation_id in self._accepted_replays:
                self._accepted_replays.discard(invocation_id)
                peer = self._failed_dispatch_result(entry, invocation_id, "background")
                self._store_dispatch_result(invocation_id, peer)
                self._dispatch_receipts.pop(invocation_id, None)
                return peer
            peer = self._enqueue_peer(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                message=message,
                user_id=user_id,
                source_thread_id=source_thread_id,
                original_user_prompt=original_user_prompt,
                job_id=jid,
                metadata=metadata,
                dispatch_options=options,
                invocation_id=invocation_id,
                retry_of=retry_of,
            )
            self._store_dispatch_result(invocation_id, peer)
            return peer
        except Exception as exc:
            failed = replace(receipt, transport_status="unknown", error_code="PEER_ACCEPTANCE_FAILED")
            self._dispatch_receipts[invocation_id] = failed
            self._store_dispatch_result(invocation_id, self._failed_dispatch_result(entry, invocation_id, "background"))
            self._dispatch_receipts.pop(invocation_id, None)
            raise PeerDispatchError("Peer dispatch acceptance failed", failed) from exc
        except asyncio.CancelledError:
            unknown = replace(receipt, transport_status="unknown", error_code="PEER_ACCEPTANCE_INTERRUPTED")
            self._dispatch_receipts[invocation_id] = unknown
            self._store_dispatch_result(invocation_id, self._failed_dispatch_result(entry, invocation_id, "background"))
            self._dispatch_receipts.pop(invocation_id, None)
            raise


__all__ = ["TeamManager"]
