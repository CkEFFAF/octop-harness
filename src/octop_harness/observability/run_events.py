"""Public tool lifecycle observations correlated with a host run, without tool bodies."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.config import get_config

from octop_harness.observability.request_snapshot import _redact

_SOURCE_TOOLS = frozenset(
    {"web_fetch", "tavily_search", "brave_search", "google_search", "kimi_search", "searchfree_search"}
)
_RESULT_CONTRACT_TOOLS = _SOURCE_TOOLS | {"ask_agent", "task", "browser_use"}


def _result_failed(name: str, output: Any, parsed: Any) -> bool:
    """Respect semantic error envelopes used by existing public tools.

    File contents may legitimately contain error-looking text. Interpret these
    envelopes only for tools whose result contracts define them as errors.
    """
    if getattr(output, "status", None) == "error":
        return True
    if name not in _RESULT_CONTRACT_TOOLS:
        return False
    content = getattr(output, "content", output)
    if isinstance(content, str) and content.lstrip().startswith("Error:"):
        return True
    return isinstance(parsed, dict) and (
        bool(parsed.get("error"))
        or parsed.get("success") is False
        or parsed.get("status") in {"error", "failed", "cancelled"}
    )


def _source_url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            return None
        # Query strings can carry device/provider credentials. Retain the
        # retrieved resource identity without copying those values to history.
        return _redact(urlunsplit((url.scheme, url.netloc, url.path, "", "")))
    except ValueError:
        return None


def _dispatch_status(name: str, parsed: Any, status: str, values: dict[str, Any]) -> str:
    if name != "ask_agent" or not isinstance(parsed, dict):
        return status
    receipt = parsed.get("dispatch_receipt")
    if not isinstance(receipt, dict):
        return status
    for key in ("invocation_id", "task_outcome", "error_code", "retryable", "child_thread_id"):
        if key in receipt:
            values[key] = receipt[key]
    transport = receipt.get("transport_status")
    if transport == "running":
        return "accepted"
    return (
        transport
        if isinstance(transport, str) and transport in {"accepted", "completed", "failed", "cancelled", "unknown"}
        else status
    )


def _retrieved_urls(value: Any) -> list[str]:
    found: set[str] = set()
    pending = [value]
    for _ in range(200):
        if not pending or len(found) >= 20:
            break
        item = pending.pop()
        if isinstance(item, dict):
            for key, entry in list(item.items())[:30]:
                if key in {"url", "link", "source_url"} and (url := _source_url(entry)):
                    found.add(url)
                elif isinstance(entry, (list, dict)):
                    pending.append(entry)
        elif isinstance(item, list):
            pending.extend(item[:30])
    return sorted(found)


class RunEventCallback(BaseCallbackHandler):
    """Send bounded tool/plan facts to a host; transport output remains separate."""

    def __init__(self, sink: Callable[[dict[str, Any]], None]) -> None:
        self.sink = sink
        self._pending: dict[UUID, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            cfg = get_config().get("configurable") or {}
        except RuntimeError:
            cfg = metadata or {}
        context = cfg.get("octop_execution_context") or {}
        values = inputs or {}
        name = _redact(str(serialized.get("name") or "tool"))[:200]
        payload: dict[str, Any] = {"name": name}
        for key in ("file_path", "path", "cwd", "job_id", "expert", "subagent_type"):
            if isinstance(values.get(key), str):
                payload[key] = _redact(values[key])[:4096]
        if name == "write_todos" and isinstance(values.get("todos"), list):
            payload["todos"] = [
                {
                    "content": _redact(str(item.get("content") or ""))[:512],
                    "status": _redact(str(item.get("status") or ""))[:30],
                }
                for item in values["todos"][:32]
                if isinstance(item, dict)
            ]
            payload["truncated"] = len(values["todos"]) > 32
        path = str(values.get("file_path") or values.get("path") or "")
        plan_file = path.startswith("/.agent/plans/")
        if plan_file and isinstance(values.get("content"), str):
            payload["content_sha256"] = hashlib.sha256(values["content"].encode()).hexdigest()
        if name == "web_fetch" and (url := _source_url(values.get("url"))):
            payload["source_urls"] = [url]
        if name == "ask_agent":
            payload["background"] = values.get("mode") == "background" or cfg.get("peer_invoke_mode") == "async"
        event = {
            "event_id": f"tool:{run_id}",
            "run_id": cfg.get("octop_task_run_id"),
            "owner_user_id": cfg.get("user"),
            "thread_id": cfg.get("thread_id"),
            "turn_id": cfg.get("octop_turn_id"),
            "context_id": context.get("context_id"),
            "kind": "plan"
            if name == "write_todos" or plan_file
            else "dispatch"
            if name in {"ask_agent", "task"}
            else "tool",
            "status": "started",
            "created_at": time.time(),
            "payload": {
                **payload,
                "tool_call_id": kwargs.get("tool_call_id"),
                "parent_run_id": str(kwargs["parent_run_id"]) if kwargs.get("parent_run_id") else None,
            },
        }
        with self._lock:
            self._pending[run_id] = event
        self.sink(event)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        values = {"tool_call_id": output.tool_call_id} if getattr(output, "tool_call_id", None) else {}
        with self._lock:
            event = self._pending.get(run_id)
        content = getattr(output, "content", output)
        parsed: Any = content if isinstance(content, dict) else None
        if isinstance(content, str) and len(content) <= 256_000:
            with suppress(ValueError, TypeError):
                parsed = json.loads(content)
        name = event["payload"]["name"] if event else ""
        status = "failed" if _result_failed(name, output, parsed) else "completed"
        if event and status == "completed":
            if event["payload"].get("background") or (
                event["kind"] == "dispatch" and isinstance(parsed, dict) and parsed.get("job_id")
            ):
                status = "accepted"
            if name in _SOURCE_TOOLS:
                urls = set(event["payload"].get("source_urls", [])) | set(_retrieved_urls(parsed))
                values["source_urls"] = sorted(urls)[:20]
        elif name in _SOURCE_TOOLS:
            values["source_urls"] = []
        self._finish(run_id, _dispatch_status(name, parsed, status, values), **values)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(
            run_id,
            "unknown" if isinstance(error, asyncio.CancelledError) else "failed",
            error_type=type(error).__name__,
            source_urls=[],
        )

    def _finish(self, run_id: UUID, status: str, **values: Any) -> None:
        with self._lock:
            event = self._pending.pop(run_id, None)
        if event is not None:
            self.sink(
                {
                    **event,
                    "status": status,
                    "payload": {
                        **event["payload"],
                        **values,
                        "duration_ms": (time.time() - event["created_at"]) * 1000,
                    },
                }
            )


__all__ = ["RunEventCallback"]
