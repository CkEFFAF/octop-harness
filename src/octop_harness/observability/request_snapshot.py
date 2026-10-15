"""Provider-bound request lifecycle snapshots through public LangChain callbacks.

The host owns persistence and authorization. These records capture the actual
model input after middleware, with bounded text and references instead of media
payloads, model credentials or transport headers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langgraph.config import get_config

from octop_harness.context_usage import content_tokens, extract_usage_from_message
from octop_harness.middleware.pii import detect_pii
from octop_harness.observability.compression import request_transformation
from octop_harness.observability.sources import message_parts

_TEXT_LIMIT = 16_000
_REQUEST_TEXT_LIMIT = 128_000
_COMPONENT_LIMIT = 256


class _MetadataBudget:
    """Bound every retained metadata leaf, including caller-supplied IDs."""

    def __init__(self) -> None:
        self.remaining = 32_000
        self.nodes = 1024
        self.truncated = False

    def text(self, value: Any, limit: int = 512) -> str | None:
        if value is None:
            return None
        if not isinstance(value, (str, int)):
            self.truncated = True
            return None
        redacted = _redact(str(value))
        retained = redacted[: max(0, min(limit, self.remaining))]
        self.remaining -= len(retained)
        self.truncated |= len(retained) < len(redacted)
        return retained

    def value(self, value: Any, depth: int = 0) -> Any:
        if self.nodes <= 0:
            self.truncated = True
            return None
        self.nodes -= 1
        if value is None or isinstance(value, (bool, int, float, str)):
            return self.text(value) if isinstance(value, str) else value
        if depth >= 4:
            self.truncated = True
            return None
        if isinstance(value, list):
            self.truncated |= len(value) > 256
            return [self.value(item, depth + 1) for item in value[:256]]
        if isinstance(value, dict):
            self.truncated |= len(value) > 32
            result = {}
            for key, item in list(value.items())[:32]:
                if not isinstance(key, str):
                    self.truncated = True
                    continue
                safe_key = _redact(key)
                self.truncated |= len(safe_key) > 64
                result[safe_key[:64]] = self.value(item, depth + 1)
            return result
        self.truncated = True
        return None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text") or "")
            if isinstance(block, dict) and block.get("type") == "text"
            else f"[{block.get('type') or 'media'!s} content omitted]"
            if isinstance(block, dict)
            else str(block)
            for block in content
        )
    return str(content or "")


def _redact(text: str) -> str:
    for match in sorted(detect_pii(text), key=lambda item: item["start"], reverse=True):
        text = text[: match["start"]] + "[redacted]" + text[match["end"] :]
    return text


def _retain(text: str, remaining: int) -> tuple[str, bool, int]:
    limit = max(0, min(_TEXT_LIMIT, remaining))
    # Detect on the complete value: a phone/key/ID can straddle the retention
    # boundary and must not leave an almost-complete secret in history.
    redacted = _redact(text)
    retained = redacted[:limit]
    return retained, len(retained) < len(redacted), remaining - len(retained)


class RequestSnapshotCallback(BaseCallbackHandler):
    """Emit started/completed/failed snapshots to a synchronous host sink.

    Async graph execution invokes this callback in LangChain's callback executor,
    so a host may perform blocking persistence here without blocking streaming.
    A failure in an observational sink does not replay or change model calls.
    """

    def __init__(self, sink: Callable[[dict[str, Any]], None]) -> None:
        self.sink = sink
        self._pending: dict[UUID, tuple[dict[str, Any], float, float | None]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            cfg = get_config().get("configurable") or {}
        except RuntimeError:
            cfg = metadata or {}
        context = cfg.get("octop_execution_context") or {}
        invocation = kwargs.get("invocation_params") or {}
        tools = invocation.get("tools") or []
        components: list[dict[str, Any]] = []
        estimated_by_kind: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        metadata_budget = _MetadataBudget()
        # Preserve routing identity before allocating the shared detail budget.
        retained_metadata = metadata_budget.value(
            {
                "thread_id": cfg.get("thread_id"),
                "agent_id": cfg.get("agent_id"),
                "owner_user_id": cfg.get("user"),
                "turn_id": cfg.get("octop_turn_id"),
                "run_id": cfg.get("octop_task_run_id"),
                "context_id": context.get("context_id"),
                "model": cfg.get("model") or invocation.get("model") or invocation.get("model_name"),
                "parent_run_id": str(kwargs["parent_run_id"]) if kwargs.get("parent_run_id") else None,
                "transformation": request_transformation.get(),
            }
        )
        remaining = _REQUEST_TEXT_LIMIT
        for batch, group in enumerate(messages):
            read_paths: dict[str, str] = {}
            for index, msg in enumerate(group):
                text = _text(msg.content)
                calls = getattr(msg, "tool_calls", None)
                if calls:
                    text += "\n" + json.dumps({"tool_calls": calls}, ensure_ascii=False, default=str)
                    for call in calls:
                        if call.get("name") == "read_file" and isinstance(call.get("args"), dict):
                            path = call["args"].get("file_path")
                            if isinstance(path, str) and isinstance(call.get("id"), str):
                                read_paths[call["id"]] = path
                parts = message_parts(msg, text)
                for segment_index, part in enumerate(parts):
                    if (
                        msg.type == "tool"
                        and getattr(msg, "status", None) != "error"
                        and (path := read_paths.get(str(getattr(msg, "tool_call_id", ""))))
                    ):
                        part.update(source_type="file_tool_result", source_path=path)
                    part_text = part.pop("text")
                    estimate = (
                        content_tokens(msg.content) + (content_tokens(calls) if calls else 0)
                        if len(parts) == 1
                        else content_tokens(part_text)
                    )
                    counts[part["kind"]] += 1
                    estimated_by_kind[part["kind"]] += estimate
                    if len(components) >= _COMPONENT_LIMIT:
                        continue
                    retained, truncated, remaining = _retain(part_text, remaining)
                    components.append(
                        {
                            **part,
                            "source_id": metadata_budget.text(part.get("source_id")),
                            "source_path": metadata_budget.text(part.get("source_path"), 4096),
                            "tool_call_id": metadata_budget.text(getattr(msg, "tool_call_id", None)),
                            "batch": batch,
                            "index": index,
                            "segment_index": segment_index,
                            "estimated_tokens": estimate,
                            "content": retained,
                            "content_truncated": truncated,
                            "sha256": hashlib.sha256(part_text.encode()).hexdigest(),
                        }
                    )
        for index, tool in enumerate(tools):
            text = json.dumps(tool, ensure_ascii=False, default=str)
            estimate = content_tokens(text)
            counts["tool_schema"] += 1
            estimated_by_kind["tool_schema"] += estimate
            if len(components) >= _COMPONENT_LIMIT:
                continue
            retained, truncated, remaining = _retain(text, remaining)
            components.append(
                {
                    "kind": "tool_schema",
                    "index": index,
                    "estimated_tokens": estimate,
                    "content": retained,
                    "content_truncated": truncated,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
        snapshot = {
            **retained_metadata,
            "request_id": str(run_id),
            "status": "started",
            "created_at": time.time(),
            "components": components,
            "component_count": sum(counts.values()),
            "component_counts": dict(counts),
            "components_truncated": sum(counts.values()) > len(components),
            "omitted_component_count": sum(counts.values()) - len(components),
            "estimated_tokens_by_kind": dict(estimated_by_kind),
            "estimated_input_tokens": sum(estimated_by_kind.values()),
            "metadata_truncated": metadata_budget.truncated,
            "actual_usage": None,
            "duration_ms": None,
            "ttft_ms": None,
        }
        with self._lock:
            self._pending[run_id] = (snapshot, time.monotonic(), None)
        self.sink(snapshot)

    def on_llm_new_token(self, token: str | list[str | dict[str, Any]], *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            item = self._pending.get(run_id)
            if item is not None and item[2] is None and token:
                self._pending[run_id] = (*item[:2], time.monotonic())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        usage: dict[str, int] | None = None
        for group in response.generations:
            for generation in group:
                message = getattr(generation, "message", None)
                current = extract_usage_from_message(message) if message is not None else None
                if current is not None:
                    usage = {key: (usage or {}).get(key, 0) + value for key, value in current.items()}
        self._finish(run_id, "completed", actual_usage=usage)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(
            run_id,
            "cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
            error_type=type(error).__name__,
        )

    def _finish(self, request_id: UUID, status: str, **values: Any) -> None:
        with self._lock:
            item = self._pending.pop(request_id, None)
        if item is None:
            return
        snapshot, started, first = item
        metadata_budget = _MetadataBudget()
        retained_values = metadata_budget.value(values)
        self.sink(
            {
                **snapshot,
                **retained_values,
                "metadata_truncated": snapshot["metadata_truncated"] or metadata_budget.truncated,
                "status": status,
                "duration_ms": (time.monotonic() - started) * 1000,
                "ttft_ms": (first - started) * 1000 if first is not None else None,
            }
        )


__all__ = ["RequestSnapshotCallback"]
