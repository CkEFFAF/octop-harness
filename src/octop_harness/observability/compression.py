"""Observe summarization at its public model-call boundary, without raw bodies.

The request callback reads a context-local transformation receipt. A smaller
request alone is not evidence of summarization; only a new summary explicitly
marked by the summarizer is reported as such.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage

from octop_harness.context_usage import content_tokens
from octop_harness.middleware.memory_recall import replay_recall_snapshots

request_transformation: ContextVar[dict[str, Any] | None] = ContextVar("request_transformation", default=None)


def _fingerprint(message: BaseMessage) -> str:
    value = {"type": message.type, "content": message.content, "tool_calls": getattr(message, "tool_calls", None)}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _is_summary(message: BaseMessage) -> bool:
    return message.type == "human" and message.additional_kwargs.get("lc_source") == "summarization"


def _tokens(messages: list[Any]) -> int:
    return sum(
        content_tokens(message.content) + content_tokens(getattr(message, "tool_calls", None) or "")
        for message in replay_recall_snapshots(messages)
    )


def _effective_input(request: ModelRequest) -> list[Any]:
    # The documented summarization state keeps the raw transcript intact:
    # effective input is the stored summary followed by the raw retained tail.
    event = request.state.get("_summarization_event")
    if isinstance(event, dict):
        cutoff, summary = event.get("cutoff_index"), event.get("summary_message")
        if isinstance(cutoff, int) and cutoff >= 0 and isinstance(summary, BaseMessage):
            return [summary, *request.messages[cutoff:]]
    return list(request.messages)


def _receipt(before: list[Any], after: list[Any]) -> dict[str, Any] | None:
    before_hashes, after_hashes = [_fingerprint(msg) for msg in before], [_fingerprint(msg) for msg in after]
    before_keys = [(msg.id, digest) for msg, digest in zip(before, before_hashes, strict=True)]
    after_keys = [(msg.id, digest) for msg, digest in zip(after, after_hashes, strict=True)]
    if before_keys == after_keys:
        return None
    previous_summaries = {_fingerprint(msg) for msg in before if _is_summary(msg)}
    summaries = [_fingerprint(msg) for msg in after if _is_summary(msg)]
    new_summary = next((value for value in summaries if value not in previous_summaries), None)
    retained = Counter(after_keys)
    removed = []
    for msg, key in zip(before, before_keys, strict=True):
        if retained[key]:
            retained[key] -= 1
        elif msg.id:
            removed.append(str(msg.id))
    return {
        "kind": "summarization" if new_summary else "message_rewrite",
        "scope": "messages",
        "before_estimated_tokens": _tokens(before),
        "after_estimated_tokens": _tokens(after),
        "before_message_count": len(before),
        "after_message_count": len(after),
        "summary_sha256": new_summary,
        "removed_source_ids": removed[:256],
        "source_ids_truncated": len(removed) > 256,
    }


class ObservedSummarizationMiddleware(SummarizationMiddleware):
    """Keep upstream policy/state while observing actual outgoing transforms."""

    @property
    def name(self) -> str:
        # Replace the default middleware by its public name, retaining its
        # original state schema and all other lifecycle hooks.
        return "SummarizationMiddleware"

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        before = _effective_input(request)

        def observe(current: ModelRequest) -> ModelResponse:
            token = request_transformation.set(_receipt(before, list(current.messages)))
            try:
                return handler(current)
            finally:
                request_transformation.reset(token)

        return super().wrap_model_call(request, observe)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        before = _effective_input(request)

        async def observe(current: ModelRequest) -> ModelResponse:
            token = request_transformation.set(_receipt(before, list(current.messages)))
            try:
                return await handler(current)
            finally:
                request_transformation.reset(token)

        return await super().awrap_model_call(request, observe)


__all__ = ["ObservedSummarizationMiddleware", "request_transformation"]
