"""Request-local provenance, matched against the actual provider-bound input."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage

request_sources: ContextVar[tuple[dict[str, str], ...]] = ContextVar("request_sources", default=())


def _sources(request: ModelRequest) -> tuple[dict[str, str], ...]:
    sources: list[dict[str, str]] = []
    state = cast(dict[str, Any], request.state)
    for path, raw in (state.get("memory_contents") or {}).items():
        if isinstance(path, str) and isinstance(raw, str):
            # Deepagents' documented AGENTS.md injection removes author comments.
            content = re.sub(r"<!--[\s\S]*?-->", "", raw).rstrip()
            if content:
                sources.append(
                    {"type": "reference_file", "path": path, "text": f"{path}\n\n{content}", "role": "system"}
                )
    for skill in state.get("skills_metadata") or []:
        if not isinstance(skill, dict) or not all(
            isinstance(skill.get(key), str) for key in ("name", "description", "path")
        ):
            continue
        # Match the actual catalog entry, not every mention of a path/name.
        sources.append(
            {
                "type": "skill_catalog",
                "path": skill["path"],
                "text": f"- **{skill['name']}**: {skill['description']}",
                "role": "system",
            }
        )
    for message in state.get("messages") or []:
        if not isinstance(message, BaseMessage):
            continue
        recall = message.additional_kwargs.get("_octop_memory_recall")
        if isinstance(recall, dict) and isinstance(recall.get("suffix"), str) and recall["suffix"]:
            sources.append(
                {"type": "memory_recall", "path": str(message.id or ""), "text": recall["suffix"], "role": "human"}
            )
    return tuple(sources[:512])


class RequestSourcesMiddleware(AgentMiddleware[Any, Any, Any]):
    """Observe public middleware state without altering messages or provider kwargs."""

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> Any:
        token = request_sources.set(_sources(request))
        try:
            return handler(request)
        finally:
            request_sources.reset(token)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        token = request_sources.set(_sources(request))
        try:
            return await handler(request)
        finally:
            request_sources.reset(token)


def message_parts(message: BaseMessage, text: str) -> list[dict[str, Any]]:
    """Return disjoint slices; attribution never adds the same tokens twice."""
    base = {
        "kind": {"human": "user", "ai": "assistant"}.get(message.type, message.type),
        "source_type": "message",
        "source_id": message.id,
    }
    source = message.additional_kwargs.get("lc_source")
    if message.type == "human" and source == "summarization":
        return [{**base, "kind": "injected_context", "source_type": "conversation_summary", "text": text}]
    if message.type == "human" and source == "pinned_skill":
        skill = message.additional_kwargs.get("skill")
        if isinstance(skill, dict) and isinstance(skill.get("path"), str):
            return [
                {**base, "kind": "injected_context", "source_type": "skill", "source_path": skill["path"], "text": text}
            ]
    spans = []
    for item in request_sources.get():
        if item["role"] != message.type or (item["type"] == "memory_recall" and item["path"] != str(message.id or "")):
            continue
        start = text.find(item["text"])
        if start < 0:
            continue
        end = start + len(item["text"])
        if item["type"] == "skill_catalog":
            # Attribute only if its emitted location accompanies the metadata.
            location = f"  -> Read `{item['path']}` for full instructions"
            location_start = text.find(location, end)
            if location_start < 0 or "\n- **" in text[end:location_start]:
                continue
            end = location_start + len(location)
        spans.append((start, end, item))
    parts, position = [], 0
    for start, end, item in sorted(spans, key=lambda value: value[0]):
        if start < position:
            continue
        if start > position:
            parts.append({**base, "text": text[position:start]})
        parts.append(
            {
                **base,
                "kind": "injected_context",
                "source_type": item["type"],
                "source_path": item["path"],
                "text": text[start:end],
            }
        )
        position = end
    if position < len(text) or not parts:
        parts.append({**base, "text": text[position:]})
    return parts
