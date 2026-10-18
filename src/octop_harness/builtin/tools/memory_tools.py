"""LLM tool functions for the memory system.

These are registered as agent tools at construction time. They use
``langchain_core.tools.tool`` to produce ``StructuredTool`` instances.

The agent exposes search/get plus bounded list/write/update/delete tools.
Writes use the namespace-bound octop-memory fact APIs. Updates create a
successor and deletes soft-retire a fact, preserving history and audit.
Raw transcripts and generated summaries are never rewritten by these tools.

Each tool wraps the underlying service call in a broad ``except`` so that
storage / DB faults degrade into a human-readable string instead of
crashing the agent turn or leaking a stack trace into the conversation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool, tool

if TYPE_CHECKING:
    from octop_memory import MemoryService

logger = logging.getLogger(__name__)


def build_memory_tools(service: MemoryService) -> list[StructuredTool]:
    """Build :class:`MemoryService`-backed LLM tools.

    Search/get resolve recall paths. List/write/update/delete maintain only
    this agent's fact atoms. Mutation requires an exact current assertion
    and a reason; the database namespace is bound by the runtime, never by
    a model-supplied path or agent ID.

    Args:
        service: A :class:`MemoryService` to bind the tools to.
    """

    @tool
    def memory_search(query: str, max_results: int = 5) -> str:
        """Search the agent's memory across atoms, entity pages, raw events.

        Use this when the user references something that happened in a
        previous turn or session, or asks about a fact / preference /
        decision they expect the agent to remember.

        Args:
            query: Natural-language search query.
            max_results: Cap on returned hits (default 5, max ~20).
        """
        try:
            result = service.search(query, max_results=max_results)
        except Exception as exc:  # pragma: no cover - storage / ranker / model fault; pylint: disable=broad-except
            logger.warning("memory_search failed: %s", exc, exc_info=True)
            return f"memory_search failed: {exc.__class__.__name__}"
        hits = result.get("hits", []) if isinstance(result, dict) else []
        if not hits:
            empty_reason = result.get("empty_reason") if isinstance(result, dict) else None
            return f"No matching memory entries. ({empty_reason or 'empty'})"
        lines: list[str] = []
        for h in hits:
            path = h.get("path", "?")
            layer = h.get("layer") or "?"
            snippet = (h.get("snippet") or "").strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            lines.append(f"- [{layer}] {path}\n  {snippet}")
        return "Memory hits:\n" + "\n".join(lines)

    @tool
    def memory_get(path: str, start: int | None = None, lines: int | None = None) -> str:
        """Resolve a virtual memory path to its source text.

        Pass a path returned by ``memory_search`` (e.g.
        ``atom/<id>.md`` / ``page/<entity>.md`` / ``raw/<date>/<id>.md``)
        to read the underlying markdown.

        Args:
            path: The virtual path string from a ``memory_search`` hit.
            start: Optional 1-based line number to start at.
            lines: Optional number of lines to return.
        """
        try:
            result = service.get(path, start=start, lines=lines)
        except Exception as exc:  # pragma: no cover - storage / IO fault; pylint: disable=broad-except
            logger.warning("memory_get failed: %s", exc, exc_info=True)
            return f"memory_get failed: {exc.__class__.__name__}"
        if not isinstance(result, dict):
            return str(result)
        if result.get("error"):
            return f"memory_get error: {result.get('error')}"
        content = result.get("content", result.get("excerpt"))
        if isinstance(content, str):
            return content
        return str(result)

    def manage(action: str, **kwargs: object) -> str:
        # Import only when a tool runs: memory.__init__ also exposes the
        # runtime that imports this module during graph construction.
        from octop_harness.memory.management import _management_result, is_memory_busy

        try:
            return _management_result(service, action, **kwargs)
        except ValueError as exc:
            return f"memory_{action} error: {exc}"
        except Exception as exc:  # pragma: no cover - storage fault; pylint: disable=broad-except
            logger.warning("memory_%s failed", action, exc_info=True)
            busy = is_memory_busy(exc)
            return json.dumps(
                {
                    "status": "error",
                    "code": "MEMORY_BUSY" if busy else "MEMORY_STORAGE_ERROR",
                    "retryable": busy,
                    "error": f"memory_{action} failed: {exc.__class__.__name__}",
                    "next_action": "Read the current fact and retry the correction; do not create a duplicate fact."
                    if busy
                    else "Inspect the storage error before retrying.",
                }
            )

    @tool
    def memory_list(query: str = "", limit: int = 20) -> str:
        """List this agent's current fact atoms, optionally matching query; limit 1..100.

        Returns exact content and atom paths for safe updates. Raw conversations
        and deprecated history remain available through host audit management.
        """
        return manage("list", query=query, limit=limit)

    @tool
    def memory_write(content: str, topic: str, reason: str) -> str:
        """Save a verified durable fact for this agent, with its topic and reason.

        Search first to avoid duplicate or contradictory facts. Do not save
        guesses or transient errors as permanent facts. Use memory_update to
        correct an existing atom and memory_get to verify the returned path.
        """
        return manage("write", content=content, topic=topic, reason=reason)

    @tool
    def memory_update(path: str, content: str, expected_content: str, reason: str) -> str:
        """Correct one atom/<id>.md using the exact current assertion and evidence/reason.

        Creates a successor, retires the old fact, and keeps an audit trail.
        Read the returned successor path to verify. Raw/page paths and another
        agent's records cannot be modified. Do not turn a past event into a
        timeless claim: include the date when a capability changed.
        """
        return manage("update", path=path, content=content, expected_content=expected_content, reason=reason)

    @tool
    def memory_delete(path: str, expected_content: str, reason: str) -> str:
        """Retire one incorrect or obsolete atom/<id>.md; preserve original history.

        Requires its exact current assertion and a reason. Use memory_update
        when a corrected fact exists. This does not erase raw conversation
        history, other agents' facts, or a whole memory store.
        """
        return manage("delete", path=path, expected_content=expected_content, reason=reason)

    return [memory_search, memory_get, memory_list, memory_write, memory_update, memory_delete]  # type: ignore[list-item]


__all__ = ["build_memory_tools"]
