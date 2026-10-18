"""Agent-scoped memory maintenance using octop-memory's audited fact APIs.

Raw evidence and old atoms remain available to host audit tools. Retired
evidence is excluded from model recall instead of rewriting conversations.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from octop_memory import MemoryService
from octop_memory.application.path_projection import excerpt_metadata, parse_path, render_atom_md, slice_lines
from octop_memory.pipeline.recall import RecallResult

if TYPE_CHECKING:
    from octop_memory import Memory
    from octop_memory.types import AtomCard

_PREFIX = "harness.memory_correction.v1:"


def is_memory_busy(exc: BaseException) -> bool:
    """SQLite busy/snapshot conflicts are retryable; other storage failures are not."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    code = getattr(exc, "sqlite_errorcode", 0)
    return (code & 0xFF) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or str(exc).lower() in {
        "database is locked",
        "database table is locked",
    }


def _retry_correction(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    # A stale WAL read snapshot cannot be promoted to a writer even with a
    # busy_timeout. Roll back and re-read the exact-content precondition on
    # every attempt, rather than retrying only save_atom inside the old txn.
    delays = (0.02, 0.05, 0.1, 0.2)
    for attempt in range(len(delays) + 1):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if not is_memory_busy(exc) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])
    raise AssertionError("unreachable")


class _MemoryManagement:
    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        # ToolNode runs synchronous tools on worker threads. Bind that
        # thread's connection before opening a nested maintenance transaction:
        # octop-memory1.0.0 initializes its transaction flag on first read.
        self.memory.get_meta(_PREFIX + "connection")

    @staticmethod
    def _text(value: str, name: str, maximum: int = 12000) -> str:
        if not value.strip() or len(value) > maximum:
            raise ValueError(f"{name} must be non-empty and at most {maximum} characters")
        return value.strip()

    def _atom(self, path: str, expected_content: str) -> AtomCard:
        ref = parse_path(path)
        if ref.kind != "atom" or ref.id is None:
            raise ValueError("Only atom/<id>.md facts can be changed; raw history and pages are not editable facts")
        atom = self.memory.get_atom(ref.id)
        if atom is None or atom.deprecated_at is not None:
            raise ValueError("Fact not found or already retired; search or list current facts again")
        if atom.assertion != expected_content:
            raise ValueError("Fact changed since it was read; get the current content before retrying")
        return atom

    @staticmethod
    def _record(atom: AtomCard) -> dict[str, Any]:
        return {
            "path": f"atom/{atom.id}.md",
            "content": atom.assertion,
            "entity_id": atom.entity_id,
            "created_at": atom.created_at.isoformat(),
            "deprecated": atom.deprecated_at is not None,
            "superseded_by": f"atom/{atom.superseded_by}.md" if atom.superseded_by else None,
        }

    def list(self, *, query: str = "", limit: int = 20) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        atoms = (
            self.memory.search_atoms(query, limit=limit + 1)
            if query.strip()
            else self.memory.list_atoms(limit=limit + 1)
        )
        return {"entries": [self._record(a) for a in atoms[:limit]], "has_more": len(atoms) > limit}

    def write(self, content: str, *, topic: str, reason: str) -> dict[str, Any]:
        content = self._text(content, "content")
        topic = self._text(topic, "topic", 200)
        reason = self._text(reason, "reason", 2000)
        atom, _, _ = self.memory.create_atom(content, entity_name=topic, actor="auto", note=reason)
        return {"status": "created", **self._record(atom)}

    def _retire_sources(self, atom: AtomCard) -> None:
        # Namespace-scoped metadata uses existing backend transactions, so a
        # restart or another runtime cannot resurrect this evidence as fact.
        for raw_id in {*atom.raw_event_ids, atom.quote_event_id} - {""}:
            self.memory.set_meta(_PREFIX + "raw:" + raw_id, atom.id)
        self.memory.set_meta(_PREFIX + "page:" + atom.entity_id, atom.id)
        page = self.memory.get_entity_page(atom.entity_id)
        if page is not None:
            # RecallSnippet flattens page_headline to layer=atom and keeps
            # page.id, whereas get_entity_page takes the entity ID.
            self.memory.set_meta(_PREFIX + "page-id:" + page.id, atom.entity_id)

    def update(self, path: str, content: str, *, expected_content: str, reason: str) -> dict[str, Any]:
        content = self._text(content, "content")
        reason = self._text(reason, "reason", 2000)

        def attempt() -> dict[str, Any]:
            with self.memory.backend.transaction():
                old = self._atom(path, expected_content)
                new = self.memory.replace_atom(old.id, assertion=content, actor="auto", note=reason)
                if new is None:
                    raise ValueError("Fact was retired concurrently; search current facts again")
                if new.id != old.id:
                    self._retire_sources(old)
                    self.memory.set_meta(_PREFIX + "successor:" + new.id, old.id)
            return {
                "status": "unchanged" if new.id == old.id else "replaced",
                "previous_path": path,
                **self._record(new),
            }

        return _retry_correction(attempt)

    def delete(self, path: str, *, expected_content: str, reason: str) -> dict[str, Any]:
        reason = self._text(reason, "reason", 2000)

        def attempt() -> dict[str, Any]:
            with self.memory.backend.transaction():
                old = self._atom(path, expected_content)
                if not self.memory.deprecate_atom(old.id, actor="auto", note=reason):
                    raise ValueError("Fact was retired concurrently; search current facts again")
                self._retire_sources(old)
            return {"status": "deprecated", "path": path, "history_preserved": True}

        return _retry_correction(attempt)

    def blocked(self, kind: str, source_id: str) -> str | None:
        if kind == "atom":
            atom = self.memory.get_atom(source_id)
            if atom is not None:
                if atom.deprecated_at is not None:
                    successor = f" Use atom/{atom.superseded_by}.md instead." if atom.superseded_by else ""
                    return "This fact has been retired and is not current evidence." + successor
                return None
            # Recall projects entity page headlines onto the atom layer.
            kind = "page"
        if kind == "page":
            entity_id = self.memory.get_meta(_PREFIX + "page-id:" + source_id) or source_id
            page = self.memory.get_entity_page(entity_id)
            if page and page.dirty and self.memory.get_meta(_PREFIX + "page:" + entity_id):
                return "This summary predates a correction; use current atom facts until it is regenerated."
        if kind == "raw":
            if self.memory.get_meta(_PREFIX + "raw:" + source_id):
                return "This historical source was used by a retired fact; use current atom facts."
            # Episodes are projected onto the raw layer by octop-memory.
            episode = self.memory.get_episode(source_id)
            if episode and any(self.memory.get_meta(_PREFIX + "raw:" + key) for key in episode.raw_event_ids):
                return "This episode uses evidence from a retired fact; use current atom facts."
        return None


class _ManagedMemoryService(MemoryService):  # type: ignore[misc]  # octop-memory ships without py.typed
    """Apply the same freshness checks to explicit tools and automatic recall."""

    def search(
        self, query: str, *, max_results: int = 5, corpus: str = "all", thread_id: str | None = None
    ) -> dict[str, Any]:
        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
        result = super().search(query, max_results=max_results * 3, corpus=corpus, thread_id=thread_id)
        manager = _MemoryManagement(self.memory)
        hits = [
            hit
            for hit in result.get("hits", [])
            if manager.blocked(str(hit.get("layer", "")), str(hit.get("source_id", ""))) is None
        ]
        hits = hits[:max_results]
        return {**result, "hits": hits, "total": len(hits), "empty_reason": None if hits else "no_current_matches"}

    def get(self, path: str, *, start: int | None = None, lines: int | None = None) -> dict[str, Any]:
        ref = parse_path(path)
        reason = _MemoryManagement(self.memory).blocked(ref.kind, ref.id or "")
        if reason:
            return {"path": path, "error": reason}
        if ref.kind == "atom" and ref.id and self.memory.get_meta(_PREFIX + "successor:" + ref.id):
            atom = self.memory.get_atom(ref.id)
            if atom is not None:
                # A successor retains the original quote for host audit. Do
                # not present that superseded quote as the new fact's body.
                body = render_atom_md(replace(atom, verbatim_quote=""))
                return {
                    "path": path,
                    "kind": "atom",
                    "excerpt": slice_lines(body, from_line=start, lines=lines),
                    "metadata": {"historical_sources_preserved": True},
                    **excerpt_metadata(full_text=body, from_line=start, lines=lines),
                }
        return cast(dict[str, Any], super().get(path, start=start, lines=lines))

    def recall(
        self, query: str, *, thread_id: str | None = None, session_id: str | None = None, limit: int = 5
    ) -> RecallResult:
        result = super().recall(query, thread_id=thread_id, session_id=session_id, limit=limit)
        manager = _MemoryManagement(self.memory)
        current = [snippet for snippet in result.snippets if manager.blocked(snippet.layer, snippet.source_id) is None]
        if len(current) == len(result.snippets):
            return result
        # Never reuse the original rendered body after filtering: it embeds
        # the stale text even when its source snippet has been removed.
        rendered = "\n".join(f"- [{snippet.layer}] {snippet.text}" for snippet in current)
        return RecallResult(snippets=current, rendered=rendered)


def _management_result(service: MemoryService, action: str, **kwargs: Any) -> str:
    manager = _MemoryManagement(service.memory)
    if action == "list":
        result = manager.list(**kwargs)
    elif action == "write":
        result = manager.write(**kwargs)
    elif action == "update":
        result = manager.update(**kwargs)
    elif action == "delete":
        result = manager.delete(**kwargs)
    else:
        raise ValueError("Unknown memory action")
    return json.dumps(result, ensure_ascii=False)
