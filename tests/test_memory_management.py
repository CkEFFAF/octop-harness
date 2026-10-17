"""Real SQLite coverage for agent-scoped corrections and stale-source recall."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from octop_memory import Memory, MemoryService
from octop_memory.pipeline.recall import RecallResult, RecallSnippet
from octop_memory.types import EntityPage

from octop_harness.builtin.tools.memory_tools import build_memory_tools
from octop_harness.memory.management import _ManagedMemoryService, _MemoryManagement


@pytest.fixture
def memory(tmp_path: Path):
    memory = Memory(namespace="owner", backend_config={"db_path": str(tmp_path / "memory.sqlite")})
    yield memory
    memory.backend.close()


def tools(memory: Memory):
    return {tool.name: tool for tool in build_memory_tools(_ManagedMemoryService(memory))}


def test_fact_lifecycle_is_audited_and_history_preserved(memory: Memory) -> None:
    bound = tools(memory)
    old_text = "Memory files cannot be read or written."
    created = json.loads(
        bound["memory_write"].invoke({"content": old_text, "topic": "GPU", "reason": "old observation"})
    )
    old_id = created["path"][5:-3]
    old = memory.get_atom(old_id)
    assert old is not None
    sources = list(old.raw_event_ids)
    assert old_text in bound["memory_get"].invoke({"path": created["path"]})
    listed = json.loads(bound["memory_list"].invoke({}))
    assert listed["entries"][0]["content"] == old_text

    current = "As of 2026-10-17, controller memory files can be read, edited and read back."
    updated = json.loads(
        bound["memory_update"].invoke(
            {
                "path": created["path"],
                "content": current,
                "expected_content": old_text,
                "reason": "verified file read/edit/readback after deployment",
            }
        )
    )
    assert updated["status"] == "replaced" and updated["path"] != created["path"]
    retired = memory.get_atom(old_id)
    assert retired is not None and retired.deprecated_at and retired.superseded_by == updated["path"][5:-3]
    assert all(memory.get_raw(key) is not None for key in sources)
    assert "retired" in bound["memory_get"].invoke({"path": created["path"]})
    assert current in bound["memory_get"].invoke({"path": updated["path"]})
    assert old_text not in bound["memory_get"].invoke({"path": updated["path"]})
    assert old_id not in [a.id for a in memory.search_atoms("Memory")]
    actions = {row.action for row in memory.list_journal(limit=50)}
    assert {"create", "user_edit"} <= actions

    deleted = json.loads(
        bound["memory_delete"].invoke(
            {"path": updated["path"], "expected_content": current, "reason": "user asked to forget this fact"}
        )
    )
    assert deleted["status"] == "deprecated" and deleted["history_preserved"] is True
    assert json.loads(bound["memory_list"].invoke({}))["entries"] == []
    assert memory.list_atoms(include_deprecated=True)
    assert any(row.action == "deprecate" for row in memory.list_journal(limit=50))


def test_exact_content_precondition_and_namespace_isolation(memory: Memory, tmp_path: Path) -> None:
    other = Memory(namespace="other", backend_config={"db_path": str(tmp_path / "memory.sqlite")})
    try:
        atom, _, _ = other.create_atom("other expert private fact", entity_name="Private")
        owner_tools = tools(memory)
        for action in ["memory_update", "memory_delete"]:
            args = {"path": f"atom/{atom.id}.md", "expected_content": atom.assertion, "reason": "try other namespace"}
            if action == "memory_update":
                args["content"] = "not permitted"
            assert "not found" in owner_tools[action].invoke(args)
        own, _, _ = memory.create_atom("current exact assertion", entity_name="Local")
        result = owner_tools["memory_update"].invoke(
            {
                "path": f"atom/{own.id}.md",
                "content": "replacement",
                "expected_content": "stale content",
                "reason": "must not overwrite",
            }
        )
        assert "changed since" in result
        assert memory.get_atom(own.id).deprecated_at is None
        assert other.get_atom(atom.id).deprecated_at is None
    finally:
        other.backend.close()


@pytest.mark.parametrize("path", ["raw/2026-10-16/event.md", "page/entity.md", "../../memory.sqlite", "/root/file"])
def test_mutations_do_not_accept_raw_pages_or_host_paths(memory: Memory, path: str) -> None:
    result = tools(memory)["memory_update"].invoke(
        {"path": path, "content": "new", "expected_content": "old", "reason": "invalid target"}
    )
    assert "error:" in result
    assert memory.list_atoms() == []


def test_freshness_filter_survives_service_recreation_and_cached_recall(memory: Memory) -> None:
    old, _, _ = memory.create_atom("outdated GPU permission failure", entity_name="GPU")
    manager = _MemoryManagement(memory)
    stale_snippet = RecallSnippet(
        source_id=old.raw_event_ids[0],
        timestamp_iso="2026-10-16T00:00:00Z",
        role_hint="assistant",
        text=old.assertion,
        layer="raw",
    )
    cached = RecallResult(snippets=[stale_snippet], rendered="outdated GPU permission failure")
    updated = manager.update(
        f"atom/{old.id}.md",
        "GPU memory access verified on 2026-10-17",
        expected_content=old.assertion,
        reason="deployment verified",
    )
    service = _ManagedMemoryService(memory)
    with patch.object(MemoryService, "recall", return_value=cached):
        assert service.recall("permission").rendered == ""
    with patch.object(
        MemoryService,
        "search",
        return_value={
            "hits": [
                {"source_id": old.id, "layer": "atom", "snippet": old.assertion},
                {"source_id": old.raw_event_ids[0], "layer": "raw", "snippet": old.assertion},
                {"source_id": updated["path"][5:-3], "layer": "atom", "snippet": updated["content"]},
            ]
        },
    ):
        hits = service.search("permission")["hits"]
        assert len(hits) == 1 and hits[0]["snippet"] == updated["content"]
    assert service.get(f"raw/2026-10-16/{old.raw_event_ids[0]}.md").get("error")


def test_corrected_entity_page_is_hidden_only_until_regenerated(memory: Memory) -> None:
    old, entity, _ = memory.create_atom("old summary assertion", entity_name="GPU")
    now = datetime.now(UTC)
    page = EntityPage(
        id="page-fixture",
        entity_id=entity.id,
        summary_markdown="old summary assertion",
        headline="old summary assertion",
        topics=[],
        dirty=False,
        regen_attempt_count=0,
        summary_version=1,
        last_regen_at=now,
        last_user_edit_at=None,
        created_at=now,
        updated_at=now,
    )
    memory.upsert_entity_page(page)
    # create_atom already owns a page row; upsert preserves that row's ID.
    page = memory.get_entity_page(entity.id)
    assert page is not None and page.id != entity.id
    _MemoryManagement(memory).delete(f"atom/{old.id}.md", expected_content=old.assertion, reason="obsolete")
    service = _ManagedMemoryService(memory)
    assert service.get(f"page/{entity.id}.md").get("error")
    stale = RecallResult(
        snippets=[
            RecallSnippet(
                source_id=page.id,
                timestamp_iso=now.isoformat(),
                role_hint="atom:high",
                text=page.headline,
                layer="atom",
            )
        ],
        rendered=page.headline,
    )
    with patch.object(MemoryService, "recall", return_value=stale):
        assert service.recall("summary").rendered == ""
    with patch.object(
        MemoryService,
        "search",
        return_value={"hits": [{"layer": "atom", "source_id": page.id, "snippet": page.headline}]},
    ):
        assert service.search("summary")["hits"] == []
    memory.upsert_entity_page(replace(page, dirty=False, summary_markdown="fresh summary", headline="fresh summary"))
    assert "fresh summary" in service.get(f"page/{entity.id}.md")["excerpt"]


def test_failed_correction_rolls_back_fact_and_recall_markers(memory: Memory) -> None:
    old, _, _ = memory.create_atom("old fact", entity_name="GPU")
    with (
        patch.object(memory, "set_meta", side_effect=RuntimeError("test storage fault")),
        pytest.raises(RuntimeError, match="storage fault"),
    ):
        _MemoryManagement(memory).update(f"atom/{old.id}.md", "new fact", expected_content=old.assertion, reason="fix")
    assert memory.get_atom(old.id).deprecated_at is None
    assert [a.assertion for a in memory.list_atoms()] == ["old fact"]


def test_tool_input_bounds_do_not_write(memory: Memory) -> None:
    bound = tools(memory)
    assert "error:" in bound["memory_write"].invoke({"content": " ", "topic": "GPU", "reason": "fix"})
    assert "error:" in bound["memory_write"].invoke({"content": "fact", "topic": "GPU", "reason": " "})
    assert "error:" in bound["memory_list"].invoke({"limit": 10000})
    assert memory.list_atoms() == []


def test_real_search_does_not_revive_old_fact_or_raw_after_database_reopen(tmp_path: Path) -> None:
    config = {"db_path": str(tmp_path / "durable.sqlite")}
    memory = Memory(namespace="owner", backend_config=config)
    old, _, _ = memory.create_atom("GPU controller MEMORY.md permission denied", entity_name="GPU")
    service = _ManagedMemoryService(memory)
    assert service.search("MEMORY.md")["hits"]
    changed = _MemoryManagement(memory).update(
        f"atom/{old.id}.md",
        "As of 2026-10-17 GPU controller MEMORY.md read and write verified",
        expected_content=old.assertion,
        reason="observed read, write and readback after deployment",
    )
    memory.backend.close()
    reopened = Memory(namespace="owner", backend_config=config)
    try:
        service = _ManagedMemoryService(reopened)
        hits = service.search("MEMORY.md")["hits"]
        assert any(hit["source_id"] == changed["path"][5:-3] for hit in hits)
        assert all("permission denied" not in hit["snippet"] for hit in hits)
        assert old.assertion not in service.recall("MEMORY.md").rendered
        assert service.get(f"atom/{old.id}.md").get("error")
    finally:
        reopened.backend.close()
