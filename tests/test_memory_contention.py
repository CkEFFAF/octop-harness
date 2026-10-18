"""A busy SQLite correction retries the whole transaction without duplicating facts."""

from __future__ import annotations

import json
import sqlite3

from octop_memory import Memory

from octop_harness.builtin.tools.memory_tools import build_memory_tools
from octop_harness.memory.management import _ManagedMemoryService


def test_snapshot_conflict_retries_atomic_replacement(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite"
    memory = Memory(namespace="owner", backend_config={"db_path": str(path)})
    writer = sqlite3.connect(path)
    try:
        atom, _, _ = memory.create_atom("old capability", entity_name="runtime")
        original = memory.backend.save_atom
        attempts = []

        def save(value):
            attempts.append(value.id)
            if len(attempts) == 1:
                # Commit another writer after the maintenance transaction read
                # its snapshot. SQLite must reject that snapshot's first write.
                writer.execute("CREATE TABLE contention_probe (value TEXT)")
                writer.commit()
            return original(value)

        monkeypatch.setattr(memory.backend, "save_atom", save)
        tools = {tool.name: tool for tool in build_memory_tools(_ManagedMemoryService(memory))}
        reply = json.loads(
            tools["memory_update"].invoke(
                {
                    "path": f"atom/{atom.id}.md",
                    "expected_content": atom.assertion,
                    "content": "verified capability",
                    "reason": "fresh runtime evidence",
                }
            )
        )
        assert reply["status"] == "replaced"
        assert len(attempts) >= 2
        current = memory.list_atoms()
        assert [value.assertion for value in current] == ["verified capability"]
        old = memory.get_atom(atom.id)
        assert old.superseded_by == reply["path"][5:-3]
        assert old.deprecated_at is not None
        assert [row.action for row in memory.list_journal(limit=50)].count("user_edit") == 1
    finally:
        writer.close()
        memory.backend.close()


def test_exhausted_busy_failure_is_typed_and_old_fact_remains_current(tmp_path, monkeypatch):
    memory = Memory(namespace="owner", backend_config={"db_path": str(tmp_path / "memory.sqlite")})
    try:
        atom, _, _ = memory.create_atom("current fact", entity_name="runtime")
        journal_before = memory.list_journal(limit=50)

        def busy(value):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(memory.backend, "save_atom", busy)
        tools = {tool.name: tool for tool in build_memory_tools(_ManagedMemoryService(memory))}
        reply = json.loads(
            tools["memory_update"].invoke(
                {
                    "path": f"atom/{atom.id}.md",
                    "expected_content": atom.assertion,
                    "content": "replacement",
                    "reason": "verified",
                }
            )
        )
        assert reply["status"] == "error" and reply["code"] == "MEMORY_BUSY"
        assert reply["retryable"] is True
        assert memory.get_atom(atom.id).deprecated_at is None
        assert [value.assertion for value in memory.list_atoms()] == [atom.assertion]
        assert memory.list_journal(limit=50) == journal_before
    finally:
        memory.backend.close()


def test_delete_busy_rolls_back_retirement_and_journal(tmp_path, monkeypatch):
    memory = Memory(namespace="owner", backend_config={"db_path": str(tmp_path / "memory.sqlite")})
    try:
        atom, _, _ = memory.create_atom("current fact", entity_name="runtime")
        journal_before = memory.list_journal(limit=50)

        def busy(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        # Fail after deprecate_atom has written both atom and journal: neither
        # may survive rollback, including the final exhausted attempt.
        monkeypatch.setattr(memory, "set_meta", busy)
        tools = {tool.name: tool for tool in build_memory_tools(_ManagedMemoryService(memory))}
        reply = json.loads(
            tools["memory_delete"].invoke(
                {
                    "path": f"atom/{atom.id}.md",
                    "expected_content": atom.assertion,
                    "reason": "verified correction",
                }
            )
        )
        assert reply["code"] == "MEMORY_BUSY"
        assert memory.get_atom(atom.id).deprecated_at is None
        assert memory.list_journal(limit=50) == journal_before
    finally:
        memory.backend.close()


def test_non_busy_storage_failure_is_not_retried(tmp_path, monkeypatch):
    memory = Memory(namespace="owner", backend_config={"db_path": str(tmp_path / "memory.sqlite")})
    try:
        atom, _, _ = memory.create_atom("current fact", entity_name="runtime")
        attempts = []

        def fail(value):
            attempts.append(value.id)
            raise sqlite3.OperationalError("no such table: corrupted_schema")

        monkeypatch.setattr(memory.backend, "save_atom", fail)
        tools = {tool.name: tool for tool in build_memory_tools(_ManagedMemoryService(memory))}
        reply = json.loads(
            tools["memory_update"].invoke(
                {
                    "path": f"atom/{atom.id}.md",
                    "expected_content": atom.assertion,
                    "content": "replacement",
                    "reason": "verified correction",
                }
            )
        )
        assert reply["code"] == "MEMORY_STORAGE_ERROR"
        assert reply["retryable"] is False
        assert len(attempts) == 1
        assert memory.get_atom(atom.id).deprecated_at is None
    finally:
        memory.backend.close()


def test_busy_retry_rechecks_exact_content_after_another_writer(tmp_path, monkeypatch):
    path = str(tmp_path / "memory.sqlite")
    memory = Memory(namespace="owner", backend_config={"db_path": path})
    other = Memory(namespace="owner", backend_config={"db_path": path})
    try:
        atom, _, _ = memory.create_atom("original", entity_name="runtime")
        attempts = []

        def busy(value):
            attempts.append(value.id)
            raise sqlite3.OperationalError("database is locked")

        def concurrent_write(delay):
            other.replace_atom(atom.id, assertion="new verified fact", actor="auto", note="other writer")

        monkeypatch.setattr(memory.backend, "save_atom", busy)
        monkeypatch.setattr("octop_harness.memory.management.time.sleep", concurrent_write)
        tools = {tool.name: tool for tool in build_memory_tools(_ManagedMemoryService(memory))}
        reply = tools["memory_update"].invoke(
            {
                "path": f"atom/{atom.id}.md",
                "expected_content": "original",
                "content": "stale overwrite",
                "reason": "must re-read after rollback",
            }
        )
        assert "already retired" in reply
        assert len(attempts) == 1
        assert [value.assertion for value in memory.list_atoms()] == ["new verified fact"]
    finally:
        other.backend.close()
        memory.backend.close()
