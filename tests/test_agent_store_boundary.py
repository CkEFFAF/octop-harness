"""Expert content permissions are scoped independently of the execution host."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop_harness.backends.execution import StorageExecutionBackend
from octop_harness.backends.workspace import BackendWorkspace
from octop_harness.middleware.filesystem_guard import AgentStoreGuardMiddleware
from octop_harness.middleware.native_filesystem import NativeFilesystemMiddleware


def guard(tmp_path: Path) -> AgentStoreGuardMiddleware:
    store = tmp_path / "controller" / "agent"
    store.mkdir(parents=True, exist_ok=True)
    workspace = BackendWorkspace(
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True), store, system_files_path=".octop"
    )
    return AgentStoreGuardMiddleware(
        [
            FilesystemPermission(
                operations=["read", "write"], paths=["/controller/**", "/**/.env", "/private/**"], mode="deny"
            )
        ],
        workspace=workspace,
        read_roots=["/packages/approved"],
    )


def blocked(middleware: AgentStoreGuardMiddleware, path: str, tool: str = "read_file") -> ToolMessage | None:
    key = "path" if tool in {"ls", "glob", "grep"} else "file_path"
    request = ToolCallRequest(
        tool_call={"name": tool, "args": {key: path}, "id": "test"}, tool=None, state={}, runtime=None
    )
    return middleware._blocked_message(request)


def test_owned_content_uses_logical_policy_and_keeps_virtual_path_in_error(tmp_path: Path) -> None:
    middleware = guard(tmp_path)
    for name in ("MEMORY.md", "AGENTS.md", "USER.md", ".octop/notes/facts.md", ".octop/skills/example/SKILL.md"):
        assert blocked(middleware, "/.agent/controller/agent/" + name) is None
    message = blocked(middleware, "/.agent/controller/agent/private/file")
    assert message is not None
    assert "expert storage policy" in str(message.content)
    assert "/.agent/controller/agent/private/file" in str(message.content)


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".octop/.env",
        ".octop/auth/token",
        ".octop/sessions/chat.jsonl",
        ".octop/memory.sqlite-wal",
        ".octop/checkpoints.sqlite",
        ".octop/.bootstrapped",
        ".octop-auth/token",
    ],
)
@pytest.mark.parametrize("tool", ["read_file", "write_file", "edit_file", "delete"])
def test_runtime_state_and_secrets_are_not_agent_content(tmp_path: Path, name: str, tool: str) -> None:
    assert blocked(guard(tmp_path), "/.agent/controller/agent/" + name, tool) is not None


@pytest.mark.parametrize(
    "path",
    [
        "/.agent/controller/other/MEMORY.md",
        "/.agent/controller/agent-other/MEMORY.md",
        "/.agent/controller/agent/../other/MEMORY.md",
        "/.agent/controller/agent/.octop/../../other/file",
        "/.agent/controller/agent/.octop/notes/../auth/token",
        "/.agent//controller/agent/MEMORY.md",
    ],
)
def test_sibling_and_traversal_paths_are_rejected(tmp_path: Path, path: str) -> None:
    assert blocked(guard(tmp_path), path) is not None


def test_external_skill_mounts_are_explicit_and_read_only(tmp_path: Path) -> None:
    middleware = guard(tmp_path)
    path = "/.agent/packages/approved/example/SKILL.md"
    assert blocked(middleware, path) is None
    assert blocked(middleware, path, "edit_file") is not None
    assert blocked(middleware, "/.agent/packages/unapproved/SKILL.md") is not None


@pytest.mark.parametrize("path", ["/.agent/controller/agent", "/.agent/controller/agent/.octop"])
def test_recursive_search_cannot_scan_internal_runtime_state(tmp_path: Path, path: str) -> None:
    assert blocked(guard(tmp_path), path, "grep") is not None


@pytest.mark.parametrize("path", ["/.agent", "/.agent/", "/.agent/controller/agent", "/.agent/controller/agent/.octop"])
def test_deleting_storage_ancestors_cannot_remove_runtime_state(tmp_path: Path, path: str) -> None:
    assert blocked(guard(tmp_path), path, "delete") is not None


def test_bare_storage_mount_cannot_list_sibling_experts(tmp_path: Path) -> None:
    assert blocked(guard(tmp_path), "/.agent", "ls") is not None


def test_native_project_paths_still_belong_to_execution_policy(tmp_path: Path) -> None:
    middleware = guard(tmp_path)
    assert blocked(middleware, "/home/gpu/project/.env") is None
    assert blocked(middleware, r"D:\project\file") is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink fixture")
def test_storage_symlink_cannot_escape_expert_workspace(tmp_path: Path) -> None:
    middleware = guard(tmp_path)
    other = tmp_path / "controller" / "other"
    other.mkdir()
    (tmp_path / "controller" / "agent" / "linked").symlink_to(other, target_is_directory=True)
    assert blocked(middleware, "/.agent/controller/agent/linked/MEMORY.md") is not None


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink fixture")
def test_symlink_to_internal_auth_is_still_protected(tmp_path: Path) -> None:
    middleware = guard(tmp_path)
    store = tmp_path / "controller" / "agent"
    (store / ".octop" / "auth").mkdir(parents=True)
    (store / "linked").symlink_to(store / ".octop" / "auth", target_is_directory=True)
    assert blocked(middleware, "/.agent/controller/agent/linked/token") is not None


def test_explicit_owned_path_deny_is_preserved(tmp_path: Path) -> None:
    middleware = guard(tmp_path)
    middleware._permissions.insert(
        0, FilesystemPermission(operations=["write"], paths=["/controller/agent/MEMORY.md"], mode="deny")
    )
    assert blocked(middleware, "/.agent/controller/agent/MEMORY.md") is None
    assert blocked(middleware, "/.agent/controller/agent/MEMORY.md", "edit_file") is not None


def test_missing_legacy_skill_root_is_not_advertised(tmp_path: Path) -> None:
    store = tmp_path / "agent"
    (store / ".octop" / "skills").mkdir(parents=True)
    workspace = BackendWorkspace(
        FilesystemBackend(root_dir=store, virtual_mode=True), store, system_files_path=".octop"
    )
    assert "/skills" not in workspace.skill_paths()
    assert "/.octop/skills" in workspace.skill_paths()
    (store / "skills").mkdir()
    assert "/skills" in workspace.skill_paths()


@pytest.mark.skipif(os.name != "nt", reason="Windows case insensitive local paths")
@pytest.mark.parametrize(
    "name",
    [
        ".OCTOP/AUTH/new-token",
        ".OCTOP/MEMORY.SQLITE",
        ".OCTOP-AUTH/token",
        ".octop/memory.sqlite.",
        ".octop/memory.sqlite ",
        ".octop/memory.sqlite::$DATA",
    ],
)
def test_windows_runtime_names_cannot_be_created_with_case_aliases(tmp_path: Path, name: str) -> None:
    assert blocked(guard(tmp_path), "/.agent/controller/agent/" + name, "write_file") is not None


def test_configured_runtime_locations_and_sidecars_are_protected(tmp_path: Path) -> None:
    middleware = guard(tmp_path)
    middleware._protected_paths = ("/controller/agent/state/custom.sqlite", "/controller/agent/transcripts")
    for name in ("state/custom.sqlite", "state/custom.sqlite-wal", "transcripts/chat.jsonl"):
        assert blocked(middleware, "/.agent/controller/agent/" + name) is not None
    assert blocked(middleware, "/.agent/controller/agent/state", "grep") is not None


@pytest.mark.asyncio
async def test_search_results_and_directory_deletes_enforce_descendant_rules(tmp_path: Path) -> None:
    middleware = guard(tmp_path)
    middleware._permissions.insert(
        0, FilesystemPermission(operations=["read", "write"], paths=["/notes/private/**"], mode="deny")
    )
    store = tmp_path / "controller" / "agent"
    (store / "notes" / "private").mkdir(parents=True)
    (store / "notes" / "public.txt").write_text("TOKEN_PUBLIC", encoding="utf-8")
    (store / "notes" / "private" / "secret.txt").write_text("TOKEN_PRIVATE", encoding="utf-8")
    (store / "notes" / ".env").write_text("TOKEN_ENV", encoding="utf-8")
    backend = StorageExecutionBackend(
        FilesystemBackend(root_dir=tmp_path / "project", virtual_mode=True), middleware._workspace.backend
    )
    tools = {tool.name: tool for tool in NativeFilesystemMiddleware(backend, storage_guard=middleware).tools}
    root = "/.agent/controller/agent/notes"
    result = await tools["grep"].ainvoke({"path": root, "pattern": "TOKEN"})
    assert "TOKEN_PUBLIC" in result
    assert "TOKEN_PRIVATE" not in result and "TOKEN_ENV" not in result
    result = await tools["glob"].ainvoke({"path": root, "pattern": "**/*"})
    assert "public.txt" in result and "secret.txt" not in result and "/.env" not in result
    result = await tools["delete"].ainvoke({"file_path": root})
    assert "confirmed individual content file" in result
    assert (store / "notes" / "private" / "secret.txt").exists()
    result = await tools["delete"].ainvoke({"file_path": root + "/public.txt"})
    assert json.loads(result)["error"] is None
    assert not (store / "notes" / "public.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink fixture")
@pytest.mark.asyncio
async def test_python_search_cannot_return_other_expert_through_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    middleware = guard(tmp_path)
    store = tmp_path / "controller" / "agent"
    notes = store / "notes"
    notes.mkdir()
    other = tmp_path / "controller" / "other"
    other.mkdir()
    (other / "secret.txt").write_text("TOKEN_OTHER_EXPERT", encoding="utf-8")
    (notes / "link.txt").symlink_to(other / "secret.txt")
    backend = StorageExecutionBackend(
        FilesystemBackend(root_dir=tmp_path / "project", virtual_mode=True), middleware._workspace.backend
    )
    tools = {tool.name: tool for tool in NativeFilesystemMiddleware(backend, storage_guard=middleware).tools}
    monkeypatch.setattr("deepagents.backends.filesystem.shutil.which", lambda name: None)
    result = await tools["grep"].ainvoke({"path": "/.agent/controller/agent/notes", "pattern": "TOKEN"})
    assert "TOKEN_OTHER_EXPERT" not in result
