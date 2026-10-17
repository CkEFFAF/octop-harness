"""Filesystem tool boundary: path deny rules + outside-root soft failures.

deepagents ``FilesystemMiddleware`` rejects ``permissions`` when the backend
implements ``SandboxBackendProtocol`` (``local_shell``). Harness still needs
deny rules for read/write filesystem tools there.

Separately, deepagents ``FilesystemBackend._resolve_path`` raises ``ValueError``
for paths outside ``root_dir``, but ``read`` / ``write`` / … only catch
``(OSError, RuntimeError)`` — so the exception escapes ToolNode and aborts the
turn. This middleware softens those into ``ToolMessage(status="error")``.
"""

from __future__ import annotations

import logging
import os
import posixpath
import re
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from deepagents.backends.utils import validate_path
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from wcmatch import glob as wcglob

if TYPE_CHECKING:
    from octop_harness.backends.workspace import BackendWorkspace

logger = logging.getLogger(__name__)

# Aligned with deepagents.middleware.filesystem._DEFAULT_FS_TOOL_OPS /
# _check_fs_permission / _FS_WCMATCH_FLAGS (private there; mirrored here).
_FS_TOOL_OPS: dict[str, Literal["read", "write"]] = {
    "ls": "read",
    "read_file": "read",
    "glob": "read",
    "grep": "read",
    "write_file": "write",
    "edit_file": "write",
    "delete": "write",
}
_FS_WCMATCH_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR | wcglob.DOTGLOB

_OUTSIDE_ROOT_REMEDIATION = (
    "The path is outside this agent's storage root. "
    "Use a path under the storage root (workspace-relative), or ask the user to "
    "widen the storage root so it contains this path "
    "(on Windows, set an explicit drive path such as D:/octop-data)."
)

_WINDOWS_ABS_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def is_windows_absolute_path(raw: str) -> bool:
    """True for leftover ``D:\\…`` / ``D:/…`` / UNC host paths."""
    return bool(_WINDOWS_ABS_RE.match(raw.strip()))


def rewrite_legacy_windows_fs_path(
    raw: str,
    *,
    root_dir: str | Path | None = None,
    workspace_dir: str | Path | None = None,
) -> str:
    """Map a leftover Windows absolute path onto the current storage jail.

    Conversation history and memory often keep ``D:\\octop-data\\…`` after the
    backend root moves (or after migrating off Windows). deepagents virtual
    mode treats that drive-letter string as a relative key; pathlib then jumps
    to the other drive and raises ``Path … outside root directory``, which
    used to abort the whole turn as a generic model-call failure.

    When the same suffix exists under *root_dir* or *workspace_dir*, return a
    jail-safe virtual path (``/data/…``). Otherwise return *raw* unchanged so
    the outside-root soft-fail can explain the storage-root mismatch.
    """
    text = raw.strip()
    if not text or not is_windows_absolute_path(text):
        return raw

    for base in (root_dir, workspace_dir):
        mapped = _virtual_path_if_under_base(text, base)
        if mapped is not None:
            return mapped

    posix = text.replace("\\", "/")
    stripped = posix.split(":", 1)[-1].lstrip("/") if ":" in posix[:3] else posix.lstrip("/")
    parts = [part for part in stripped.split("/") if part]
    bases = [Path(item) for item in (root_dir, workspace_dir) if item]
    for index in range(len(parts)):
        suffix = "/".join(parts[index:])
        for base in bases:
            try:
                candidate = (base.expanduser() / suffix).resolve()
            except OSError:
                continue
            if candidate.is_file() or candidate.is_dir():
                return f"/{suffix}"
    return raw


def _virtual_path_if_under_base(windows_path: str, base: str | Path | None) -> str | None:
    if base is None:
        return None
    try:
        resolved_base = Path(base).expanduser().resolve()
        resolved = Path(windows_path).expanduser().resolve()
        rel = resolved.relative_to(resolved_base).as_posix()
    except (OSError, ValueError):
        return None
    return "/" if not rel or rel == "." else f"/{rel}"


def _tool_base_name(name: str) -> str:
    trimmed = name.strip()
    slash = trimmed.rfind("/")
    return trimmed[slash + 1 :] if slash >= 0 else trimmed


def _path_from_tool_args(tool_name: str, params: dict[str, Any]) -> str | None:
    base = _tool_base_name(tool_name)
    if base in {"read_file", "write_file", "edit_file", "delete"}:
        for key in ("file_path", "path"):
            raw = params.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return None
    if base in {"ls", "glob", "grep"}:
        raw = params.get("path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _realpath_if_available(path: str) -> str | None:
    try:
        if os.path.lexists(path):
            return os.path.realpath(path)
    except OSError:
        return None
    return None


def _expand_path_candidates(path: str) -> tuple[str, ...]:
    """Path forms to match against deny/allow patterns.

    Includes the logical path and its realpath (macOS ``/etc`` → ``/private/etc``).
    """
    out: list[str] = [path]
    real = _realpath_if_available(path)
    if real and real not in out:
        out.append(real)
    return tuple(out)


def _expand_patterns(pattern: str) -> tuple[str, ...]:
    """Expand one policy pattern so directory roots are covered.

    ``/etc/**`` must also match ``/etc`` itself (``ls /etc``). When the prefix
    resolves through a symlink (macOS), also match the realpath forms.
    """
    patterns: list[str] = [pattern]
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        if prefix and prefix not in patterns:
            patterns.append(prefix)
        real_prefix = _realpath_if_available(prefix) if prefix else None
        if real_prefix:
            if real_prefix not in patterns:
                patterns.append(real_prefix)
            real_glob = f"{real_prefix}/**"
            if real_glob not in patterns:
                patterns.append(real_glob)
    return tuple(patterns)


def _path_matches_patterns(path: str, patterns: Iterable[str]) -> bool:
    for candidate in _expand_path_candidates(path):
        for pattern in patterns:
            for expanded in _expand_patterns(pattern):
                if wcglob.globmatch(candidate, expanded, flags=_FS_WCMATCH_FLAGS):
                    return True
    return False


def _check_fs_permission(
    rules: list[FilesystemPermission],
    operation: Literal["read", "write"],
    path: str,
) -> Literal["allow", "deny", "interrupt"]:
    """First-match path rule — directory roots and symlink aliases included."""
    for rule in rules:
        if operation not in rule.operations:
            continue
        if _path_matches_patterns(path, rule.paths):
            return rule.mode
    return "allow"


def filesystem_guard_block_reason(
    permissions: list[FilesystemPermission],
    *,
    tool_name: str,
    params: dict[str, Any],
    native_paths: bool = False,
) -> str | None:
    """Return a rejection message when *tool_name* is denied by *permissions*.

    Matched sensitive paths refuse access for read/write filesystem tools
    (``ls``, ``read_file``, ``glob``, ``grep``, ``write_file``, ``edit_file``).
    """
    if not permissions:
        return None
    base = _tool_base_name(tool_name)
    operation = _FS_TOOL_OPS.get(base)
    if operation is None:
        return None
    raw_path = _path_from_tool_args(tool_name, params)
    if raw_path is None:
        return None
    try:
        validated_path = raw_path if native_paths else validate_path(raw_path)
    except ValueError:
        return None
    if _check_fs_permission(permissions, operation, validated_path) == "deny":
        return f"Error: permission denied for {operation} on {validated_path}"
    return None


def is_path_outside_root_error(exc: BaseException) -> bool:
    """True when *exc* is the deepagents virtual-path jail ``ValueError``."""
    if not isinstance(exc, ValueError):
        return False
    lower = str(exc).lower()
    return "outside root directory" in lower or "path traversal not allowed" in lower


def path_outside_root_tool_message(
    *,
    tool_name: str,
    tool_call_id: str,
    exc: BaseException,
) -> ToolMessage:
    """Build a model-visible tool error for an outside-root failure."""
    return ToolMessage(
        content=f"Error: {exc}. {_OUTSIDE_ROOT_REMEDIATION}",
        name=tool_name or "tool",
        tool_call_id=tool_call_id,
        status="error",
    )


def _tool_call_meta(request: ToolCallRequest) -> tuple[str, str]:
    tool_call = request.tool_call
    return str(tool_call.get("name") or ""), str(tool_call.get("id") or "")


class FilesystemGuardMiddleware(AgentMiddleware[Any, Any]):
    """Filesystem tool boundary: optional deny rules + outside-root soft-fail.

    Always safe to mount with an empty permission list (soft-fail only).
    Outside-root softening applies only to filesystem tools (``ls``,
    ``read_file``, …), not arbitrary tool ValueErrors.
    """

    def __init__(
        self,
        permissions: list[FilesystemPermission] | None = None,
        *,
        root_dir: str | Path | None = None,
        workspace_dir: str | Path | None = None,
    ) -> None:
        self._permissions = list(permissions or ())
        self._root_dir = root_dir
        self._workspace_dir = workspace_dir

    def _rewrite_legacy_windows_request(self, request: ToolCallRequest) -> ToolCallRequest:
        tool_name, _ = _tool_call_meta(request)
        raw_args: Any = request.tool_call.get("args") or {}
        path = (
            _path_from_tool_args(tool_name, raw_args)
            if _tool_base_name(tool_name) in _FS_TOOL_OPS and isinstance(raw_args, dict)
            else None
        )
        rewritten = (
            rewrite_legacy_windows_fs_path(
                path,
                root_dir=self._root_dir,
                workspace_dir=self._workspace_dir,
            )
            if path
            else None
        )
        if not path or rewritten is None or rewritten == path:
            return request
        new_args: dict[str, Any] = dict(raw_args)
        key = next((name for name in ("file_path", "path") if new_args.get(name) == path), None)
        if key is None:
            return request
        new_args[key] = rewritten
        logger.info("FilesystemGuard rewrote leftover Windows path %s -> %s", path, rewritten)
        new_call: ToolCall = {
            "name": str(request.tool_call.get("name") or ""),
            "args": new_args,
            "id": request.tool_call.get("id"),
        }
        if request.tool_call.get("type") == "tool_call":
            new_call["type"] = "tool_call"
        override = getattr(request, "override", None)
        if callable(override):
            updated = override(tool_call=new_call)
            if isinstance(updated, ToolCallRequest):
                return updated
        request.tool_call = new_call
        return request

    def _blocked_message(self, request: ToolCallRequest) -> ToolMessage | None:
        tool_name, call_id = _tool_call_meta(request)
        raw_args: Any = request.tool_call.get("args") or {}
        params: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        reason = filesystem_guard_block_reason(
            self._permissions,
            tool_name=tool_name,
            params=params,
        )
        if reason is None:
            return None
        logger.info("FilesystemGuard blocked %s", tool_name)
        return ToolMessage(
            content=reason,
            name=tool_name or "tool",
            tool_call_id=call_id,
            status="error",
        )

    def _soften_outside_root(
        self,
        request: ToolCallRequest,
        exc: ValueError,
    ) -> ToolMessage | None:
        if not is_path_outside_root_error(exc):
            return None
        tool_name, call_id = _tool_call_meta(request)
        if _tool_base_name(tool_name) not in _FS_TOOL_OPS:
            return None
        logger.info("FilesystemGuard softened outside-root %s: %s", tool_name, exc)
        return path_outside_root_tool_message(
            tool_name=tool_name,
            tool_call_id=call_id,
            exc=exc,
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        request = self._rewrite_legacy_windows_request(request)
        blocked = self._blocked_message(request)
        if blocked is not None:
            return blocked
        try:
            return await handler(request)
        except ValueError as exc:
            softened = self._soften_outside_root(request, exc)
            if softened is not None:
                return softened
            raise

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        request = self._rewrite_legacy_windows_request(request)
        blocked = self._blocked_message(request)
        if blocked is not None:
            return blocked
        try:
            return handler(request)
        except ValueError as exc:
            softened = self._soften_outside_root(request, exc)
            if softened is not None:
                return softened
            raise


class AgentStoreGuardMiddleware(FilesystemGuardMiddleware):
    """Authorize expert content separately from the selected device's OS.

    Storage keys can include a controller host path such as /root/agents/id.
    That prefix identifies the expert, not the device user's permissions.
    Rules inside the owned store use workspace relative paths. Explicit skill
    source roots are readable, but do not grant write access outside the store.
    Runtime databases, sessions and authentication remain host managed.
    """

    def __init__(
        self,
        permissions: list[FilesystemPermission] | None = None,
        *,
        workspace: BackendWorkspace | None = None,
        read_roots: Iterable[str] = (),
        runtime_paths: Iterable[str] = (),
    ) -> None:
        super().__init__(permissions)
        self._workspace = workspace
        from octop_harness.backends.utils import is_local_class_backend

        backend: Any = workspace.backend if workspace else None
        while getattr(backend, "default", None) is not None:
            backend = backend.default
        self._case_insensitive = os.name == "nt" and is_local_class_backend(backend)
        self._store_root = workspace._deepagents_source_path(".").replace("\\", "/").rstrip("/") if workspace else ""
        self._read_roots = tuple(root.replace("\\", "/").rstrip("/") for root in read_roots)
        self._protected_paths = tuple(path.replace("\\", "/").rstrip("/") for path in runtime_paths)
        prefixes = {""}
        if workspace and workspace.system_files_path:
            prefixes.add("/" + workspace.system_files_path.strip("/"))
        self._runtime_dirs = (
            "/.octop-auth",
            *(prefix + name for prefix in prefixes for name in ("/auth", "/sessions")),
        )
        self._runtime_files = tuple(
            prefix + name for prefix in prefixes for name in ("/.bootstrapped", "/checkpoints.sqlite", "/memory.sqlite")
        )
        if self._case_insensitive:
            self._runtime_dirs = tuple(path.casefold() for path in self._runtime_dirs)
            self._runtime_files = tuple(path.casefold() for path in self._runtime_files)

    def _relative(self, path: str, root: str) -> str | None:
        if not root:
            return path
        compared_path = path.casefold() if self._case_insensitive else path
        compared_root = root.casefold() if self._case_insensitive else root
        if compared_path == compared_root:
            return "/"
        if compared_path.startswith(compared_root + "/"):
            return path[len(root) :]
        return None

    def _resolved_relative(self, path: str, root: str) -> str | None:
        """Check symlinks against both the store boundary and logical rules."""
        resolver = getattr(self._workspace.backend, "_resolve_path", None) if self._workspace else None
        if not callable(resolver):
            return self._relative(path, root)
        try:
            relative = Path(resolver(path)).resolve().relative_to(Path(resolver(root or "/")).resolve())
        except (OSError, ValueError, RuntimeError):
            return None
        return "/" if relative == Path(".") else "/" + relative.as_posix()

    def _runtime_path(self, path: str, *, recursive: bool) -> bool:
        if self._case_insensitive:
            path = path.casefold()
        if any(path == root or path.startswith(root + "/") for root in self._runtime_dirs):
            return True
        if any(path == name or path.startswith(name + "-") for name in self._runtime_files):
            return True
        return recursive and any(root.startswith(path.rstrip("/") + "/") for root in self._runtime_dirs)

    def _protected_storage_path(self, path: str, *, recursive: bool) -> bool:
        if self._case_insensitive:
            path = path.casefold()
        for configured_root in self._protected_paths:
            root = configured_root.casefold() if self._case_insensitive else configured_root
            if path == root or path.startswith((root + "/", root + "-")):
                return True
            if recursive and root.startswith(path.rstrip("/") + "/"):
                return True
        return False

    def _rewrite_legacy_windows_request(self, request: ToolCallRequest) -> ToolCallRequest:
        return request

    def _content_policy_reason(self, name: str, logical: str, resolved: str) -> str | None:
        candidates = {logical, resolved}
        if self._case_insensitive and any(
            part.endswith((".", " ")) or ":" in part for candidate in candidates for part in candidate.split("/")
        ):
            return "Windows storage paths must not use trailing dots, spaces or alternate data stream aliases"
        if any(
            self._runtime_path(candidate, recursive=_tool_base_name(name) in {"glob", "grep", "delete"})
            for candidate in candidates
        ):
            return (
                "host managed authentication, sessions and memory databases are not editable content; "
                "use memory tools or a content directory"
            )
        # Preserve explicit rules written against this store's full key,
        # while broad host paths (for example /root/**) stay host scoped.
        rules = [
            FilesystemPermission(
                operations=rule.operations,
                paths=[
                    (self._relative(pattern, self._store_root) or pattern).casefold()
                    if self._case_insensitive
                    else self._relative(pattern, self._store_root) or pattern
                    for pattern in rule.paths
                ],
                mode=rule.mode,
            )
            for rule in self._permissions
        ]
        for candidate in candidates:
            reason = filesystem_guard_block_reason(
                rules,
                tool_name=name,
                params={
                    "path": candidate.casefold() if self._case_insensitive else candidate,
                    "file_path": candidate.casefold() if self._case_insensitive else candidate,
                },
                native_paths=True,
            )
            if reason:
                return reason
        return None

    def storage_block_reason(self, name: str, path: str) -> str | None:
        """Authorize one model visible content path, including search results."""
        from octop_harness.backends.execution import AGENT_STORE_PREFIX

        if path != AGENT_STORE_PREFIX.rstrip("/") and not path.startswith(AGENT_STORE_PREFIX):
            return None
        storage_path = path[len(AGENT_STORE_PREFIX) - 1 :].replace("\\", "/") or "/"
        if self._workspace is None:
            return filesystem_guard_block_reason(
                self._permissions,
                tool_name=name,
                params={"path": storage_path, "file_path": storage_path},
                native_paths=True,
            )

        reason = None
        logical = self._relative(storage_path, self._store_root)
        root = self._store_root
        if (
            ".." in storage_path.split("/")
            or storage_path.startswith("//")
            or posixpath.normpath(storage_path) != (storage_path.rstrip("/") or "/")
        ):
            reason = "path must be normalized and stay inside this expert's storage"
        elif logical is None:
            source = next(
                (source for source in self._read_roots if self._relative(storage_path, source) is not None), None
            )
            if source is None or _FS_TOOL_OPS.get(_tool_base_name(name)) != "read":
                reason = "path is outside this expert's writable storage and approved skill sources"
            else:
                root, logical = source, self._relative(storage_path, source)
        resolved = self._resolved_relative(storage_path, root) if reason is None else None
        if reason is None and resolved is None:
            reason = "storage path resolves outside its approved root"
        if (
            reason is None
            and resolved is not None
            and any(
                self._protected_storage_path(candidate, recursive=_tool_base_name(name) in {"glob", "grep", "delete"})
                for candidate in (storage_path, root + resolved)
            )
        ):
            reason = "path contains configured host runtime persistence"
        if reason is None and logical is not None and resolved is not None:
            reason = self._content_policy_reason(name, logical, resolved)
        if reason is not None:
            reason = (
                f"Error: expert storage policy denied {name} on {path}: {reason}. "
                "This is controller storage, not a device OS permission error."
            )
        return reason

    def _blocked_message(self, request: ToolCallRequest) -> ToolMessage | None:
        name, call_id = _tool_call_meta(request)
        path = _path_from_tool_args(name, dict(request.tool_call.get("args") or {}))
        reason = self.storage_block_reason(name, path) if path else None
        return ToolMessage(content=reason, name=name, tool_call_id=call_id, status="error") if reason else None


__all__ = [
    "FilesystemGuardMiddleware",
    "filesystem_guard_block_reason",
    "is_path_outside_root_error",
    "is_windows_absolute_path",
    "path_outside_root_tool_message",
    "rewrite_legacy_windows_fs_path",
]
