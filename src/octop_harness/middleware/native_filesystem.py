"""Public filesystem middleware for execution backends using native OS paths."""

from __future__ import annotations

import json
import posixpath
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.tools import ToolException, tool

if TYPE_CHECKING:
    from octop_harness.middleware.filesystem_guard import AgentStoreGuardMiddleware


class NativeFilesystemMiddleware(FilesystemMiddleware):
    """Replace the standard named middleware through deepagents' public merge.

    Native paths are interpreted by the execution backend. They must not pass
    through virtual-path validation, controller expansion, or drive rewriting.
    The inherited middleware still provides tool-result offloading.
    """

    @property
    def name(self) -> str:
        return "FilesystemMiddleware"

    def __init__(self, backend: BackendProtocol, *, storage_guard: AgentStoreGuardMiddleware | None = None) -> None:
        super().__init__(backend=backend, tools=["read_file"])

        def authorize(name: str, path: str) -> None:
            reason = storage_guard.storage_block_reason(name, path) if storage_guard else None
            if reason:
                raise ToolException(reason)

        def filter_paths(result: Any, field: str) -> Any:
            rows = getattr(result, field, None)
            if storage_guard is None or rows is None:
                return result
            visible = [row for row in rows if not storage_guard.storage_block_reason("read_file", row["path"])]
            return replace(result, **{field: visible})

        def render(result: Any) -> str:
            if getattr(result, "error", None):
                raise ToolException(str(result.error))
            return json.dumps(asdict(result), ensure_ascii=False, default=str)

        @tool
        async def ls(path: str = ".") -> str:
            """List a directory on the execution target using a native or relative path."""
            authorize("ls", path)
            return render(filter_paths(await backend.als(path), "entries"))

        @tool
        async def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
            """Read UTF-8 text on the execution target; offset and limit count lines."""
            authorize("read_file", file_path)
            return render(await backend.aread(file_path, offset, limit))

        @tool
        async def write_file(file_path: str, content: str) -> str:
            """Create a text file on the execution target. Use edit_file for an existing file."""
            authorize("write_file", file_path)
            return render(await backend.awrite(file_path, content))

        @tool
        async def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
            """Replace exact text in a file on the execution target, following its native paths."""
            authorize("edit_file", file_path)
            return render(await backend.aedit(file_path, old_string, new_string, replace_all))

        @tool
        async def delete(file_path: str) -> str:
            """Delete a file on the execution target."""
            authorize("delete", file_path)
            if storage_guard and (file_path == "/.agent" or file_path.startswith("/.agent/")):
                parent = posixpath.dirname(file_path.rstrip("/"))
                listing = await backend.als(parent)
                entry = next(
                    (row for row in listing.entries or [] if row["path"].rstrip("/") == file_path.rstrip("/")), None
                )
                if listing.error or entry is None or entry.get("is_dir") is not False:
                    raise ToolException(
                        "Expert storage delete requires a confirmed individual content file. "
                        "Use host management tools to remove a skill or a directory."
                    )
            return render(await backend.adelete(file_path))

        @tool
        async def glob(pattern: str, path: str = ".") -> str:
            """Find paths matching a glob beneath a native execution-target directory."""
            authorize("glob", path)
            return render(filter_paths(await backend.aglob(pattern, path), "matches"))

        @tool
        async def grep(pattern: str, path: str = ".", glob: str | None = None, max_count: int | None = None) -> str:
            """Search literal text on the execution target with an optional glob and per-file match limit."""
            authorize("grep", path)
            return render(filter_paths(await backend.agrep(pattern, path, glob, max_count=max_count), "matches"))

        @tool
        async def execute(command: str, timeout: int = 60) -> str:
            """Execute a short shell command on the execution target. Long-running tasks use Jobs."""
            method = getattr(backend, "aexecute", None)
            if not callable(method):
                raise ToolException("execution backend does not support shell commands")
            return render(await method(command, timeout=timeout))

        self.tools = [ls, read_file, write_file, edit_file, delete, glob, grep, execute]
        for native_tool in self.tools:
            native_tool.handle_tool_error = True
