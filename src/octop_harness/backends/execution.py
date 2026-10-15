"""Project execution and explicitly addressed internal agent content."""

from __future__ import annotations

from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import BackendProtocol, GlobResult, GrepResult

AGENT_STORE_PREFIX = "/.agent/"
__all__ = ["AGENT_STORE_PREFIX", "StorageExecutionBackend"]


class StorageExecutionBackend(CompositeBackend):
    """Project tools use execution; /.agent/ addresses stable agent storage.

    This is an internal storage address, not a device working directory or a
    filesystem permission boundary. Command execution delegates to the project
    backend. Bootstrap and persistence continue using BackendWorkspace.
    """

    def __init__(self, execution: BackendProtocol, storage: BackendProtocol) -> None:
        super().__init__(
            default=execution, routes={AGENT_STORE_PREFIX: storage}, artifacts_root=AGENT_STORE_PREFIX.rstrip("/")
        )

    @staticmethod
    def storage_path(path: str) -> str:
        return AGENT_STORE_PREFIX + path.lstrip("/")

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        if path and path.startswith(AGENT_STORE_PREFIX):
            return super().glob(pattern, path)
        return self.default.glob(pattern, path or ".")

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        if path and path.startswith(AGENT_STORE_PREFIX):
            return await super().aglob(pattern, path)
        return await self.default.aglob(pattern, path or ".")

    def grep(
        self, pattern: str, path: str | None = None, glob: str | None = None, *, max_count: int | None = None
    ) -> GrepResult:
        if path and path.startswith(AGENT_STORE_PREFIX):
            return super().grep(pattern, path, glob, max_count=max_count)
        return self.default.grep(pattern, path or ".", glob, max_count=max_count)

    async def agrep(
        self, pattern: str, path: str | None = None, glob: str | None = None, *, max_count: int | None = None
    ) -> GrepResult:
        if path and path.startswith(AGENT_STORE_PREFIX):
            return await super().agrep(pattern, path, glob, max_count=max_count)
        return await self.default.agrep(pattern, path or ".", glob, max_count=max_count)
