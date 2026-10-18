"""Validated dispatch intent and host-authored collaboration receipts.

Options request a target and resource IDs; only the embedding host can authorize
them. A completed transport does not establish a successful task outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

TargetPolicy = Literal["inherit", "member_default", "explicit"]
TransportStatus = Literal["accepted", "running", "completed", "failed", "cancelled", "unknown"]
TaskOutcome = Literal["unknown", "succeeded", "blocked", "failed", "needs_input"]
MAX_DISPATCH_RESOURCES = 32


@dataclass(frozen=True)
class DispatchOptions:
    """Target intent and bounded opaque resource IDs, subject to host authorization."""

    target_policy: TargetPolicy = "member_default"
    device_id: str | None = None
    target_kind: Literal["device", "local"] | None = None
    working_dir: str | None = None
    resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target_policy not in {"inherit", "member_default", "explicit"}:
            raise ValueError("invalid dispatch target_policy")
        if self.target_kind not in {None, "device", "local"}:
            raise ValueError("invalid dispatch target_kind")
        for name, value, maximum in (("device_id", self.device_id, 256), ("working_dir", self.working_dir, 4096)):
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value
            ):
                raise ValueError(f"invalid dispatch {name}")
        if self.target_policy != "explicit":
            if any(value is not None for value in (self.device_id, self.target_kind, self.working_dir)):
                raise ValueError("target fields require target_policy=explicit")
        elif self.target_kind == "local":
            if self.device_id is not None:
                raise ValueError("local dispatch cannot specify device_id")
        elif not self.device_id:
            raise ValueError("explicit dispatch requires device_id or target_kind=local")
        self._validate_resources()

    def _validate_resources(self) -> None:
        if not isinstance(self.resource_ids, tuple) or len(self.resource_ids) > MAX_DISPATCH_RESOURCES:
            raise ValueError(f"resource_ids must be a tuple of at most {MAX_DISPATCH_RESOURCES} IDs")
        for resource_id in self.resource_ids:
            if (
                not isinstance(resource_id, str)
                or not resource_id.strip()
                or len(resource_id) > 256
                or any(char in resource_id for char in ("/", "\\", "\x00"))
            ):
                raise ValueError("resource_ids must contain non-empty opaque IDs, not paths")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise ValueError("resource_ids must be unique")

    def requested_target(self) -> dict[str, Any] | None:
        """Return the explicit requested target, without granting access to it."""
        if self.target_policy != "explicit":
            return None
        return {
            "device_id": self.device_id,
            "target_kind": self.target_kind or "device",
            "working_dir": self.working_dir,
        }


@dataclass(frozen=True)
class DispatchReceipt:
    """Compact invocation metadata; resolved targets and outcomes are host-authored."""

    invocation_id: str
    transport_status: TransportStatus = "accepted"
    task_outcome: TaskOutcome = "unknown"
    target_policy: TargetPolicy = "member_default"
    requested_target: dict[str, Any] | None = None
    resolved_target: dict[str, Any] | None = None
    error_code: str | None = None
    retryable: bool = False
    source_thread_id: str | None = None
    child_thread_id: str | None = None
    root_run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id.strip():
            raise ValueError("receipt requires invocation_id")
        if self.transport_status not in {"accepted", "running", "completed", "failed", "cancelled", "unknown"}:
            raise ValueError("invalid receipt transport_status")
        if self.task_outcome not in {"unknown", "succeeded", "blocked", "failed", "needs_input"}:
            raise ValueError("invalid receipt task_outcome")
        if self.target_policy not in {"inherit", "member_default", "explicit"}:
            raise ValueError("invalid receipt target_policy")

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON-compatible receipt fields."""
        return asdict(self)


class PeerDispatchError(RuntimeError):
    """A peer invocation failed, with the last observable dispatch receipt."""

    def __init__(self, message: str, receipt: DispatchReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


__all__ = ["DispatchOptions", "DispatchReceipt", "PeerDispatchError", "TargetPolicy", "TaskOutcome", "TransportStatus"]
