"""Run-scoped guard for repeated governed Tool invocations."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any


_VOLATILE_KEYS = frozenset(
    {
        "action_id",
        "call_key",
        "request_id",
        "trace_id",
        "invocation_id",
        "event_id",
        "timestamp",
        "created_at",
        "updated_at",
    }
)


@dataclass(frozen=True)
class InvocationGuardSnapshot:
    fingerprint: str | None
    consecutive_count: int
    blocked: bool = False


class RunInvocationGuard:
    def __init__(
        self,
        *,
        repeated_invocation_limit: int | None,
        initial_fingerprint: str | None = None,
        initial_count: int = 0,
    ) -> None:
        self._limit = repeated_invocation_limit
        self._fingerprint = initial_fingerprint
        self._count = initial_count
        self._blocked = False
        self._lock = asyncio.Lock()

    async def admit(
        self,
        *,
        capability_id: str,
        provider_id: str,
        arguments: Mapping[str, Any],
        exempt: bool = False,
    ) -> bool:
        fingerprint = tool_execution_fingerprint(
            capability_id=capability_id,
            provider_id=provider_id,
            arguments=arguments,
        )
        async with self._lock:
            count = self._count + 1 if self._fingerprint == fingerprint else 1
            self._fingerprint = fingerprint
            self._count = count
            self._blocked = bool(
                not exempt and self._limit is not None and count >= self._limit
            )
            return not self._blocked

    async def snapshot(self) -> InvocationGuardSnapshot:
        async with self._lock:
            return InvocationGuardSnapshot(
                fingerprint=self._fingerprint,
                consecutive_count=self._count,
                blocked=self._blocked,
            )


def tool_execution_fingerprint(
    *,
    capability_id: str,
    provider_id: str,
    arguments: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        {
            "capability_id": capability_id,
            "provider_id": provider_id,
            "arguments": _remove_volatile(arguments),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _remove_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _remove_volatile(item)
            for key, item in value.items()
            if str(key).lower() not in _VOLATILE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_remove_volatile(item) for item in value]
    return value


_CURRENT_GUARD: ContextVar[RunInvocationGuard | None] = ContextVar(
    "adaptive_agent_runtime_invocation_guard",
    default=None,
)


@contextmanager
def bind_run_invocation_guard(guard: RunInvocationGuard) -> Iterator[None]:
    token: Token[RunInvocationGuard | None] = _CURRENT_GUARD.set(guard)
    try:
        yield
    finally:
        _CURRENT_GUARD.reset(token)


def current_run_invocation_guard() -> RunInvocationGuard | None:
    return _CURRENT_GUARD.get()
