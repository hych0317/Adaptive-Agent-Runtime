"""In-memory inference trace sink for deterministic verification."""

from __future__ import annotations

import asyncio
from uuid import UUID

from adaptive_agent_runtime.llm.gateway.models import (
    InferenceGatewayTraceEntry,
    InferenceGatewayTraceEvent,
)


class InMemoryInferenceGatewayTrace:
    module_id = "llm.inference_trace.in_memory"

    def __init__(self) -> None:
        self._entries: list[InferenceGatewayTraceEntry] = []
        self._lock = asyncio.Lock()

    async def record(
        self,
        event: InferenceGatewayTraceEvent,
    ) -> InferenceGatewayTraceEntry:
        async with self._lock:
            entry = InferenceGatewayTraceEntry(
                sequence=len(self._entries) + 1,
                event=event,
            )
            self._entries.append(entry)
            return entry

    def entries(
        self,
        request_id: UUID | None = None,
        *,
        run_id: UUID | None = None,
        task_id: UUID | None = None,
        node_id: UUID | None = None,
        action_id: UUID | None = None,
    ) -> tuple[InferenceGatewayTraceEntry, ...]:
        return tuple(
            entry
            for entry in self._entries
            if (request_id is None or entry.event.request_id == request_id)
            and (
                run_id is None
                or entry.event.correlation.run_id == run_id
            )
            and (
                task_id is None
                or entry.event.correlation.task_id == task_id
            )
            and (
                node_id is None
                or entry.event.correlation.node_id == node_id
            )
            and (
                action_id is None
                or entry.event.correlation.action_id == action_id
            )
        )
