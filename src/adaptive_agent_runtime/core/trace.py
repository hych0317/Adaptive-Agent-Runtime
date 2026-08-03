"""Basic append-only trace recording for Phase 1."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from adaptive_agent_runtime.core.models import RuntimeEvent, TraceEntry


class InMemoryTraceSink:
    """Assign monotonic sequence numbers and retain trace entries in memory."""

    module_id = "trace.in_memory"

    def __init__(self) -> None:
        self._entries: defaultdict[UUID, list[TraceEntry]] = defaultdict(list)

    async def record(self, event: RuntimeEvent) -> TraceEntry:
        run_entries = self._entries[event.run_id]
        entry = TraceEntry(sequence=len(run_entries) + 1, event=event)
        run_entries.append(entry)
        return entry

    def entries_for(self, run_id: UUID) -> tuple[TraceEntry, ...]:
        """Return a run trace for inspection; this is not part of TraceSink."""

        return tuple(self._entries.get(run_id, ()))

