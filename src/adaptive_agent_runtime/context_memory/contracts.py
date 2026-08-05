"""Storage and processing contracts for Context-Memory Runtime."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from uuid import UUID

from adaptive_agent_runtime import RuntimeModule
from adaptive_agent_runtime.context_memory.context_models import (
    ContextArchiveReference,
    ContextAssembly,
    ContextCompressionResult,
    ContextLifecycleDecision,
    ContextLifecycleResult,
    ContextPressure,
    ContextRequirement,
    ContextSchedule,
    ContextUnit,
)
from adaptive_agent_runtime.context_memory.memory_models import (
    MemoryCandidate,
    MemoryRecallQuery,
    MemoryUnit,
    MemoryUpdateResult,
    MemoryBatchWrite,
)


@runtime_checkable
class ContextStore(RuntimeModule, Protocol):
    async def save(
        self,
        unit: ContextUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        """Atomically create, or replace the exact expected resident/tombstone revision."""

        ...

    async def load(self, context_id: UUID) -> ContextUnit | None: ...

    async def delete(
        self,
        context_id: UUID,
        *,
        expected_revision: int,
    ) -> None: ...

    async def list_for_run(self, run_id: UUID) -> tuple[ContextUnit, ...]: ...


@runtime_checkable
class ContextCompressor(RuntimeModule, Protocol):
    async def compress(self, unit: ContextUnit) -> ContextCompressionResult: ...


@runtime_checkable
class ContextArchive(RuntimeModule, Protocol):
    async def archive(
        self,
        unit: ContextUnit,
        *,
        reference: ContextArchiveReference | None = None,
    ) -> ContextArchiveReference: ...

    async def discard(self, reference: ContextArchiveReference) -> None: ...

    async def restore(
        self,
        reference: ContextArchiveReference,
    ) -> ContextUnit: ...

    async def find_latest(
        self,
        context_id: UUID,
    ) -> ContextArchiveReference | None: ...


@runtime_checkable
class ContextPressureMonitoring(RuntimeModule, Protocol):
    def measure(
        self,
        requirement: ContextRequirement,
        resident: Sequence[ContextUnit],
    ) -> ContextPressure: ...


@runtime_checkable
class ContextLifecyclePlanning(RuntimeModule, Protocol):
    def decide(
        self,
        requirement: ContextRequirement,
        resident: Sequence[ContextUnit],
        pressure: ContextPressure,
    ) -> tuple[ContextLifecycleDecision, ...]: ...


@runtime_checkable
class ContextLifecycleActionExecutor(RuntimeModule, Protocol):
    async def execute(
        self,
        decision: ContextLifecycleDecision,
    ) -> ContextLifecycleResult: ...


@runtime_checkable
class ContextLifecycleManagement(RuntimeModule, Protocol):
    async def reconcile(
        self,
        requirement: ContextRequirement,
    ) -> tuple[
        ContextPressure,
        ContextPressure,
        tuple[ContextLifecycleResult, ...],
    ]: ...


@runtime_checkable
class ContextScheduling(RuntimeModule, Protocol):
    def schedule(
        self,
        requirement: ContextRequirement,
        candidates: Sequence[ContextUnit],
    ) -> ContextSchedule: ...


@runtime_checkable
class ContextAssemblyBuilder(RuntimeModule, Protocol):
    def assemble(self, schedule: ContextSchedule) -> ContextAssembly: ...


@runtime_checkable
class MemoryStore(RuntimeModule, Protocol):
    async def save(
        self,
        memory: MemoryUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        """Atomically persist one candidate and enforce expected revision."""

        ...

    async def load(self, memory_id: UUID) -> MemoryUnit | None: ...

    async def load_applied_candidate(
        self,
        candidate_id: UUID,
    ) -> MemoryUnit | None: ...

    async def list_all(self) -> tuple[MemoryUnit, ...]: ...

    async def save_batch(
        self,
        writes: tuple[MemoryBatchWrite, ...],
        *,
        effect_fingerprint: str,
    ) -> tuple[MemoryUnit, ...]: ...

    async def load_applied_effect(
        self,
        effect_fingerprint: str,
    ) -> tuple[MemoryUnit, ...] | None: ...


@runtime_checkable
class MemoryConsolidation(RuntimeModule, Protocol):
    async def consolidate(
        self,
        candidate: MemoryCandidate,
    ) -> MemoryUpdateResult: ...

    async def consolidate_batch(
        self,
        candidates: tuple[MemoryCandidate, ...],
        *,
        effect_fingerprint: str,
    ) -> tuple[MemoryUpdateResult, ...]: ...


@runtime_checkable
class MemoryRecall(RuntimeModule, Protocol):
    async def recall(
        self,
        query: MemoryRecallQuery,
    ) -> tuple[MemoryUnit, ...]: ...
